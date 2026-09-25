"""
Dataset + collator.

Fixes relative to v1:
  * NEVER truncates one side of a pair. v1 cropped mel at max_audio_length and phonemes at
    max_text_length independently, producing pairs whose audio lacks words the text has
    (or the reverse) — exactly the data that teaches a model to skip or cut off.
    Over-length or impossible utterances are DROPPED at manifest time and reported.
  * strict symbol lookup (unknown symbol -> error with the utterance id), language IDs per
    token, speaker/boundary/rate conditioning carried through to the batch.
  * mel targets are normalised with corpus statistics (as in Matcha-TTS) and padded with 0
    after normalisation (padding is masked everywhere anyway).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from ..text.phonemizer import CzechSlovakPhonemizer, tokens_to_ids
from .audio import AudioProcessor

log = logging.getLogger(__name__)

BOUNDARY_IDS = {"mid": 0, "para_end": 1, "chapter_end": 2}


@dataclass
class FilterReport:
    kept: int
    too_long: List[str]
    too_short: List[str]
    impossible_alignment: List[str]

    def summary(self) -> str:
        return (f"kept {self.kept}; dropped {len(self.too_long)} over-length, "
                f"{len(self.too_short)} too short, {len(self.impossible_alignment)} with fewer "
                f"frames than tokens")


def load_manifest(path: str) -> List[Dict]:
    """JSONL, one utterance per line. Required: id, audio, text, lang. See the plan, Step 4."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def filter_manifest(rows: Sequence[Dict], n_tokens: Dict[str, int], hop_s: float,
                    max_duration_s: float = 20.0, min_duration_s: float = 0.5,
                    min_frames_per_token: float = 1.0) -> Tuple[List[Dict], FilterReport]:
    """
    Drop (never crop) utterances that are too long for the GPU budget, too short to be useful,
    or have fewer mel frames than tokens (MAS cannot give every token >= 1 frame).
    `n_tokens` maps utterance id -> token count from the phonemizer.
    """
    kept, too_long, too_short, impossible = [], [], [], []
    for r in rows:
        dur = float(r["duration_s"])
        frames = int(dur / hop_s)
        if dur > max_duration_s:
            too_long.append(r["id"])
        elif dur < min_duration_s:
            too_short.append(r["id"])
        elif frames < min_frames_per_token * n_tokens[r["id"]]:
            impossible.append(r["id"])
        else:
            kept.append(r)
    report = FilterReport(len(kept), too_long, too_short, impossible)
    log.info(report.summary())
    return kept, report


class TTSDataset(Dataset):
    def __init__(self, rows: Sequence[Dict], audio: AudioProcessor, vocab: Dict[str, int],
                 phonemizers: Dict[str, CzechSlovakPhonemizer], speaker_ids: Dict[str, int],
                 mel_mean: float = 0.0, mel_std: float = 1.0, mel_cache_dir: Optional[str] = None,
                 default_rates: Tuple[float, float] = (12.0, 0.1)):
        """default_rates: used until the Step 4.4 labels exist; pass (config rate_mean, mean pause
        ratio) so the tempo input is constant and carries no information during bootstrapping."""
        self.rows = list(rows)
        self.default_rates = default_rates
        self.audio = audio
        self.vocab = vocab
        self.g2p = phonemizers
        self.speaker_ids = speaker_ids
        self.mel_mean, self.mel_std = mel_mean, mel_std
        self.cache = Path(mel_cache_dir) if mel_cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.rows)

    def _mel(self, row: Dict) -> torch.Tensor:
        if self.cache:
            p = self.cache / f"{row['id']}.pt"
            if p.exists():
                return torch.load(p)
        mel = self.audio.compute_mel(self.audio.load_audio(row["audio"]))
        if self.cache:
            torch.save(mel, p)
        return mel

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        tokens = self.g2p[row["lang"]].tokenize(row["text"])  # text is ALREADY normalised
        ids, lang_ids = tokens_to_ids(tokens, self.vocab, strict=True, context=f"utterance {row['id']}")
        mel = (self._mel(row) - self.mel_mean) / self.mel_std
        if mel.shape[1] < len(ids):
            raise ValueError(f"{row['id']}: {mel.shape[1]} frames < {len(ids)} tokens; run filter_manifest")
        return {
            "id": row["id"],
            "phoneme_ids": torch.tensor(ids, dtype=torch.long),
            "lang_ids": torch.tensor(lang_ids, dtype=torch.long),
            "mel": mel,
            "speaker_id": self.speaker_ids[row.get("speaker", "narrator")],
            "boundary_id": BOUNDARY_IDS[row.get("boundary", "mid")],
            # utterance-level tempo labels (phones/s without pauses, pause share), Step 4.4
            "rate": torch.tensor([row.get("articulation_rate", self.default_rates[0]),
                                  row.get("pause_ratio", self.default_rates[1])]),
        }


class TTSCollator:
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        B = len(batch)
        t_len = torch.tensor([b["phoneme_ids"].numel() for b in batch])
        m_len = torch.tensor([b["mel"].shape[1] for b in batch])
        n_mels = batch[0]["mel"].shape[0]
        ids = torch.zeros(B, int(t_len.max()), dtype=torch.long)       # <pad> = 0
        langs = torch.zeros(B, int(t_len.max()), dtype=torch.long)
        mels = torch.zeros(B, n_mels, int(m_len.max()))                 # normalised mel, pad 0
        for i, b in enumerate(batch):
            ids[i, : t_len[i]] = b["phoneme_ids"]
            langs[i, : t_len[i]] = b["lang_ids"]
            mels[i, :, : m_len[i]] = b["mel"]
        return {
            "ids": [b["id"] for b in batch],
            "phoneme_ids": ids, "lang_ids": langs, "phoneme_lengths": t_len,
            "mels": mels, "mel_lengths": m_len,
            "speaker_ids": torch.tensor([b["speaker_id"] for b in batch]),
            "boundary_ids": torch.tensor([b["boundary_id"] for b in batch]),
            "rates": torch.stack([b["rate"] for b in batch]),
        }


def corpus_mel_stats(dataset: TTSDataset, max_items: int = 2000) -> Tuple[float, float]:
    """Scalar mean/std of log-mel over (a sample of) the training set, as in Matcha-TTS."""
    s, s2, n = 0.0, 0.0, 0
    for i in range(min(len(dataset), max_items)):
        m = dataset._mel(dataset.rows[i])
        s += m.sum().item(); s2 += (m ** 2).sum().item(); n += m.numel()
    mean = s / n
    return mean, (s2 / n - mean ** 2) ** 0.5
