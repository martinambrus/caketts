# Czech/Slovak TTS System: Implementation Plan v2
## Matcha-TTS core · multilingual SSL discriminator · BigVGAN v2

**Target languages:** Slovak, Czech, English (one narrator's voice)
**Data:** 10–40 h of the narrator, plus optional public cs/sk/en speech for pretraining
**Architecture:** non-autoregressive; explicit per-phoneme durations (Monotonic Alignment Search) + OT-CFM decoder on 100-band mel + BigVGAN v2
**Hard requirements:** (1) no skipped or repeated words, gibberish or cut-offs; (2) the same pace within a text and across texts
**Hallucination risk:** structurally minimised by explicit durations, then *gated*: every rendered sentence is checked by ASR (Step 12). v1's "zero, guaranteed by NAR design" was overstated.

**Version:** v2, 25 September 2026. Supersedes v1 (November 2025). Architecture changes follow the research report *TTS breakthroughs since late 2025*.

---

## What changed in v2

### The seven v1 code defects: fixed, each guarded by a test

| # | v1 defect | What it would have caused | v2 fix | Test that guards it |
|---|---|---|---|---|
| 1 | Length regulator: `round()` then dropped tokens with 0 frames (`torch.round` rounds half to even) | Every token predicted at ≤ 0.5 frames vanished: skipped phonemes, and skipped words when several fall in a row | `ceil(exp(logw) · scale)`, clamped to ≥ 1 frame per real token **after** any scaling | `test_no_token_is_dropped` |
| 2 | Dataset cropped mel and phonemes independently (latent: both limits defaulted to `None`) | Misaligned pairs that teach the model to skip and cut off | Over-length and impossible utterances are **dropped** at manifest time and reported; nothing is ever cropped | `test_filter_drops_instead_of_cropping`, `test_items_are_never_truncated` |
| 3 | Flow matching trained with data at t=0 and noise at t=1, but the sampler started from noise at t=0 and integrated forward | The sampler ran the data→noise flow and could only output noise | Matcha-TTS convention: noise at t=0, data at t=1, target `x1 − (1−σ_min)·z` | `test_sampler_transports_noise_to_data`, `test_overfits_one_utterance` |
| 4 | Hand-written phoneme list with a silent `<unk>` fallback; stress marks and punctuation glued onto phones; espeak-ng code `en` crashes | 41–62% of real Czech/Slovak tokens and 88% of English tokens became `<unk>` | Vocabulary built from real espeak-ng output; stress, word boundaries and punctuation are separate tokens; `en` → `en-us`; unknown symbols raise | `test_tokens_are_atomic`, `test_every_training_token_is_known`, `test_unknown_symbol_raises` |
| 5 | `_apply_custom_pronunciation` was a stub | Lexicon fixes for names and espeak-ng errors silently did nothing | Word-level lexicon looked up before espeak-ng | `test_lexicon_override_is_applied` |
| 6 | 80 HTK-scale, unnormalised mel bands with `center=True` | Features the BigVGAN checkpoint was never trained on (it expects 100 Slaney-normalised bands) | BigVGAN's exact mel computation | `test_identical_to_bigvgan_mel`, `test_mel_basis_matches_bigvgan_training` |
| 7 | Trim at −40 dB below the loudest frame with no padding; the "LUFS" step was peak normalisation | Soft onsets and word endings cut off; clip edges varied by session | Bounds from forced alignment or the recording's noise floor, fixed 150 ms pads; EBU R128 per source file | `test_quiet_onset_is_not_cut`, `test_reaches_target_lufs` |

### Further defects found while fixing

| v1 issue | Consequence | v2 fix |
|---|---|---|
| `DurationPredictor` applied `nn.LayerNorm(hidden)` to a `(B, hidden, T)` tensor | Crashes whenever `T ≠ hidden` | Channel LayerNorm, masking, detached encoder input |
| The learned `AlignmentModule` had no loss training it, and `MatchaTTS` / `compute_total_loss` were referenced but never specified | MAS would have aligned on random scores | Matcha-TTS Gaussian-prior MAS with prior, duration and CFM losses (Step 8) |
| The U-Net doubled channels per level; with the config's `n_layers: 6` that is an 8,192-channel bottleneck | About a billion parameters and out-of-memory | Fixed width per level |
| The text encoder added positional encodings to padding and ran unmasked convolutions | A sentence's encoding depended on batch padding | Masking everywhere; exact padding-invariance test |
| MAS was a pure-Python double loop | Seconds per batch | numba DP, verified against exhaustive search |
| `torchaudio.load` / `save` | Need TorchCodec since torchaudio 2.9 | soundfile I/O |
| espeak-ng 1.52 errors the plan assumed away | "rozhlas" → [roshlas]; Slovak ô split; inconsistent voicing across word boundaries; stress on the second half of Slovak diphthongs ("viem" → v i ˈe m) | Post-rules and a stress policy (Step 3), each with a test |
| Unnormalised digits would be dropped silently by a tokenizer | A skipped word | The tokenizer raises on digits and symbols |

### Architecture changes from the research

| Component | v1 | v2 | Confidence |
|---|---|---|---|
| Speaker conditioning | ECAPA-TDNN vectors via FiLM | Learned speaker table (the narrator is one row); speaker-verification models are used only for evaluation | Medium |
| SLM discriminator | WavLM-Large, "works for ANY language" (it was pretrained on English only) | XLS-R-300M by default (Czech and Slovak in its pretraining data); ablate against Omnilingual wav2vec 2.0, w2v-BERT 2.0 and no SSL discriminator | High that WavLM is English-only; Low on the size of the gain |
| Tempo | Implicit | Articulation-rate and pause-ratio inputs to the duration predictor, fixed at inference; paragraph/chapter boundary flag; pauses inserted by rule | Medium |
| Text encoder | Phones only | Plus a per-token language ID (cs/sk/en) | Medium |
| G2P | espeak-ng as-is | Plus lexicon, post-rules, stress policy and strict vocabulary | High |
| Decoder training | CFM only | Optional repeat/skip negatives and condition dropout (CFG, self-purifying filtering) | Medium |
| Training | Acoustic stage, then joint training with SLM | Multilingual pretraining → narrator fine-tune → vocoder fine-tune → short, late adversarial phase → RL last | Medium |
| Quality control | One line: "hallucination detection via ASR" | ASR verify-and-retry on every sentence with relative thresholds, failure-shape detectors, and pace metrics M1–M6 | High that it is needed |
| Data | ParCzech4Speech, Common Voice | Plus SloPalSpeech (2,806 h Slovak, CC BY 4.0) and ParlaSpeech-CZ (1,218 h), re-filtered | Medium |

### Text normalisation: num2words rewritten (Step 2.1)

Both number modules were rewritten from published grammar. v1 built every declined cardinal from ordinal stems (100 G "stého", 1000 G "tisíceho" / "tisícího"), wrote Czech compounds as one word, and got decimals wrong ("tri celých"). v2 adds animacy, the Czech construction types, and declension options for long numbers.

`NUM2WORDS_CHANGES.md` lists every change with its source. It also names the Slovak forms requested in December 2025 that v2 reverses, and the decisions made in the native review.

### Reference implementation status

Every code block marked **(tested)** is the exact content of a file in the reference implementation: the `src/`, `tests/` and `scripts/` folders of the [caketts repository](https://github.com/martinambrus/caketts). The 25 September 2026 state was also packaged as `czech_slovak_tts_reference_v2.zip`.

**200 tests:** 102 for the TTS components, 94 for num2words and 4 for the environment. All pass except `test_cuda_available`, which skips without a GPU. They run on Python 3.13 with the versions pinned in `uv.lock`, among them torch 2.14 (CPU build), torchaudio 2.11, librosa 1.0, numpy 2.5, numba 0.67, transformers 5.17, phonemizer 3.4.0 and espeak-ng 1.52 (via espeakng-loader 0.2.4), and with BigVGAN `main` (commit 7d2b454).

The end-to-end test trains a tiny model on a synthetic language. It checks that MAS recovers the true segmentation, that the duration predictor learns it, that synthesis keeps every token, and that the generated content is right.

Not tested here (no GPU and no large model downloads): real XLS-R weights, BigVGAN inference, NeMo ASR models and multi-GPU training. Those steps are prompts with acceptance criteria.

---

# Table of Contents

1. [Project Structure](#1-project-structure)
2. [Step 1: Environment Setup](#step-1-environment-setup)
3. [Step 2: Text Normalization](#step-2-text-normalization)
4. [Step 3: G2P Phonemizer](#step-3-g2p-phonemizer)
5. [Step 4: Data Pipeline](#step-4-data-pipeline)
6. [Step 5: Text Encoder](#step-5-text-encoder)
7. [Step 6: Durations, MAS and Length Regulation](#step-6-durations-mas-and-length-regulation)
8. [Step 7: Flow-Matching Decoder](#step-7-flow-matching-decoder)
9. [Step 8: Model Assembly and Narrator/Tempo Conditioning](#step-8-model-assembly-and-narratortempo-conditioning)
10. [Step 9: Multilingual SSL Discriminator](#step-9-multilingual-ssl-discriminator)
11. [Step 10: BigVGAN Vocoder](#step-10-bigvgan-vocoder)
12. [Step 11: Training Pipeline](#step-11-training-pipeline)
13. [Step 12: Inference Pipeline with ASR Verification](#step-12-inference-pipeline-with-asr-verification)
14. [Step 13: Production API](#step-13-production-api)
15. [Step 14: Evaluation and QA](#step-14-evaluation-and-qa)
16. [Appendix: Test Suite](#appendix-test-suite)

---

# 1. Project Structure

```
czech_slovak_tts/
├── configs/
│   ├── model/matcha_base.yaml
│   ├── training/{pretrain.yaml, finetune_narrator.yaml, vocoder.yaml, adversarial.yaml}
│   └── inference/inference.yaml
├── lexicon/
│   ├── lexicon.json                 # {"cs": {...}, "sk": {...}, "en": {...}} word -> phones
│   └── vocab.json                   # built from real espeak-ng output (Step 3.3)
├── src/
│   ├── text/      normalizer.py  segmenter.py  num2words_cs.py  num2words_sk.py  phonemizer.py
│   ├── data/      audio.py  dataset.py  tempo.py  prepare.py
│   ├── model/     text_encoder.py  duration.py  flow_matching.py  matcha_tts.py  discriminators.py
│   ├── vocoder/   bigvgan.py
│   ├── training/  trainer.py  stages.py
│   ├── inference/ synthesizer.py  api.py
│   └── eval/      asr_check.py  pace.py  report.py
├── tests/         test_phonemizer.py  test_audio.py  test_data_pipeline.py  test_model_components.py
│                  test_eval_and_discriminator.py  test_integration.py
│                  test_num2words_sk.py  test_num2words_cs.py
│                  test_environment.py  test_text_normalization.py  test_report.py
│                  fixtures/mel_basis_24k_100.npz
├── scripts/       prepare_corpus.py  build_vocab.py  g2p_review.py  cldr_crosscheck.py
│                  validate_all_sk.py  validate_all_cs.py
│                  train.py  tempo_labels.py  synthesize_book.py  evaluate.py
├── NUM2WORDS_CHANGES.md             # num2words v2: changes, sources, native-review decisions
├── third_party/BigVGAN/             # git clone https://github.com/NVIDIA/BigVGAN, then check out 7d2b454 (Prompt 1.1)
├── pyproject.toml                   # dependencies (uv); uv.lock pins every version (Prompt 1.1)
├── uv.lock
├── .python-version                  # 3.13
└── pytest.ini                       # [pytest] pythonpath = .   markers = slow
```

---

# Step 1: Environment Setup

## Claude Code Prompt 1.1: Create Project Structure (v2: done)

```text
Create the project structure shown in Section 1, with __init__.py in every package under src/.
Clone https://github.com/NVIDIA/BigVGAN into third_party/BigVGAN and check out commit 7d2b454,
the one the tests ran with (its mel code is the reference for our features and its model is our
vocoder).

Dependencies are managed with uv and are already in the repository: pyproject.toml lists them
(minimums = the versions the tests were run with), uv.lock pins every version, .python-version
pins Python 3.13, and python-preference = "only-managed" keeps an active conda or system Python
out of the project. `uv sync` creates .venv; run every command with `uv run`.
  - torch and torchaudio come from the PyTorch CPU index. On a CUDA machine, delete the
    pytorch-cpu index and the [tool.uv.sources] entries (PyPI's Linux torch wheels are the
    CUDA 13 build), then run `uv lock`.
  - torchaudio is used only for resample / forced_align; audio I/O goes through soundfile.
  - link-mode = "hardlink" shares installed files with the uv cache instead of copying them.
  - To upgrade: `uv lock --upgrade`, run the whole test suite, commit uv.lock.
Evaluation-only tools stay out of uv.lock:
  - PyICU (CLDR cross-check, Step 2.4): `uv run --with PyICU scripts/cldr_crosscheck.py`;
    needs libicu-dev
  - nemo_toolkit[asr]>=2.3 (Parakeet-TDT-0.6B-v3, Canary-1B-v2): its own environment, created
    when Steps 4.2, 12 and 14 start, because it pins many packages

Create pytest.ini with pythonpath = . and a "slow" marker.
```

## Claude Code Prompt 1.2: Create Base Configuration (v2: done)

```text
Create configs/model/matcha_base.yaml:
```

```yaml
model:
  n_langs: 3              # cs=0, sk=1, en=2 (must match src/text/phonemizer.py LANG_IDS)
  n_speakers: 1           # pretraining speakers + the narrator (Step 8)
  n_boundaries: 3         # 0 mid-paragraph, 1 paragraph-final, 2 chapter-final sentence
  n_mels: 100             # MUST equal the vocoder's num_mels
  encoder:   {dim: 192, layers: 6, heads: 2, ff: 768, convnext: 2, dropout: 0.1}
  duration:  {hidden: 256, kernel: 3, layers: 2, dropout: 0.1}
  conditioning: {spk_dim: 64, cond_dim: 64}
  decoder:   {dim: 256, levels: 2, sigma_min: 1.0e-4, p_uncond: 0.0}   # fixed width per level
  discriminators:           # only used in the late adversarial stage (Step 11, stage 4)
    ssl: {backbone: facebook/wav2vec2-xls-r-300m, n_layers: 25, hidden: 1024, head_dim: 256}
    mpd: {periods: [2, 3, 5, 7, 11]}
    mrd: {resolutions: [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]]}

audio:                    # identical to nvidia/bigvgan_v2_24khz_100band_256x config.json
  sample_rate: 24000
  n_fft: 1024
  hop_length: 256
  win_length: 1024
  n_mels: 100
  fmin: 0
  fmax: null              # = sample_rate / 2, exactly as BigVGAN
  target_lufs: -23.0      # EBU R128, applied per source file before segmentation

tempo:                    # filled from the narrator corpus in Step 4.4
  rate_mean: null         # phones per second of speech
  rate_std: null
  inference_rate: null    # the narrator's median; fixed for every sentence of a book
  length_scale: {cs: 1.0, sk: 1.0, en: 1.0}

training:
  batch_frames: 40000     # batch by total mel frames, not by utterance count
  learning_rate: 1.0e-4
  weight_decay: 0.0
  grad_clip: 1.0
  precision: bf16
  loss_weights: {duration: 1.0, prior: 1.0, cfm: 1.0}
```

## Test 1: Environment Validation (tested)

```python
# tests/test_environment.py
import json
import os
from pathlib import Path

import pytest
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
BIGVGAN_DIR = Path(os.environ.get("BIGVGAN_DIR", ROOT / "third_party" / "BigVGAN"))


def test_versions():
    import torch, torchaudio
    assert Version(torch.__version__.split("+")[0]) >= Version("2.14")   # not a string comparison
    assert Version(torchaudio.__version__.split("+")[0]) >= Version("2.11")


def test_espeak_languages():
    from src.text.phonemizer import CzechSlovakPhonemizer
    for lang in ("cs", "sk", "en"):   # 'en' is mapped to espeak-ng's 'en-us'
        assert CzechSlovakPhonemizer(lang).tokenize("test")


def test_audio_config_matches_vocoder():
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "model" / "matcha_base.yaml").read_text())["audio"]
    voc = json.loads((BIGVGAN_DIR / "configs" / "bigvgan_v2_24khz_100band_256x.json").read_text())
    assert (cfg["n_mels"], cfg["hop_length"], cfg["n_fft"], cfg["sample_rate"]) == \
        (voc["num_mels"], voc["hop_size"], voc["n_fft"], voc["sampling_rate"])


def test_cuda_available():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available: training will be slow")
```

---

# Step 2: Text Normalization

## 2.1 num2words for Slovak and Czech (v2: done)

`src/text/num2words_sk.py` and `src/text/num2words_cs.py` turn numbers into words: cardinals, ordinals and decimals, in three genders, six cases and three animacy classes. Both were rewritten on 25 September 2026 from published grammar:

- **Slovak:** Morfológia slovenského jazyka, the Pravidlá slovenského pravopisu (quoted by Jarošová 2021), Navrátil 2003, Beliana and JÚĽŠ language columns.
- **Czech:** the Internetová jazyková příručka of ÚJČ AV ČR, Czech Wikipedia, NESČ and Český rozhlas.

`NUM2WORDS_CHANGES.md` lists every change against v1 with its source. It also lists the decisions a native speaker made in review on 25 September 2026 where the sources were silent or allowed variants. Examples: Slovak *stodve knihy*, *dvoma stromami* but *dvomi stenami*, and Czech ordinal years *tisíc devět set devadesátý první*. No questions remain open. In the old modules, every declined cardinal was built from **ordinal** stems (100 G "stého", 1000 G "tisíceho" / "tisícího"), and the Czech compounds were written as one word.

| | Slovak | Czech |
|---|---|---|
| 25, genitive | dvadsiatich piatich | dvaceti pěti |
| 21, genitive | dvadsaťjeden (never declines) | jednadvaceti |
| 100 / 1000, genitive | sto / tisíc (do not decline) | sta / tisíce |
| 2000 | dvetisíc | dva tisíce |
| 2 000 000, genitive | dvoch miliónov | dvou milionů |
| 1847, genitive | tisícosemstoštyridsiatich siedmich | tisíc osm set čtyřiceti sedmi |
| 2024th | dvetisícdvadsiaty štvrtý | dva tisíce dvacátý čtvrtý |
| 3,14 | tri celé štrnásť stotín | tři celé čtrnáct setin |

The API is v1-compatible: `num2words(n, to="cardinal"|"ordinal", gender=..., case=...)`. It adds these keywords:

- `animacy="inanimate"|"animate"|"personal"`
- `construction`
- `plural` (ordinals)
- Slovak only: `declined`, `codified`
- Czech only: `oblique_style`, `ordinal_style`, `inverted`

Decimals are accepted as a float, a `Decimal` or a string.

**Training transcripts must say what the narrator said.** Where the narrator uses a standard variant, set the matching option, or correct that transcript by hand. Otherwise the text-audio pair is misaligned. Examples of such variants:

- Slovak *dvojtisíci* (the codified form)
- the undeclined *dvadsaťdva*
- Czech *jednadvacet*
- Czech all-ordinal years, "tisící devítistý devadesátý první"

Keep per-book choices in the book config, so that training and inference normalise alike.

## 2.2 Claude Code Prompt 2.2: Create Text Normalizer

```text
Create src/text/normalizer.py. Contract: its output contains NO digits and NO symbols such as
% / & + @ # (the G2P raises on them, because dropping them would be a skipped word).

Handle:
 1. Numbers -> words through num2words(n, to=..., gender=..., case=..., animacy=...). The hard part
    is choosing case, gender and animacy from context: "s 5 lidmi" -> instrumental "s pěti lidmi";
    Slovak "2 muži" -> animacy="personal" -> "dvaja muži". Use a morphological tagger on the sentence
    (UDPipe 2 or Stanza, both with Czech and Slovak models) to read the case, gender and animacy of
    the governing noun or preposition. Fall back to the nominative masculine inanimate and LOG the
    fallback, so native review can catch it. Keep the module defaults unless the book config sets a
    variant (Step 2.1). Decimals are read in the nominative ("tri celé štrnásť stotín").
 2. Dates ("1. ledna 2024" -> "prvního ledna dva tisíce dvacet čtyři"), times, currency, units that
    agree with their number ("5 km" -> "pět kilometrů"), Roman numerals ("Karel IV." -> "Karel čtvrtý").
 3. Abbreviations (keep the v1 tables for cs and sk and extend them).
 4. Dashes: normalise "–" and " - " to "—"; normalise "..." to "…".
 5. English spans the narrator reads in English: wrap them as <en>Harry Potter</en> (Step 3).
    Names the narrator adapts to Slovak/Czech stay untagged and go through the lexicon instead.
 6. Optional LLM proposals for hard cases must be verified by rules; reject any edit that
    changes more than 5% of characters outside the number span (the Granary pattern).
```

## 2.3 Claude Code Prompt 2.3: Sentence and boundary segmentation

```text
Create src/text/segmenter.py: split normalised book text into sentences, keeping paragraph and
chapter structure. Label every sentence with boundary = "mid" | "para_end" | "chapter_end".
The same labels are used for training data (Step 4.2) and at inference (Step 12), so
paragraph cadence comes from the label, never from "end of input".
```

## 2.4 CLDR cross-check (script, run here)

`scripts/cldr_crosscheck.py` loads the current Unicode CLDR spell-out rules into ICU and diffs every gender × case × number cell. It writes the disagreements to a TSV **for native review**.

Against the v1 modules it reported 1,848 Czech and 1,493 Slovak disagreements. Against v2 it reports 721 Czech and 1,154 Slovak. Every remaining group was reviewed, and none is a v2 bug. In each group, either CLDR contradicts the normative sources, or the form was chosen in the native review (`NUM2WORDS_CHANGES.md` §4). For example, CLDR writes "dve tisíce", "tretom", "tisícieho" and "stý prvý" where Slovak has "dvetisíc", "treťom", "tisíceho" and "stoprvý", and it uses the Czech agreement type "dvacet jeden". Re-run the script after any change to the modules.

```python
# scripts/cldr_crosscheck.py
#!/usr/bin/env python3
"""
Cross-check num2words_cs / num2words_sk against Unicode CLDR spell-out rules (Step 2.4).

CLDR is NOT ground truth (its Slovak data writes 2000 as "dve tisíce", and older ICU builds
cannot render every rule), so this prints DISAGREEMENTS FOR NATIVE REVIEW, grouped by
gender x case, ignoring spacing and soft hyphens. Needs PyICU, which builds against ICU
(apt install libicu-dev), and network access to GitHub for the current CLDR rule files.

usage: uv run --with PyICU scripts/cldr_crosscheck.py --module-dir src/text --out cldr_diff.tsv
"""
import argparse
import collections
import html
import importlib
import re
import sys
import urllib.request

import icu

CLDR = "https://raw.githubusercontent.com/unicode-org/cldr/main/common/rbnf/{}.xml"
CASES = ["nominative", "genitive", "dative", "accusative", "instrumental", "locative"]
GENDERS = ["masculine", "feminine", "neuter"]
NUMBERS = list(range(0, 130)) + [199, 200, 201, 222, 300, 345, 400, 500, 999, 1000, 1001, 1100,
                                 1999, 2000, 2001, 2024, 3000, 5000, 10000, 21000, 100000,
                                 1_000_000, 2_000_000, 5_000_000, 1_000_000_000, 2_000_000_000]


def cldr_formatter(lang: str) -> icu.RuleBasedNumberFormat:
    xml = urllib.request.urlopen(CLDR.format(lang)).read().decode("utf-8")
    rules = html.unescape(re.findall(r"<rbnfRules>(.*?)</rbnfRules>", xml, re.S)[0])
    return icu.RuleBasedNumberFormat(rules, icu.Locale(lang))


def norm(s: str) -> str:
    return s.replace("­", "").replace(" ", "").lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-dir", default="src/text")
    ap.add_argument("--out", default="cldr_diff.tsv")
    args = ap.parse_args()
    sys.path.insert(0, args.module_dir)
    rows = []
    for lang in ("cs", "sk"):
        mod = importlib.import_module(f"num2words_{lang}")
        fmt = cldr_formatter(lang)
        per_cell, total = collections.Counter(), 0
        for kind in ("cardinal", "ordinal"):
            for g in GENDERS:
                for c in CASES:
                    rs = f"%spellout-{kind}-{g}" + ("" if c == "nominative" else f"-{c}")
                    try:
                        fmt.setDefaultRuleSet(rs)
                    except icu.ICUError:
                        continue  # rule set absent in this CLDR/ICU combination
                    for n in NUMBERS:
                        if kind == "ordinal" and n == 0:
                            continue
                        ref = fmt.format(n)
                        if not ref or "|" in ref or "%" in ref:  # rule syntax this ICU cannot render
                            continue
                        ours = mod.num2words(n, to=kind, gender=g, case=c)
                        total += 1
                        if norm(ours) != norm(ref):
                            per_cell[(kind, g, c)] += 1
                            rows.append((lang, kind, g, c, n, ours, ref))
        print(f"{lang}: {sum(per_cell.values())} of {total} forms differ from CLDR (spacing ignored)")
        for (kind, g, c), k in per_cell.most_common(6):
            print(f"   {kind:8s} {g:9s} {c:12s} {k}")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("lang\tkind\tgender\tcase\tnumber\tours\tcldr\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    print(f"wrote {len(rows)} disagreements to {args.out} — review them, do not auto-apply")


if __name__ == "__main__":
    main()
```

## Test 2: Text Normalization Tests

The num2words suites ship with the reference implementation:

- `tests/test_num2words_sk.py` and `tests/test_num2words_cs.py`: 94 tests. Each expectation is quoted from a named source or was decided in native review.
- `scripts/validate_all_sk.py` and `scripts/validate_all_cs.py`: print every form, for native review.

Normalizer tests (to be written with Prompt 2.2):

```python
# tests/test_text_normalization.py
import pytest
from src.text.normalizer import TextNormalizer


@pytest.fixture
def cs():
    return TextNormalizer(language="cs")


@pytest.fixture
def sk():
    return TextNormalizer(language="sk")


def test_no_digits_or_symbols_survive(cs):
    out = cs.normalize("Dne 1.1.2024 v 14:30 zaplatil 100 Kč, tj. cca 4 € (20 %).")
    assert not any(c.isdigit() for c in out) and "%" not in out and "€" not in out


def test_numbers_cs(cs):
    assert cs.normalize("Mám 5 jablek.") == "Mám pět jablek."
    assert cs.normalize("Je mi 25 let.") == "Je mi dvacet pět let."          # Czech: separate words


def test_numbers_sk(sk):
    assert sk.normalize("Mám 25 rokov.") == "Mám dvadsaťpäť rokov."          # Slovak: one word


def test_case_from_context_cs(cs):
    assert cs.normalize("Šel s 5 přáteli.") == "Šel s pěti přáteli."


@pytest.mark.parametrize("text,expected", [
    ("1. ledna 2024", "Prvního ledna dva tisíce dvacet čtyři"),
    ("15.3.2024", "Patnáctého března dva tisíce dvacet čtyři"),
])
def test_dates_cs(cs, text, expected):
    assert cs.normalize(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("Karel IV.", "Karel čtvrtý."), ("XXI. století", "Dvacáté první století"), ("III. díl", "Třetí díl"),
])
def test_roman_numerals_cs(cs, text, expected):
    assert cs.normalize(text) == expected


def test_abbreviations(cs, sk):
    assert cs.normalize("např. toto") == "například toto"
    assert sk.normalize("atď.") == "a tak ďalej."


def test_dashes_and_ellipsis(cs):
    assert cs.normalize("A pak – nic...") == "A pak — nic…"


def test_punctuation_and_empty(cs):
    assert cs.normalize("Máš 5 jablek?").endswith("?")
    assert cs.normalize("") == ""
    assert cs.normalize("Toto je věta bez čísel.") == "Toto je věta bez čísel."
```

---

# Step 3: G2P Phonemizer

Once alignment is hard and durations are explicit, most errors a listener hears come from the text front end. espeak-ng has had no release since 1.52 (December 2024) and no Czech or Slovak rule changes since. It also has errors this plan now corrects. Probed on 1.52:

| Input | espeak-ng 1.52 | Correct | v2 handling |
|---|---|---|---|
| cs/sk "rozhlas", "bezhlavý" | r o **s h** l a s | r o **z ɦ** l a s | word rule `_fix_zh` |
| sk "kôň" | k **u o** ɲ | k **u̯o** ɲ | word rule `_merge_sk_o_circumflex` |
| cs "k domu", "s bratrem" | **k** \| d…, **s** \| b… | **ɡ** \| d…, **z** \| b… | cross-word voicing rules |
| cs "muž je", "led je" | **ʒ** \| j…, **d** \| j… | **ʃ** \| j…, **t** \| j… (voiceless before a sonorant) | cross-word voicing rules |
| sk "brat ide", "vták letí" | **t** \| i…, **k** \| l… | **d** \| i…, **ɡ** \| l… (Slovak voices before vowels and sonorants) | cross-word voicing rules |
| sk "viem", "nie" | v i **ˈe** m, ɲ i **ˌe** | stress on the first syllable | cs/sk stress marks dropped (stress is fixed) |
| "s", "a" spoken alone (per-word G2P) | **e s**, **aː** (letter names) | s, a | built-in one-letter word entries |

Open items for **native review**, which is not automated:

- **Slovak ia/ie/iu:** diphthongs in native words ("viem", "piatok", "hovoria") but hiatus [ija] in loanwords ("akcia", "rádio", "štúdium"). espeak-ng renders all of them as two vowels. Decide per corpus, and add lexicon entries for frequent words.
- **Slovak ä:** espeak-ng gives [e]. Keep it if the narrator says [e].
- **Czech "sh":** [sx] or [zɦ].
- **English names:** tag them `<en>…</en>` if the narrator reads them in English; otherwise add a Slovak/Czech lexicon entry.
- **Joined numerals** ("dvadsaťjeden"): espeak-ng gives them one primary stress. Listen for whether the narrator puts a secondary stress on the second part.

## Claude Code Prompt 3.1: Create the phonemizer (reference implementation below)

```text
Create src/text/phonemizer.py exactly as the reference below, and tests/test_phonemizer.py.
Requirements it satisfies:
 - espeak-ng codes cs, sk, en-us ('en' alone crashes espeak-ng)
 - output is a list of Tokens (phone | stress | boundary | punct), each with a language ID
 - stress, word boundaries and punctuation are never glued onto phones
 - <en>...</en> spans are phonemized as English, as whole phrases (weak forms stay right)
 - the lexicon is applied per word BEFORE espeak-ng; cs/sk are phonemized per word (cached),
   so orthography-conditioned rules always see the right word
 - word rules and cross-word voicing rules fix the espeak-ng errors in the table above
 - digits and symbols raise (the normalizer must verbalise them first)
 - the vocabulary is built from real tokenizer output; unknown symbols raise UnknownSymbolError
```

### Reference: `src/text/phonemizer.py` (tested)

```python
# src/text/phonemizer.py
"""
Czech / Slovak / English G2P on top of espeak-ng (via `phonemizer`).

What this module guarantees (and the v1 plan did not):
  * valid espeak-ng language codes ('en' -> 'en-us'; 'en' alone crashes espeak-ng)
  * TOKEN output, not a string: phones, word boundaries ('|') and punctuation are
    separate tokens, so nothing like 'ˈo', 'iː|' or 'e?' ever reaches the vocabulary
  * one language ID per token (for the text encoder's language embedding)
  * inline language spans:  "Čítal <en>Harry Potter</en> po anglicky."
  * a word-level exception lexicon that is really applied (looked up per word,
    before espeak-ng)
  * post-rules for known espeak-ng 1.52 errors in Czech/Slovak:
      - 'zh' devoiced to [s h] ("rozhlas" -> [roshlas])            -> fixed per word
      - Slovak 'ô' split into two vowels [u o]                       -> merged to [u̯o]
      - voicing at word boundaries (prepositions k/s/v/z, final
        obstruents) left inconsistent                                -> fixed per language
  * cs/sk stress marks dropped (stress is fixed on the first syllable; espeak-ng
    puts marks on the second half of diphthongs, e.g. "viem" -> v i ˈe m)
  * a vocabulary built from real espeak-ng output and STRICT id lookup:
    an unknown symbol raises instead of silently becoming <unk>

Rules are defaults for native-speaker review: every one of them can be switched off
or extended, and the test-suite pins their behaviour.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# espeak-ng backend loading
# --------------------------------------------------------------------------------------

LANG_CODES: Dict[str, str] = {"cs": "cs", "sk": "sk", "en": "en-us"}
LANG_IDS: Dict[str, int] = {"cs": 0, "sk": 1, "en": 2}

_BACKENDS: Dict[Tuple[str, bool], object] = {}


def _ensure_espeak_library() -> None:
    """Use the system libespeak-ng if phonemizer finds it, else the copy bundled by `espeakng-loader`."""
    from phonemizer.backend import EspeakBackend

    try:
        EspeakBackend.version()
        return
    except RuntimeError:
        pass
    import espeakng_loader  # bundles espeak-ng 1.52 (a dependency in pyproject.toml)
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())


def _backend(lang: str, with_stress: bool):
    key = (lang, with_stress)
    if key not in _BACKENDS:
        _ensure_espeak_library()
        from phonemizer.backend import EspeakBackend

        _BACKENDS[key] = EspeakBackend(
            LANG_CODES[lang],
            preserve_punctuation=False,  # punctuation is tokenized by us, not by espeak
            with_stress=with_stress,
            language_switch="remove-flags",
        )
    return _BACKENDS[key]


# --------------------------------------------------------------------------------------
# Token inventory helpers
# --------------------------------------------------------------------------------------

STRESS_MARKS = ("ˈ", "ˌ")
WORD_BOUNDARY = "|"
# Canonical punctuation tokens kept for prosody; everything else is mapped or dropped.
PUNCT_TOKENS = (",", ".", "!", "?", ":", ";", "…", "—")
_PUNCT_MAP = {
    ",": ",", ".": ".", "!": "!", "?": "?", ":": ":", ";": ";",
    "…": "…", "—": "—", "–": "—", "-": None,  # a lone hyphen between spaces behaves like a dash
    "(": ",", ")": ",", "[": ",", "]": ",",
}
_QUOTES = set("\"'„“”‚‘’«»‹›`´")

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", WORD_BOUNDARY, *PUNCT_TOKENS]

_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
_SPAN_RE = re.compile(r"<(cs|sk|en)>(.*?)</\1>", re.DOTALL)

# Paired obstruents in espeak-ng 1.52 cs/sk output (voiced -> voiceless)
VOICED_TO_VOICELESS = {
    "b": "p", "d": "t", "ɟ": "c", "ɡ": "k", "v": "f", "z": "s", "ʒ": "ʃ",
    "h": "x", "dz": "ts", "dʒ": "tʃ", "dʲ": "tʲ", "dʑ": "tɕ", "r̝": "r̝̊",
}
VOICELESS_TO_VOICED = {v: k for k, v in VOICED_TO_VOICELESS.items()}
OBSTRUENTS = set(VOICED_TO_VOICELESS) | set(VOICELESS_TO_VOICED)
# 'v' undergoes devoicing but does not trigger voicing; ř is left out as a trigger.
NON_TRIGGERS = {"v", "f", "r̝", "r̝̊"}


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------


@dataclass
class Token:
    symbol: str
    lang: str
    kind: str  # 'phone' | 'stress' | 'boundary' | 'punct'

    @property
    def lang_id(self) -> int:
        return LANG_IDS[self.lang]


@dataclass
class G2PConfig:
    keep_stress: Dict[str, bool] = field(default_factory=lambda: {"cs": False, "sk": False, "en": True})
    apply_voicing_rules: bool = True
    apply_word_rules: bool = True


# --------------------------------------------------------------------------------------
# Word-level post-rules (orthography available)
# --------------------------------------------------------------------------------------


def _fix_zh(word: str, phones: List[str]) -> List[str]:
    """'zh' inside a cs/sk word is [z ɦ] (espeak-ng writes 'h' for ɦ), never [s h]."""
    n_zh = word.lower().count("zh")
    out = list(phones)
    i = 0
    while n_zh and i < len(out) - 1:
        if out[i] == "s" and out[i + 1] == "h":
            out[i] = "z"
            n_zh -= 1
        i += 1
    return out


def _merge_sk_o_circumflex(word: str, phones: List[str]) -> List[str]:
    """Slovak 'ô' is the diphthong [u̯o]; espeak-ng splits it into 'u o'."""
    n = word.lower().count("ô")
    out: List[str] = []
    i = 0
    while i < len(phones):
        if n and phones[i] == "u" and i + 1 < len(phones) and phones[i + 1] == "o":
            out.append("u̯o")
            n -= 1
            i += 2
        else:
            out.append(phones[i])
            i += 1
    return out


WORD_RULES = {
    "cs": [_fix_zh],
    "sk": [_fix_zh, _merge_sk_o_circumflex],
    "en": [],
}

# Built-in entries for one-letter words (applied unless the letter is an initial like "K.")
FUNCTION_WORDS = {
    "cs": {"a": "a", "i": "i", "o": "o", "u": "u", "k": "k", "s": "s", "v": "v", "z": "z"},
    "sk": {"a": "a", "i": "i", "o": "o", "u": "u", "k": "k", "s": "s", "v": "v", "z": "z"},
}


# --------------------------------------------------------------------------------------
# Cross-word voicing (cs/sk)
# --------------------------------------------------------------------------------------


def _first_phone(tokens: Sequence[Token], start: int) -> Optional[Token]:
    for t in tokens[start:]:
        if t.kind == "phone":
            return t
        if t.kind in ("punct", "boundary"):
            return None
    return None


def apply_voicing_rules(tokens: List[Token]) -> List[Token]:
    """
    Regressive voicing assimilation at word boundaries for Czech and Slovak.

    For the obstruent cluster at the end of each cs/sk word:
      next word starts with a voicing trigger (obstruent except v/f/ř) -> copy its voicing
      next is punctuation or end of input (pause)                    -> voiceless
      next word starts with a vowel or sonorant: Czech -> voiceless, Slovak -> voiced
    Language switches (e.g. into an English span) count as a pause.
    """
    out = list(tokens)
    n = len(out)
    i = 0
    while i < n:
        # find the end of the current word: last phone before boundary/punct/end
        if out[i].kind != "phone" or out[i].lang not in ("cs", "sk"):
            i += 1
            continue
        j = i
        while j + 1 < n and out[j + 1].kind in ("phone", "stress") and out[j + 1].lang == out[i].lang:
            j += 1
        word_lang = out[i].lang
        # collect the trailing obstruent cluster [k0..j]
        k0 = j + 1
        while k0 - 1 >= i and out[k0 - 1].kind == "phone" and out[k0 - 1].symbol in OBSTRUENTS:
            k0 -= 1
        if k0 <= j:
            nxt_idx = j + 1
            voiced: Optional[bool]
            if nxt_idx >= n or out[nxt_idx].kind == "punct":
                voiced = False  # pause
            else:
                # skip the boundary token
                look = nxt_idx + 1 if out[nxt_idx].kind == "boundary" else nxt_idx
                nxt = _first_phone(out, look)
                if nxt is None or nxt.lang != word_lang:
                    voiced = False
                elif nxt.symbol in OBSTRUENTS and nxt.symbol not in NON_TRIGGERS:
                    voiced = nxt.symbol in VOICED_TO_VOICELESS
                else:  # vowel or sonorant (incl. v)
                    voiced = word_lang == "sk"
            table = VOICELESS_TO_VOICED if voiced else VOICED_TO_VOICELESS
            for k in range(k0, j + 1):
                sym = out[k].symbol
                if sym in table:
                    out[k] = Token(table[sym], out[k].lang, "phone")
        i = j + 1
    return out


# --------------------------------------------------------------------------------------
# Phonemizer
# --------------------------------------------------------------------------------------


def _split_espeak_word(raw: str, lang: str, keep_stress: bool) -> List[List[Token]]:
    """Turn espeak output for ONE input word into one or more words of tokens."""
    words: List[List[Token]] = []
    for chunk in raw.split(WORD_BOUNDARY):
        toks: List[Token] = []
        for item in chunk.split():
            item = _nfc(item)
            while item and item[0] in STRESS_MARKS:
                if keep_stress:
                    toks.append(Token(item[0], lang, "stress"))
                item = item[1:]
            if not item:
                continue
            item = item.replace("g", "ɡ")  # ASCII g -> IPA ɡ (U+0261)
            toks.append(Token(item, lang, "phone"))
        if toks:
            words.append(toks)
    return words


class CzechSlovakPhonemizer:
    """
    Usage:
        g2p = CzechSlovakPhonemizer(language="sk", lexicon_path="lexicon.json")
        tokens = g2p.tokenize("Čítal <en>Harry Potter</en> po anglicky.")
        symbols = [t.symbol for t in tokens]; lang_ids = [t.lang_id for t in tokens]
    """

    def __init__(
        self,
        language: str = "cs",
        lexicon: Optional[Dict[str, Dict[str, str]]] = None,
        lexicon_path: Optional[str] = None,
        config: Optional[G2PConfig] = None,
    ):
        if language not in LANG_CODES:
            raise ValueError(f"language must be one of {sorted(LANG_CODES)}, got {language!r}")
        self.language = language
        self.config = config or G2PConfig()
        # lexicon: {lang: {word_casefolded: "space separated phones"}}
        self.lexicon: Dict[str, Dict[str, str]] = {k: {} for k in LANG_CODES}
        if lexicon_path:
            data = json.loads(Path(lexicon_path).read_text(encoding="utf-8"))
            for lang, entries in data.items():
                self.add_lexicon_entries(lang, entries)
        if lexicon:
            for lang, entries in lexicon.items():
                self.add_lexicon_entries(lang, entries)

    # ---- lexicon -------------------------------------------------------------------
    def add_lexicon_entries(self, lang: str, entries: Dict[str, str]) -> None:
        for word, phones in entries.items():
            self.lexicon[lang][_nfc(word).casefold()] = _nfc(phones)
        self._espeak.cache_clear()

    # ---- espeak (cached per word / per English phrase) -------------------------------
    @lru_cache(maxsize=200_000)
    def _espeak(self, lang: str, text: str) -> Tuple[Tuple[Tuple[str, str], ...], ...]:
        keep = self.config.keep_stress[lang]
        raw = _backend(lang, keep).phonemize([text], strip=True, separator=_separator())[0]
        words = _split_espeak_word(raw, lang, keep)
        return tuple(tuple((t.symbol, t.kind) for t in w) for w in words)

    def _lexicon_tokens(self, word: str, lang: str, is_initial: bool) -> Optional[List[Token]]:
        key = _nfc(word).casefold()
        entry = self.lexicon[lang].get(key)
        if entry is None and not is_initial:
            # single-letter prepositions/conjunctions: espeak-ng spells an isolated
            # "s" as the letter name [e s] and lengthens "a" to [aː]
            entry = FUNCTION_WORDS.get(lang, {}).get(key)
        if entry is None:
            return None
        return [Token(p, lang, "stress" if p in STRESS_MARKS else "phone") for p in entry.split()]

    def _word_tokens(self, word: str, lang: str, is_initial: bool = False) -> List[List[Token]]:
        lex = self._lexicon_tokens(word, lang, is_initial)
        if lex is not None:
            return [lex]
        words = [[Token(s, lang, k) for s, k in w] for w in self._espeak(lang, word)]
        if not words:
            raise ValueError(f"espeak-ng produced no phones for {word!r} ({lang}); add a lexicon entry")
        if self.config.apply_word_rules and WORD_RULES[lang] and len(words) == 1:
            symbols = [t.symbol for t in words[0]]  # rules only match adjacent phones
            for rule in WORD_RULES[lang]:
                symbols = rule(word, symbols)
            words = [[Token(s, lang, "stress" if s in STRESS_MARKS else "phone") for s in symbols]]
        return words

    def _phrase_tokens(self, phrase: List[str], lang: str) -> List[List[Token]]:
        """English: phonemize runs of words together so weak forms ('the', 'a', 'to') are right."""
        out: List[List[Token]] = []
        run: List[str] = []

        def flush():
            if run:
                ws = [[Token(s, lang, k) for s, k in w] for w in self._espeak(lang, " ".join(run))]
                if not ws:
                    raise ValueError(f"espeak-ng produced no phones for {' '.join(run)!r} ({lang})")
                out.extend(ws)
                run.clear()

        for w in phrase:
            lex = self._lexicon_tokens(w, lang, is_initial=False) if _nfc(w).casefold() in self.lexicon[lang] else None
            if lex is not None:
                flush()
                out.append(lex)
            else:
                run.append(w)
        flush()
        return out

    # ---- public API ----------------------------------------------------------------------
    def tokenize(self, text: str) -> List[Token]:
        text = _nfc(text).replace("...", "…")
        tokens: List[Token] = []
        pos = 0
        spans: List[Tuple[str, str]] = []
        for m in _SPAN_RE.finditer(text):
            if m.start() > pos:
                spans.append((self.language, text[pos:m.start()]))
            spans.append((m.group(1), m.group(2)))
            pos = m.end()
        if pos < len(text):
            spans.append((self.language, text[pos:]))

        def add_word(w_tokens: List[Token]) -> None:
            if tokens and tokens[-1].kind in ("phone", "stress"):
                tokens.append(Token(WORD_BOUNDARY, w_tokens[0].lang, "boundary"))
            tokens.extend(w_tokens)

        for lang, chunk in spans:
            idx = 0
            phrase: List[str] = []  # English words waiting to be phonemized together
            for m in _WORD_RE.finditer(chunk):
                between = chunk[idx:m.start()]
                if lang == "en" and phrase and between.strip(" "):
                    for w in self._phrase_tokens(phrase, lang):
                        add_word(w)
                    phrase = []
                self._emit_punct(between, lang, tokens, text)
                word = m.group(0)
                if lang == "en":
                    phrase.append(word)
                else:
                    is_initial = len(word) == 1 and word.isupper() and chunk[m.end():m.end() + 1] == "."
                    for w in self._word_tokens(word, lang, is_initial):
                        add_word(w)
                idx = m.end()
            if phrase:
                for w in self._phrase_tokens(phrase, lang):
                    add_word(w)
            self._emit_punct(chunk[idx:], lang, tokens, text)

        if self.config.apply_voicing_rules:
            tokens = apply_voicing_rules(tokens)
        return tokens

    # characters that may be dropped silently (they carry no sound)
    IGNORABLE = _QUOTES | set("*_~")

    def _emit_punct(self, between: str, lang: str, tokens: List[Token], text: str) -> None:
        for ch in between:
            if ch.isspace() or ch in self.IGNORABLE:
                continue
            if ch == "-":
                # " - " is a dash; an intra-word hyphen ("Rimsky-Korsakov") is just a boundary
                mapped = "—" if between != "-" and between.strip() == "-" else None
            elif ch in _PUNCT_MAP:
                mapped = _PUNCT_MAP[ch]
            else:
                # digits, %, /, &, @ ... must be verbalised by the text normalizer first;
                # dropping them here would silently skip words
                raise ValueError(f"unnormalized character {ch!r} in {text!r}; run the normalizer first")
            if mapped is None:
                continue
            if tokens and tokens[-1].kind == "punct" and tokens[-1].symbol == mapped:
                continue  # collapse repeats like ",," or ".."
            tokens.append(Token(mapped, lang, "punct"))

    def phonemize(self, text: str) -> str:
        """Human-readable form for logs and lexicon work, e.g. 'r o z h l a s .'"""
        return " ".join(t.symbol for t in self.tokenize(text))


def _separator():
    from phonemizer.separator import Separator

    return Separator(phone=" ", word=f" {WORD_BOUNDARY} ", syllable="")


# --------------------------------------------------------------------------------------
# Vocabulary: built from real espeak-ng output, strict lookup
# --------------------------------------------------------------------------------------


class UnknownSymbolError(KeyError):
    pass


def build_vocabulary(texts_by_lang: Dict[str, Iterable[str]],
                     phonemizers: Optional[Dict[str, CzechSlovakPhonemizer]] = None) -> Dict[str, int]:
    """
    Build the symbol vocabulary from the ACTUAL tokenizer output over the corpus
    (training transcripts + a coverage text per language). Deterministic order.
    """
    symbols = set()
    for lang, texts in texts_by_lang.items():
        g2p = (phonemizers or {}).get(lang) or CzechSlovakPhonemizer(language=lang)
        for text in texts:
            symbols.update(t.symbol for t in g2p.tokenize(text))
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for s in sorted(symbols - set(vocab)):
        vocab[s] = len(vocab)
    return vocab


def tokens_to_ids(tokens: Sequence[Token], vocab: Dict[str, int], strict: bool = True,
                  context: str = "") -> Tuple[List[int], List[int]]:
    """Return (symbol_ids, lang_ids). With strict=True an unknown symbol raises."""
    ids, langs = [], []
    for t in tokens:
        if t.symbol not in vocab:
            if strict:
                raise UnknownSymbolError(f"symbol {t.symbol!r} ({t.lang}) not in vocabulary. {context}")
            ids.append(vocab["<unk>"])
        else:
            ids.append(vocab[t.symbol])
        langs.append(t.lang_id)
    return ids, langs


def save_vocabulary(vocab: Dict[str, int], path: str) -> None:
    Path(path).write_text(json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8")


def load_vocabulary(path: str) -> Dict[str, int]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
```

## 3.2 G2P review procedure (native speaker, one afternoon per language)

```text
Claude Code Prompt 3.2: Create scripts/g2p_review.py. From the narrator transcripts, list:
 (a) the 3,000 most frequent words; (b) every word containing ia/ie/iu/ô/ä/zh/sh, foreign letters
 (q, w, x, y in Slovak, and ö, ü, é in names) or capital letters mid-sentence (names);
 (c) every English span. Print word | count | phones | an example sentence to a TSV.
The reviewer marks wrong lines; the corrections become lexicon/lexicon.json:
  {"sk": {"akcia": "a k ts i j a"}, "cs": {...}, "en": {...}}
Re-run the review after the first listening test (Step 14) and after adding new books.
```

## 3.3 Build the vocabulary from the corpus

```text
Claude Code Prompt 3.3: Create scripts/build_vocab.py: tokenize every training transcript in
each language plus lexicon entries and a coverage text per language (a few hundred phonetically
rich sentences, including English ones), then save lexicon/vocab.json with build_vocabulary().
Rebuild the vocabulary only before pretraining; after that, new symbols mean a G2P change that
must be reviewed.
```

## Test 3: Phonemizer Tests (tested, 52 tests, real espeak-ng 1.52)

```python
# tests/test_phonemizer.py
"""Tests for the Czech/Slovak/English G2P. Runs against real espeak-ng 1.52."""
import pytest

from src.text.phonemizer import (
    CzechSlovakPhonemizer, PUNCT_TOKENS, STRESS_MARKS, WORD_BOUNDARY,
    UnknownSymbolError, build_vocabulary, tokens_to_ids,
)

CS = CzechSlovakPhonemizer("cs")
SK = CzechSlovakPhonemizer("sk")
EN = CzechSlovakPhonemizer("en")

COVERAGE = {
    "cs": ["Dobrý den, jak se máte? Rozhlas vysílá zprávy.",
           "Příliš žluťoučký kůň úpěl ďábelské ódy.",
           "Kde je ten muž? Led je tenký, ale hrad byl pevný.",
           "V tom domě bydlí sbor zpěváků; všechno je v pořádku!",
           "Kvůli dešti jsme šli k domu s bratrem a gólman chytil míč.",
           "Tvůj pes štěká: „Haf!“ — a pak se schoval…"],
    "sk": ["Dobrý deň, päť ľudí. Rozhlas vysiela.",
           "Kôň a vôňa, chlieb a mliečny piatok.",
           "Brat ide domov, vták letí nad hradom.",
           "Otec a mama sa s otcom Ďakujem ťave, ďaleko ňufák.",
           "Dievča hovorí: „Nie, ja to viem.“ — a odišla…",
           "Kde je krv? Gól padol v Bratislave; džús a dzurovať!"],
    "en": ["Harry Potter thought the weather was nice.",
           "Sherlock Holmes measured the church through the judge's window.",
           "She sings; they bought a boat, a toy and the cure."],
}


def symbols(tokens):
    return [t.symbol for t in tokens]


def phones(tokens):
    return [t.symbol for t in tokens if t.kind == "phone"]


def words(tokens):
    """Phones grouped per word (split at '|' and punctuation)."""
    out, cur = [], []
    for t in tokens:
        if t.kind == "phone":
            cur.append(t.symbol)
        elif t.kind in ("boundary", "punct") and cur:
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    return out


class TestTokenization:
    @pytest.mark.parametrize("g2p,text", [(CS, s) for s in COVERAGE["cs"]] +
                             [(SK, s) for s in COVERAGE["sk"]] + [(EN, s) for s in COVERAGE["en"]])
    def test_tokens_are_atomic(self, g2p, text):
        """No stress mark, boundary or punctuation glued onto a phone (v1 bug: 'ˈo', 'iː|', 'e?')."""
        for t in g2p.tokenize(text):
            if t.kind == "phone":
                assert not any(m in t.symbol for m in STRESS_MARKS), t
                assert WORD_BOUNDARY not in t.symbol, t
                assert not any(p in t.symbol for p in PUNCT_TOKENS), t
                assert " " not in t.symbol and t.symbol, t

    def test_english_code_is_valid(self):
        # v1 called espeak with 'en' and crashed; we map to 'en-us'
        assert "ɹ" in symbols(EN.tokenize("Harry Potter"))

    def test_empty_input(self):
        assert CS.tokenize("") == [] and CS.tokenize("   ") == []

    def test_unnormalized_digits_raise(self):
        # dropping '5' silently would be a skipped word; the normalizer must run first
        with pytest.raises(ValueError):
            CS.tokenize("Mám 5 jablek.")

    def test_punctuation_tokens(self):
        toks = CS.tokenize("Řekl: „Ano…“ – a odešel.")
        assert [t.symbol for t in toks if t.kind == "punct"] == [":", "…", "—", "."]

    def test_intra_word_hyphen_is_not_a_dash(self):
        assert "—" not in symbols(CS.tokenize("Rimsky-Korsakov"))

    def test_deterministic(self):
        s = "Brat ide domov, vták letí."
        assert symbols(SK.tokenize(s)) == symbols(SK.tokenize(s))


class TestStressPolicy:
    def test_no_stress_for_cs_sk(self):
        assert not any(t.kind == "stress" for t in SK.tokenize("viem, že nie"))
        assert not any(t.kind == "stress" for t in CS.tokenize("automobil"))

    def test_stress_kept_for_english(self):
        assert any(t.kind == "stress" for t in EN.tokenize("hello world"))


class TestLexiconAndSpans:
    def test_lexicon_override_is_applied(self):
        g2p = CzechSlovakPhonemizer("sk", lexicon={"sk": {"covid": "k o v i t"}})
        assert phones(g2p.tokenize("covid")) == ["k", "o", "v", "i", "t"]

    def test_lexicon_is_case_insensitive(self):
        g2p = CzechSlovakPhonemizer("cs", lexicon={"cs": {"Tolkien": "t o l k iː n"}})
        assert phones(g2p.tokenize("TOLKIEN")) == ["t", "o", "l", "k", "iː", "n"]

    def test_language_span(self):
        toks = SK.tokenize("Čítal <en>Harry Potter</en> po slovensky.")
        en = [t for t in toks if t.lang == "en" and t.kind == "phone"]
        sk = [t for t in toks if t.lang == "sk" and t.kind == "phone"]
        assert en and sk
        assert "ɹ" in [t.symbol for t in en]
        assert all(t.lang_id == 2 for t in en) and all(t.lang_id == 1 for t in sk)


class TestEspeakCorrections:
    @pytest.mark.parametrize("g2p", [CS, SK])
    def test_zh_is_voiced(self, g2p):
        p = phones(g2p.tokenize("rozhlas"))
        assert p[:4] == ["r", "o", "z", "h"], p   # espeak-ng 1.52 says [r o s h ...]

    def test_slovak_o_circumflex_is_one_diphthong(self):
        assert "u̯o" in phones(SK.tokenize("kôň"))

    @pytest.mark.parametrize("text,expected_first", [
        ("k domu", "ɡ"), ("s bratrem", "z"), ("v tom", "f"), ("k hradu", "ɡ"),
    ])
    def test_czech_preposition_voicing(self, text, expected_first):
        assert phones(CS.tokenize(text))[0] == expected_first

    @pytest.mark.parametrize("text,expected_last", [
        ("muž je", "ʃ"),       # Czech: voiceless before a sonorant
        ("led je", "t"),
        ("list byl", "d"),     # the cluster assimilates to the voiced b: [lɪzd bɪl]
        ("hrad padl", "t"),
        ("krev.", "f"),        # final devoicing before a pause
    ])
    def test_czech_word_final_voicing(self, text, expected_last):
        assert words(CS.tokenize(text))[0][-1] == expected_last

    @pytest.mark.parametrize("text,expected_last", [
        ("brat ide", "d"),     # Slovak: VOICED before a vowel
        ("vták letí", "ɡ"),    # and before a sonorant
        ("otec a mama", "dz"),
        ("od brata", "d"),
    ])
    def test_slovak_word_final_voicing(self, text, expected_last):
        assert words(SK.tokenize(text))[0][-1] == expected_last

    def test_slovak_final_devoicing_before_pause(self):
        assert phones(SK.tokenize("hrad."))[-1] == "t"

    def test_slovak_preposition_before_vowel(self):
        assert phones(SK.tokenize("s otcom"))[0] == "z"

    @pytest.mark.parametrize("g2p", [CS, SK])
    def test_one_letter_words_are_not_spelled(self, g2p):
        # espeak-ng reads an isolated "s" as the letter name [e s] and "a" as [aː]
        assert words(g2p.tokenize("a s ním"))[:2] == [["a"], ["s"]] or \
            words(g2p.tokenize("a s ním"))[:2] == [["a"], ["z"]]

    def test_initial_is_spelled(self):
        # "K." as an initial keeps its letter name
        assert words(CS.tokenize("K. Čapek"))[0] == ["k", "aː"]

    def test_english_weak_forms_in_phrases(self):
        # phrase-level phonemization keeps the reduced vowel in "a" and "the"
        w = words(EN.tokenize("a cup of the tea"))
        assert w[0] != ["eɪ"]


@pytest.fixture(scope="module")
def vocab():
    return build_vocabulary(COVERAGE)


class TestVocabulary:
    def test_special_tokens_first(self, vocab):
        assert [vocab[t] for t in ("<pad>", "<unk>", "<bos>", "<eos>", "|")] == [0, 1, 2, 3, 4]

    def test_every_training_token_is_known(self, vocab):
        # v1 mapped 41-62% of real cs/sk tokens (88% of English) to <unk>
        for lang, texts in COVERAGE.items():
            g2p = CzechSlovakPhonemizer(lang)
            for text in texts:
                ids, langs = tokens_to_ids(g2p.tokenize(text), vocab)
                assert vocab["<unk>"] not in ids and len(ids) == len(langs)

    def test_unknown_symbol_raises(self, vocab):
        toks = CzechSlovakPhonemizer("en").tokenize("the")
        toks[0].symbol = "ʘ"  # a click consonant nobody trained on
        with pytest.raises(UnknownSymbolError):
            tokens_to_ids(toks, vocab)

    def test_vocabulary_is_deterministic(self, vocab):
        assert build_vocabulary(COVERAGE) == vocab
```

---

# Step 4: Data Pipeline

## 4.1 Audio processing

```text
Claude Code Prompt 4.1: Create src/data/audio.py exactly as the reference below.
 - Mel features must be identical to BigVGAN's own mel_spectrogram (the test compares them).
 - Loudness: EBU R128 via pyloudnorm, per SOURCE FILE (chapter or session) before segmentation,
   with a gain reduction instead of clipping.
 - Trimming never cuts into speech: bounds come from a forced aligner, or from an energy detector
   relative to the recording's noise floor; a fixed pad (default 150 ms) is kept on both sides.
 - Audio I/O through soundfile (torchaudio.load/save require TorchCodec since torchaudio 2.9).
```

### Reference: `src/data/audio.py` (tested)

```python
# src/data/audio.py
"""
Audio processing for training and inference.

Fixes relative to v1:
  * mel features are computed EXACTLY like BigVGAN v2 (the vocoder we use): 100 bands,
    Slaney-scale + Slaney-normalised librosa filterbank, (n_fft - hop)/2 reflect padding,
    center=False, magnitude sqrt(re^2 + im^2 + 1e-9), log(clamp(x, 1e-5)).
    v1 used 80 HTK-scale unnormalised bands with center=True: a 3-6 nat per-band offset
    and a different frame grid than the vocoder expects.
  * loudness normalisation is real EBU R128 (pyloudnorm), applied per SOURCE FILE
    (chapter/session) before segmentation, so within-session dynamics survive.
    v1 was peak normalisation to 0.95 per clip and never used target_lufs.
  * trimming never cuts into speech: bounds come from a forced aligner when available,
    otherwise from an energy detector relative to the recording's NOISE FLOOR (not the
    clip's loudest frame), and a fixed pad (default 150 ms) is kept on both sides.
  * I/O uses soundfile: torchaudio.load/save need TorchCodec since torchaudio 2.9.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from librosa.filters import mel as librosa_mel_fn

LOG_MEL_FLOOR = float(np.log(1e-5))  # value of a silent frame; use it for padding


@dataclass
class MelConfig:
    """Must match the vocoder's config.json (nvidia/bigvgan_v2_24khz_100band_256x)."""
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 100
    fmin: float = 0.0
    fmax: Optional[float] = None  # None -> sample_rate / 2, exactly as BigVGAN's `fmax: null`

    @classmethod
    def from_bigvgan_config(cls, cfg: dict) -> "MelConfig":
        return cls(cfg["sampling_rate"], cfg["n_fft"], cfg["hop_size"], cfg["win_size"],
                   cfg["num_mels"], cfg["fmin"], cfg["fmax"])


class AudioProcessor:
    def __init__(self, mel: MelConfig = MelConfig(), target_lufs: float = -23.0,
                 peak_ceiling: float = 0.99):
        self.cfg = mel
        self.target_lufs = target_lufs
        self.peak_ceiling = peak_ceiling
        fb = librosa_mel_fn(sr=mel.sample_rate, n_fft=mel.n_fft, n_mels=mel.n_mels,
                            fmin=mel.fmin, fmax=mel.fmax)  # librosa defaults: htk=False, norm="slaney"
        self._mel_basis = torch.from_numpy(fb).float()
        self._window = torch.hann_window(mel.win_length)

    # ---- I/O ------------------------------------------------------------------------
    def load_audio(self, path: str) -> torch.Tensor:
        """Mono float32 tensor (1, T) at cfg.sample_rate."""
        data, sr = sf.read(path, dtype="float32", always_2d=True)  # (T, C)
        wav = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
        if sr != self.cfg.sample_rate:
            wav = AF.resample(wav, sr, self.cfg.sample_rate)
        return wav

    def save_audio(self, path: str, wav: torch.Tensor) -> None:
        sf.write(path, wav.squeeze(0).cpu().numpy(), self.cfg.sample_rate, subtype="PCM_24")

    # ---- loudness (per source file, before segmentation) -------------------------------
    def normalize_loudness(self, wav: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """EBU R128 integrated loudness -> target_lufs; returns (audio, applied_gain_db)."""
        import pyloudnorm as pyln

        x = wav.squeeze(0).numpy().astype(np.float64)
        if len(x) < int(0.4 * self.cfg.sample_rate):
            raise ValueError("loudness needs >= 400 ms of audio; normalise the whole source file")
        loudness = pyln.Meter(self.cfg.sample_rate).integrated_loudness(x)
        gain_db = self.target_lufs - loudness
        y = x * 10 ** (gain_db / 20)
        peak = np.abs(y).max()
        if peak > self.peak_ceiling:  # never clip: lower the gain instead (and log it)
            reduce = self.peak_ceiling / peak
            y *= reduce
            gain_db += 20 * np.log10(reduce)
        return torch.from_numpy(y.astype(np.float32)).unsqueeze(0), float(gain_db)

    # ---- trimming ------------------------------------------------------------------------
    def trim(self, wav: torch.Tensor, speech_start_s: float, speech_end_s: float,
             pad_s: float = 0.15) -> torch.Tensor:
        """Keep [first speech - pad, last speech + pad]. Bounds from a forced aligner are best."""
        sr = self.cfg.sample_rate
        a = max(0, int(round((speech_start_s - pad_s) * sr)))
        b = min(wav.shape[-1], int(round((speech_end_s + pad_s) * sr)))
        return wav[..., a:b]

    def detect_speech_bounds(self, wav: torch.Tensor, noise_floor_db: Optional[float] = None,
                             margin_db: float = 12.0, frame_s: float = 0.010) -> Tuple[float, float]:
        """
        Energy fallback when no alignment exists. The threshold is `margin_db` above the
        recording's noise floor (pass the SESSION floor; default: 5th percentile of this
        clip's frame RMS), not relative to the clip's loudest frame as in v1, so a quiet
        onset in a loud clip is not cut.
        """
        x = wav.squeeze(0).numpy()
        hop = max(1, int(frame_s * self.cfg.sample_rate))
        n = len(x) // hop
        if n == 0:
            return 0.0, len(x) / self.cfg.sample_rate
        frames = x[: n * hop].reshape(n, hop)
        rms_db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-10)
        floor = np.percentile(rms_db, 5) if noise_floor_db is None else noise_floor_db
        speech = np.nonzero(rms_db > floor + margin_db)[0]
        if len(speech) == 0:
            return 0.0, len(x) / self.cfg.sample_rate
        return speech[0] * hop / self.cfg.sample_rate, (speech[-1] + 1) * hop / self.cfg.sample_rate

    # ---- features ------------------------------------------------------------------------
    def compute_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """(1, T) or (T,) waveform -> (n_mels, T // hop) log-mel, identical to BigVGAN v2."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        c = self.cfg
        pad = (c.n_fft - c.hop_length) // 2
        y = torch.nn.functional.pad(wav.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spec = torch.stft(y, c.n_fft, hop_length=c.hop_length, win_length=c.win_length,
                          window=self._window.to(y.device), center=False, pad_mode="reflect",
                          normalized=False, onesided=True, return_complex=True)
        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
        mel = torch.matmul(self._mel_basis.to(y.device), spec)
        return torch.log(torch.clamp(mel, min=1e-5)).squeeze(0)

    def n_frames(self, n_samples: int) -> int:
        """Frames produced by compute_mel for n_samples (>= win_length - hop)."""
        return n_samples // self.cfg.hop_length
```

## 4.2 Corpus preparation (narrator + pretraining data)

```text
Claude Code Prompt 4.2: Create src/data/prepare.py and scripts/prepare_corpus.py:

 1. Loudness: normalise each source file (chapter/session) to -23 LUFS (AudioProcessor.normalize_loudness).
 2. Long-form alignment of chapter audio to book text:
      - Czech/English: Montreal Forced Aligner has pretrained models; Slovak has none.
      - All three: CTC forced alignment with torchaudio's MMS_FA bundle (romanise cs/sk text with
        uroman or unidecode first), or NVIDIA NeMo Forced Aligner with Parakeet-TDT-0.6B-v3,
        which covers Czech and Slovak and gives word and character timestamps.
 3. Segment at sentence boundaries (Step 2.3). Cut in the middle of the inter-sentence pause and
    record boundary = mid | para_end | chapter_end. Use identical rules for every session.
 4. Trim each clip from its word timestamps with fixed 150 ms pads (AudioProcessor.trim).
 5. ASR audit: transcribe every clip with Parakeet-TDT-0.6B-v3 and compute CER against the
    normalised text. Compute the narrator's median CER, then send clips above median + 5 points
    to review. The narrator departing from the text is the most common cause, so fix the text,
    not the audio. Drop clips that cannot be fixed.
 6. Write a JSONL manifest, one line per clip:
    {"id", "audio", "text" (normalised, with <en> spans), "lang", "speaker", "session",
     "boundary", "duration_s", "articulation_rate", "pause_ratio"}   (rates: Step 4.4)
 7. Filter with filter_manifest (Step 4.3): drop over-length (default > 20 s), too-short (< 0.5 s)
    and impossible clips (fewer frames than tokens). Log the report; never crop.

Pretraining data (optional, recommended for robustness on rare words, names and numbers):
 - Slovak: SloPalSpeech (LREC 2026), 2,806 h of parliament speech, CC BY 4.0. It was filtered
   only at 40% WER, so re-filter to CER < 5% with the audit above. EuroSpeech-SK (about 2,554 h
   at CER < 20%) comes from the same parliament: deduplicate by session before combining.
 - Czech: ParlaSpeech-CZ (1,218 h, 555 speakers, CC BY-SA 4.0) and ParCzech4Speech (up to 2,695 h, CC BY).
 - English: LibriTTS-R.
 - Parliament audio is formal, male-heavy and often MP3-sourced. Use it to learn alignment and
   durations; keep it out of the narrator fine-tune.
 - Hold out the 2 h CC0 Czech LibriVox single-speaker set as a read-literature test set.
```

## 4.3 Dataset and collator

```text
Claude Code Prompt 4.3: Create src/data/dataset.py exactly as the reference below.
 - filter_manifest DROPS over-length / too-short / impossible utterances and reports them;
   the dataset never truncates mel or tokens.
 - Strict symbol lookup (UnknownSymbolError names the utterance); language IDs per token.
 - Mel targets are normalised with corpus statistics (corpus_mel_stats), padded with 0.
 - Batches carry speaker_ids, boundary_ids and rates for the conditioning in Step 8.
```

### Reference: `src/data/dataset.py` (tested)

```python
# src/data/dataset.py
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
```

## 4.4 Tempo labels (the pace requirement starts here)

The ElevenLabs symptoms (slowing over the last sentences, and a "slow mode" on some texts) fit models whose durations are implicit and sampled. In this architecture tempo is decided in one place, the duration predictor, and it is given explicit inputs:

- **Articulation rate:** phones per second of speech, with pauses ≥ 120 ms excluded.
- **Pause ratio.**

Both are fixed at inference to the narrator's median, so the model cannot pick a slow mode from the meaning of the text or from session noise.

```text
Claude Code Prompt 4.4: Create src/data/tempo.py as below and scripts/tempo_labels.py:
 1. Bootstrap: train the acoustic model (Step 11, stage 1 or 2) for ~20k steps with every rate
    input set to the corpus mean (so they carry no information yet).
 2. Run MAS over the training set to get per-token durations, or use aligner timestamps (Step 4.2).
 3. Compute articulation_rate and pause_ratio per utterance with tempo_from_alignment; write them
    into the manifest; set tempo.rate_mean / rate_std / inference_rate in the config.
 4. Plot per-session and per-chapter medians. Flag utterances beyond +-12% of the median or
    +-2 SD (these thresholds are proposals). Keep expressive outliers WITH their labels; drop
    rushed or disfluent takes.
 5. Optional: time-stretch a small subset by +-5-10% (updating the labels) to separate the rate
    axis from other factors.
 6. Continue training with the labels.
```

### Reference: `src/data/tempo.py` (tested)

```python
# src/data/tempo.py
"""
Tempo labels for the narrator data (Step 4.4) and the pace-conditioning inputs.

articulation rate = phones per second of SPEECH (pauses >= min_pause_s excluded)
pause ratio       = share of the utterance spent in pauses >= min_pause_s
Computed from token durations of an alignment (MAS durations from a first training pass, or a
forced aligner). The phones/s measure follows Data-Speech's nb_phonemes / utterance_length with
long silences removed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

PAUSE_KINDS = {"punct", "boundary"}


@dataclass
class TempoLabel:
    articulation_rate: float  # phones / s of speech
    pause_ratio: float        # 0..1


def tempo_from_alignment(kinds: Sequence[str], durations_frames: Sequence[int], hop_s: float,
                         min_pause_s: float = 0.12) -> TempoLabel:
    """
    kinds: token kinds from the phonemizer ('phone' | 'punct' | 'boundary' | 'stress')
    durations_frames: frames aligned to each token (MAS / forced aligner)
    A punctuation/boundary token longer than min_pause_s counts as a pause; shorter ones
    (and all phones) count as speech time.
    """
    n_phones, speech_s, pause_s = 0, 0.0, 0.0
    for k, d in zip(kinds, durations_frames):
        dur = d * hop_s
        if k in PAUSE_KINDS and dur >= min_pause_s:
            pause_s += dur
        else:
            speech_s += dur
            n_phones += k == "phone"
    total = speech_s + pause_s
    return TempoLabel(n_phones / max(speech_s, 1e-6), pause_s / max(total, 1e-6))


def flag_rate_outliers(rates: Dict[str, float], rel_tol: float = 0.12, z: float = 2.0) -> List[str]:
    """Utterances whose articulation rate is beyond +-rel_tol of the median or +-z SD (proposals)."""
    ids = list(rates)
    x = np.array([rates[i] for i in ids])
    med, sd = np.median(x), x.std() + 1e-9
    return [i for i, r in zip(ids, x) if abs(r - med) / med > rel_tol or abs(r - med) / sd > z]
```

## Test 4: Data Pipeline Tests (tested)

`test_mel_basis_matches_bigvgan_training` compares the filterbank with `tests/fixtures/mel_basis_24k_100.npz`, which librosa 0.10.2 computed (the BigVGAN v2 era). Never regenerate it with a newer librosa: that would defeat the test. It was produced with (uv fetches Python 3.12 if needed):

```bash
uv run --no-project --python 3.12 --with librosa==0.10.2.post1 --with "numpy<2" python -c "import numpy as np, librosa; np.savez_compressed('tests/fixtures/mel_basis_24k_100.npz', mel_basis=librosa.filters.mel(sr=24000, n_fft=1024, n_mels=100, fmin=0.0, fmax=None))"
```

```python
# tests/test_audio.py
"""Audio tests: vocoder-compatible mels, real loudness normalisation, trimming that keeps speech."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.audio import AudioProcessor, MelConfig

# BigVGAN checked out at 7d2b454, in third_party/BigVGAN or at BIGVGAN_DIR (see the README Quick start)
BIGVGAN_DIR = Path(os.environ.get("BIGVGAN_DIR", Path(__file__).resolve().parents[1] / "third_party" / "BigVGAN"))


@pytest.fixture(scope="module")
def ap():
    return AudioProcessor()


def speechlike(seconds=1.0, sr=24000, seed=0):
    """Harmonic 'voice' with a slowly varying pitch plus a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    f0 = 140 + 20 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    x = sum(np.sin(k * phase) / k for k in range(1, 12)) * 0.2
    return torch.from_numpy((x + 0.003 * rng.standard_normal(len(t))).astype(np.float32)).unsqueeze(0)


class TestMel:
    def test_matches_vocoder_config(self, ap):
        cfg = json.loads((BIGVGAN_DIR / "configs" / "bigvgan_v2_24khz_100band_256x.json").read_text())
        assert MelConfig.from_bigvgan_config(cfg) == ap.cfg

    def test_identical_to_bigvgan_mel(self, ap):
        """The acoustic model's targets must be exactly what BigVGAN was trained on."""
        sys.path.insert(0, str(BIGVGAN_DIR))
        from meldataset import mel_spectrogram
        wav = speechlike(1.3)
        c = ap.cfg
        ref = mel_spectrogram(wav, c.n_fft, c.n_mels, c.sample_rate, c.hop_length, c.win_length,
                              c.fmin, c.fmax, center=False).squeeze(0)
        ours = ap.compute_mel(wav)
        assert ours.shape == ref.shape
        assert torch.allclose(ours, ref, atol=1e-5)

    def test_mel_basis_matches_bigvgan_training(self, ap):
        """BigVGAN's mel code calls librosa too, so the test above cannot see a librosa change;
        the fixture is librosa 0.10.2's filterbank, from the era BigVGAN v2 was trained in."""
        ref = np.load(Path(__file__).parent / "fixtures" / "mel_basis_24k_100.npz")["mel_basis"]
        assert torch.allclose(ap._mel_basis, torch.from_numpy(ref), rtol=0, atol=1e-7)

    def test_band_count_and_frame_count(self, ap):
        for n in (24000, 24000 + 100, 48000 - 1):
            mel = ap.compute_mel(speechlike(n / 24000))
            assert mel.shape == (100, ap.n_frames(n))  # v1 test expected 90 frames for 1 s: wrong

    def test_silence_hits_the_log_floor(self, ap):
        mel = ap.compute_mel(torch.zeros(1, 24000))
        assert torch.allclose(mel, torch.full_like(mel, np.log(1e-5)), atol=1e-4)


class TestLoudness:
    def test_reaches_target_lufs(self, ap):
        import pyloudnorm as pyln
        quiet = speechlike(3.0) * 0.05
        out, gain = ap.normalize_loudness(quiet)
        measured = pyln.Meter(24000).integrated_loudness(out.squeeze(0).numpy().astype(np.float64))
        assert abs(measured - ap.target_lufs) < 0.1 and gain > 0

    def test_never_clips(self, ap):
        spiky = speechlike(3.0) * 0.001
        spiky[0, 1000] = 0.9  # one transient: loudness gain alone would clip it
        out, _ = ap.normalize_loudness(spiky)
        assert out.abs().max() <= ap.peak_ceiling + 1e-6

    def test_refuses_short_clips(self, ap):
        with pytest.raises(ValueError):
            ap.normalize_loudness(speechlike(0.2))


class TestTrimming:
    def test_pads_are_kept(self, ap):
        wav = torch.zeros(1, 48000)
        wav[:, 12000:36000] = speechlike(1.0)
        out = ap.trim(wav, 0.5, 1.5, pad_s=0.15)
        assert out.shape[-1] == int(1.3 * 24000)

    def test_quiet_onset_is_not_cut(self, ap):
        """v1 cut at -40 dB below the loudest frame and lost a 200 ms soft onset."""
        sr = 24000
        rng = np.random.default_rng(1)
        wav = torch.from_numpy((1e-4 * rng.standard_normal(2 * sr)).astype(np.float32)).unsqueeze(0)  # -80 dBFS floor
        onset = speechlike(0.2) * (10 ** (-60 / 20) / 0.2)       # ~ -60 dBFS soft onset
        loud = speechlike(0.8)                                    # ~ -20 dBFS body
        wav[:, int(0.5 * sr):int(0.7 * sr)] += onset
        wav[:, int(0.7 * sr):int(1.5 * sr)] += loud
        start, end = ap.detect_speech_bounds(wav)
        assert start <= 0.51 and end >= 1.49
        trimmed = ap.trim(wav, start, end)
        assert trimmed.shape[-1] >= int((1.0 + 0.3) * sr) - 1
```

```python
# tests/test_data_pipeline.py
"""Data pipeline: drop-don't-crop filtering, strict symbols, conditioning carried into batches."""
import json

import numpy as np
import pytest
import soundfile as sf
import torch

from src.data.audio import AudioProcessor
from src.data.dataset import TTSCollator, TTSDataset, filter_manifest, load_manifest
from src.text.phonemizer import CzechSlovakPhonemizer, UnknownSymbolError, build_vocabulary

SR = 24000
TEXTS = ["Dobrý den, jak se máte?", "Rozhlas vysílá zprávy.", "Kde je ten muž?",
         "Tohle je velmi dlouhá věta, která se nevejde do limitu.", "Ano."]


@pytest.fixture()
def corpus(tmp_path):
    rows = []
    durations = [1.8, 1.4, 1.2, 25.0, 0.3]  # #4 too long, #5 too short
    for i, (text, dur) in enumerate(zip(TEXTS, durations)):
        path = tmp_path / f"u{i}.wav"
        t = np.arange(int(dur * SR)) / SR
        sf.write(path, (0.1 * np.sin(2 * np.pi * 150 * t)).astype(np.float32), SR)
        rows.append({"id": f"u{i}", "audio": str(path), "text": text, "lang": "cs",
                     "speaker": "narrator", "boundary": "para_end" if i == 1 else "mid",
                     "duration_s": dur, "articulation_rate": 12.5, "pause_ratio": 0.1})
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return manifest


@pytest.fixture()
def g2p():
    return {"cs": CzechSlovakPhonemizer("cs")}


def token_counts(rows, g2p):
    return {r["id"]: len(g2p[r["lang"]].tokenize(r["text"])) for r in rows}


def test_filter_drops_instead_of_cropping(corpus, g2p):
    rows = load_manifest(str(corpus))
    kept, report = filter_manifest(rows, token_counts(rows, g2p), hop_s=256 / SR, max_duration_s=20.0)
    assert [r["id"] for r in kept] == ["u0", "u1", "u2"]
    assert report.too_long == ["u3"] and report.too_short == ["u4"]


def test_filter_drops_impossible_alignments(g2p):
    rows = [{"id": "x", "text": "Rozhlas vysílá zprávy.", "lang": "cs", "duration_s": 0.6}]
    n = token_counts(rows, g2p)
    kept, report = filter_manifest(rows, n, hop_s=0.1, min_duration_s=0.1)  # 6 frames < ~20 tokens
    assert kept == [] and report.impossible_alignment == ["x"]


def test_items_are_never_truncated(corpus, g2p):
    rows = load_manifest(str(corpus))
    kept, _ = filter_manifest(rows, token_counts(rows, g2p), hop_s=256 / SR)
    vocab = build_vocabulary({"cs": [r["text"] for r in kept]})
    ap = AudioProcessor()
    ds = TTSDataset(kept, ap, vocab, g2p, speaker_ids={"narrator": 0})
    for i, r in enumerate(kept):
        item = ds[i]
        assert item["mel"].shape[1] == ap.n_frames(int(r["duration_s"] * SR))
        assert item["phoneme_ids"].numel() == len(g2p["cs"].tokenize(r["text"]))
        assert item["lang_ids"].numel() == item["phoneme_ids"].numel()


def test_unknown_symbol_is_an_error_not_unk(corpus, g2p):
    rows = load_manifest(str(corpus))[:3]
    vocab = build_vocabulary({"cs": [rows[0]["text"]]})  # deliberately too small
    ds = TTSDataset(rows, AudioProcessor(), vocab, g2p, speaker_ids={"narrator": 0})
    with pytest.raises(UnknownSymbolError):
        for i in range(len(ds)):
            ds[i]


def test_collator_carries_conditioning(corpus, g2p):
    rows = load_manifest(str(corpus))
    kept, _ = filter_manifest(rows, token_counts(rows, g2p), hop_s=256 / SR)
    vocab = build_vocabulary({"cs": [r["text"] for r in kept]})
    ds = TTSDataset(kept, AudioProcessor(), vocab, g2p, speaker_ids={"narrator": 0}, mel_mean=-5.0, mel_std=2.0)
    batch = TTSCollator()([ds[i] for i in range(len(ds))])
    assert batch["mels"].shape[:2] == (3, 100)
    assert batch["boundary_ids"].tolist() == [0, 1, 0]
    assert torch.allclose(batch["rates"][:, 0], torch.tensor(12.5))
    t_len = batch["phoneme_lengths"]
    assert (batch["phoneme_ids"][0, t_len[0]:] == 0).all()
    assert (batch["mels"][2, :, batch["mel_lengths"][2]:] == 0).all()
```

---

# Step 5: Text Encoder

```text
Claude Code Prompt 5.1: Create src/model/text_encoder.py exactly as the reference below.
 - Phone embedding + per-token LANGUAGE embedding (cs/sk/en); speaker vector added at the input.
 - Every convolution masked; padding invariance must be exact (test).
 - Outputs hidden h (for the duration predictor) and the prior mean mu_x in normalised mel space
   (for MAS and the prior loss).
 - Mask convention in the whole model: float (B, 1, T), 1 = valid.
Deferred: a Czech/Slovak phoneme-level BERT (the PL-BERT idea). The one directly relevant study
(TSD 2025, "Evaluating Phoneme-Level Pretraining in Czech TTS") could not be read; revisit it
after the baseline works.
```

### Reference: `src/model/text_encoder.py` (tested)

```python
# src/model/text_encoder.py
"""
Text encoder (Matcha-TTS style) with the v2 additions.

Changes relative to v1:
  * per-token LANGUAGE embedding (cs/sk/en) added to the phone embedding, so English names
    inside Slovak text are encoded as English (the IndexTTS 2.5 approach)
  * every convolution is MASKED: v1 added positional encodings to padded positions and ran
    unmasked convolutions over them, so a sentence's encoding changed with batch padding
  * outputs the prior mean mu_x in (normalised) mel space for Monotonic Alignment Search and
    the prior loss, as in Matcha-TTS; v1 had an alignment module nothing ever trained
Mask convention everywhere in the model: float tensor (B, 1, T), 1 = valid, 0 = padding.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def sequence_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    max_len = int(max_len or lengths.max())
    return (torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]).float().unsqueeze(1)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 7, expansion: int = 4):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * expansion)
        self.pw2 = nn.Linear(dim * expansion, dim)
        self.gamma = nn.Parameter(torch.full((dim,), 1e-6))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # x: (B, D, T)
        h = self.dwconv(x * mask).transpose(1, 2)
        h = self.pw2(torch.nn.functional.gelu(self.pw1(self.norm(h))))
        return (x + (self.gamma * h).transpose(1, 2)) * mask


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # x: (B, D, T)
        h = x.transpose(1, 2)
        pad = mask.squeeze(1) == 0
        a, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
        h = self.norm1(h + self.drop(a))
        h = self.norm2(h + self.drop(self.ff(h)))
        return h.transpose(1, 2) * mask


class TextEncoder(nn.Module):
    def __init__(self, n_vocab: int, n_langs: int = 3, n_mels: int = 100, dim: int = 192,
                 n_heads: int = 2, ff_dim: int = 768, n_layers: int = 6, n_convnext: int = 2,
                 cond_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(n_vocab, dim, padding_idx=0)
        self.lang_emb = nn.Embedding(n_langs, dim)
        nn.init.normal_(self.emb.weight, 0.0, dim ** -0.5)
        nn.init.normal_(self.lang_emb.weight, 0.0, dim ** -0.5)
        self.cond_proj = nn.Linear(cond_dim, dim) if cond_dim else None
        self.prenet = nn.ModuleList([nn.Conv1d(dim, dim, 5, padding=2) for _ in range(3)])
        self.prenet_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.blocks = nn.ModuleList([TransformerBlock(dim, n_heads, ff_dim, dropout) for _ in range(n_layers)])
        self.convnext = nn.ModuleList([ConvNeXtBlock(dim) for _ in range(n_convnext)])
        self.proj_mu = nn.Conv1d(dim, n_mels, 1)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def positional(T: int, dim: int, device) -> torch.Tensor:
        pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(T, dim, device=device)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        return pe.t().unsqueeze(0)  # (1, D, T)

    def forward(self, phoneme_ids: torch.Tensor, lang_ids: torch.Tensor, lengths: torch.Tensor,
                cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns hidden h (B, D, T), prior mean mu_x (B, n_mels, T) and x_mask (B, 1, T)."""
        mask = sequence_mask(lengths, phoneme_ids.shape[1])
        x = (self.emb(phoneme_ids) + self.lang_emb(lang_ids)) * math.sqrt(self.dim)
        x = x.transpose(1, 2)
        if self.cond_proj is not None and cond is not None:
            x = x + self.cond_proj(cond).unsqueeze(-1)
        x = (x + self.positional(x.shape[-1], self.dim, x.device)) * mask
        for conv, norm in zip(self.prenet, self.prenet_norm):
            x = x + self.drop(torch.relu(norm(conv(x * mask).transpose(1, 2)).transpose(1, 2)))
            x = x * mask
        for blk in self.blocks:
            x = blk(x, mask)
        for blk in self.convnext:
            x = blk(x, mask)
        return x, self.proj_mu(x) * mask, mask
```

---

# Step 6: Durations, MAS and Length Regulation

The fact that organises this whole plan:

- MAS and the duration predictor fix **content** (which phone plays when).
- The duration predictor alone fixes **tempo**.
- The decoder and vocoder determine **timbre and audio quality**.

The decoder never sees generated audio when it decides timing, so the self-reinforcing drift of autoregressive models cannot occur. Two risks remain, and both are handled explicitly:

- **"Near the end, go slower".** The encoder sees where the input ends. If training clips end with paragraph-final cadence, the predictor learns to slow down at the end. It is fixed by the boundary flag and identical segmentation for training and inference.
- **Session tempo leaking in as noise.** Tempo that differs between recording sessions and that the predictor cannot explain gets attached to text features. It is fixed by the rate inputs (Step 4.4).

```text
Claude Code Prompt 6.1: Create src/model/duration.py exactly as the reference below.
 - DurationPredictor: channel LayerNorm (v1's LayerNorm crashed for T != hidden), masking,
   DETACHED encoder input, FiLM conditioning on [narrator, tempo, boundary] (identity at init).
 - MAS: maximise the Gaussian log-likelihood of mel frames under mu_x (Matcha-TTS); numba DP;
   every frame -> exactly one token, every token >= 1 frame; raise if frames < tokens.
 - Length regulation: durations = ceil(exp(logw) * length_scale * token_scale), clamped to >= 1
   frame for every real token AFTER scaling. Never pass a target length at inference.
   token_scale lets the verify-and-retry loop (Step 12) slow down suspect words only.
Keep durations DETERMINISTIC at inference. If duration RL is added later (Step 11, stage 5),
sample around the mean only during rollouts and deploy the mean.
```

### Reference: `src/model/duration.py` (tested)

```python
# src/model/duration.py
"""
Durations: predictor, Monotonic Alignment Search (MAS) and length regulation.

Fixes relative to v1:
  * LENGTH REGULATION: v1 did round(d).clamp(min=0) and dropped tokens with 0 frames —
    torch.round rounds half to even, so every token predicted at <= 0.5 frames vanished
    (a skipped phoneme; several in a row = a skipped word). Here every real token gets
    at least 1 frame AFTER any length scaling: ceil(exp(logw) * length_scale).clamp(min=1).
  * DURATION PREDICTOR: v1 applied nn.LayerNorm(hidden) to a (B, hidden, T) tensor (it
    normalises the LAST dim, so it crashed whenever T != hidden) and ran unmasked convs.
    Here: channel LayerNorm, masking, detached encoder input (as in Matcha-TTS), and FiLM
    conditioning on narrator / tempo / boundary so tempo is an explicit input.
  * MAS: v1 aligned with a learned attention module that no loss trained. Here MAS maximises
    the Gaussian log-likelihood of the mel frames under the encoder's prior means (Glow-TTS /
    Matcha-TTS), with an O(T_text * T_mel) numba DP instead of a pure-Python double loop.
"""
from __future__ import annotations

import math
from typing import Optional

import numba
import numpy as np
import torch
import torch.nn as nn


class ChannelLayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class DurationPredictor(nn.Module):
    """Predicts log-durations (log frames). Tempo enters ONLY here (see Step 6 / pace rules)."""

    def __init__(self, in_dim: int, hidden: int = 256, kernel: int = 3, cond_dim: int = 0,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.films = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(nn.Conv1d(in_dim if i == 0 else hidden, hidden, kernel, padding=kernel // 2))
            self.norms.append(ChannelLayerNorm(hidden))
            self.films.append(nn.Linear(cond_dim, 2 * hidden) if cond_dim else None)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Conv1d(hidden, 1, 1)
        for film in self.films:  # start as identity: gamma = 1, beta = 0
            if film is not None:
                nn.init.zeros_(film.weight); nn.init.zeros_(film.bias)

    def forward(self, h: torch.Tensor, x_mask: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = h.detach()  # duration loss must not shape the encoder (Glow-TTS / Matcha practice)
        for conv, norm, film in zip(self.convs, self.norms, self.films):
            x = norm(torch.relu(conv(x * x_mask)))
            if film is not None and cond is not None:
                g, b = film(cond).unsqueeze(-1).chunk(2, dim=1)
                x = x * (1 + g) + b
            x = self.drop(x)
        return self.out(x * x_mask) * x_mask  # (B, 1, T) log-durations


# ---------------------------------------------------------------------------------------
# Monotonic Alignment Search
# ---------------------------------------------------------------------------------------

def gaussian_log_prior(mu_x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """log N(y_j | mu_x_i, I) for every (text i, frame j): (B, T_x, T_y). Matcha-TTS formulation."""
    n_feats = mu_x.shape[1]
    const = -0.5 * math.log(2 * math.pi) * n_feats
    factor = -0.5 * torch.ones_like(mu_x)
    y_sq = torch.matmul(factor.transpose(1, 2), y ** 2)
    y_mu = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
    mu_sq = torch.sum(factor * mu_x ** 2, 1).unsqueeze(-1)
    return y_sq - y_mu + mu_sq + const


@numba.njit(cache=True)
def _maximum_path_single(value: np.ndarray, t_x: int, t_y: int) -> np.ndarray:
    """Best monotonic path: every frame -> exactly one token, every token >= 1 frame."""
    neg_inf = -1e9
    q = np.full((t_x, t_y), neg_inf, dtype=np.float64)
    for j in range(t_y):
        for i in range(max(0, t_x + j - t_y), min(t_x, j + 1)):
            stay = q[i, j - 1] if j > 0 else (0.0 if i == 0 else neg_inf)
            move = q[i - 1, j - 1] if (i > 0 and j > 0) else neg_inf
            q[i, j] = value[i, j] + max(stay, move)
    path = np.zeros((t_x, t_y), dtype=np.float32)
    i = t_x - 1
    for j in range(t_y - 1, -1, -1):
        path[i, j] = 1.0
        if i > 0 and (i == j or q[i, j - 1] < q[i - 1, j - 1]):
            i -= 1
    return path


def maximum_path(log_prior: torch.Tensor, x_lengths: torch.Tensor, y_lengths: torch.Tensor) -> torch.Tensor:
    """Hard alignment (B, T_x, T_y) from log-likelihoods. Requires T_y >= T_x (see filter_manifest)."""
    lp = log_prior.detach().float().cpu().numpy()
    out = np.zeros(lp.shape, dtype=np.float32)
    for b in range(lp.shape[0]):
        tx, ty = int(x_lengths[b]), int(y_lengths[b])
        if ty < tx:
            raise ValueError(f"item {b}: {ty} frames < {tx} tokens; MAS needs >= 1 frame per token")
        out[b, :tx, :ty] = _maximum_path_single(lp[b, :tx, :ty].astype(np.float64), tx, ty)
    return torch.from_numpy(out).to(log_prior.device)


# ---------------------------------------------------------------------------------------
# Length regulation (inference)
# ---------------------------------------------------------------------------------------

def durations_from_logw(logw: torch.Tensor, x_mask: torch.Tensor, length_scale: float = 1.0,
                        token_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Integer frame counts; every real token gets >= 1 frame AFTER scaling (the v1 fix).
    length_scale: one scalar per language/book (>1 = slower). token_scale: optional (B, T_x)
    per-token multipliers, used by the verify-and-retry loop to slow down suspect words only.
    """
    w = torch.exp(logw) * length_scale
    if token_scale is not None:
        w = w * token_scale.unsqueeze(1)
    return torch.clamp(torch.ceil(w), min=1.0) * x_mask  # (B, 1, T_x)


def path_from_durations(durations: torch.Tensor, x_mask: torch.Tensor, max_frames: Optional[int] = None) -> torch.Tensor:
    """Monotonic hard path (B, T_x, T_y) from integer durations (B, 1, T_x)."""
    d = durations.squeeze(1).long()
    cum = torch.cumsum(d, dim=1)
    t_y = int(max_frames or cum[:, -1].max())
    frames = torch.arange(t_y, device=d.device)[None, None, :]
    end = cum.unsqueeze(-1)
    start = (cum - d).unsqueeze(-1)
    path = ((frames >= start) & (frames < end)).float()
    return path * x_mask.transpose(1, 2)
```

---

# Step 7: Flow-Matching Decoder

```text
Claude Code Prompt 7.1: Create src/model/flow_matching.py exactly as the reference below.
 - Matcha-TTS convention: noise z at t=0, data x1 at t=1,
       y_t = (1 - (1 - sigma_min) t) z + t x1,   target u = x1 - (1 - sigma_min) z,
   Euler integration from t=0 to t=1 (~10 steps is enough for offline rendering).
 - Estimator: 1-D U-Net with a FIXED channel width per level (v1 doubled it per level),
   inputs [x_t, mu_y, global conditioning], masked everywhere.
 - Tests: an analytic oracle field must map noise to the data distribution, the training
   loss must be minimised by that field, and a tiny decoder must overfit one utterance.
Keep the U-Net at this scale. A DiT decoder only pays off with hundreds of hours or more of
training data; revisit that after multilingual pretraining, not before.
```

### Reference: `src/model/flow_matching.py` (tested)

```python
# src/model/flow_matching.py
"""
OT-CFM decoder (Matcha-TTS convention) with a fixed-width 1-D U-Net estimator.

Fixes relative to v1:
  * DIRECTION: v1 trained with data at t=0 and noise at t=1 (target x1 - x0) but sampled
    from noise at t=0 integrating forward, i.e. it ran the data->noise flow and could only
    output noise. Here, as in Matcha-TTS: noise z at t=0, data x1 at t=1,
        y_t = (1 - (1 - sigma_min) t) z + t x1,   u = x1 - (1 - sigma_min) z,
    and Euler integration from t=0 to t=1. sigma_min is actually used.
  * WIDTH: v1 doubled channels at every U-Net level (hidden * 2**i); with the config's
    n_layers: 6 that is an 8,192-channel bottleneck (~1e9 parameters). Here the width is
    fixed per level, as in Matcha-TTS.
  * every convolution and norm is masked; lengths are padded to a multiple of 2**levels.
Optional robustness objectives (off by default; enable after the baseline works):
  * condition dropout -> classifier-free guidance at inference and per-sample
    conditional/unconditional losses for Self-Purifying Flow Matching (SPFM)
  * RobustSpeechFlow-style negatives (repeat / skip corruptions of the target), reimplemented
    from the paper's description (no official code) — validate with an ablation.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = 1000.0 * t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, dim: int, t_dim: int, groups: int = 8):
        super().__init__()
        self.n1, self.n2 = nn.GroupNorm(groups, dim), nn.GroupNorm(groups, dim)
        self.c1 = nn.Conv1d(dim, dim, 3, padding=1)
        self.c2 = nn.Conv1d(dim, dim, 3, padding=1)
        self.t = nn.Linear(t_dim, 2 * dim)

    def forward(self, x, mask, temb):
        h = self.c1(F.silu(self.n1(x * mask)) * mask)
        scale, shift = self.t(temb).unsqueeze(-1).chunk(2, dim=1)
        h = self.n2(h * mask) * (1 + scale) + shift
        h = self.c2(F.silu(h) * mask)
        return (x + h) * mask


class UNetEstimator(nn.Module):
    """v(x_t, t | mu_y, cond) with a fixed channel width at every level."""

    def __init__(self, n_mels: int, cond_dim: int = 0, dim: int = 256, levels: int = 2, blocks_per_level: int = 1):
        super().__init__()
        self.levels = levels
        t_dim = dim * 4
        self.t_mlp = nn.Sequential(nn.Linear(dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.dim = dim
        self.inp = nn.Conv1d(2 * n_mels + cond_dim, dim, 1)
        self.down = nn.ModuleList([nn.ModuleList([ResBlock(dim, t_dim) for _ in range(blocks_per_level)]) for _ in range(levels)])
        self.downsample = nn.ModuleList([nn.Conv1d(dim, dim, 3, stride=2, padding=1) for _ in range(levels)])
        self.mid = nn.ModuleList([ResBlock(dim, t_dim) for _ in range(2)])
        self.upsample = nn.ModuleList([nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1) for _ in range(levels)])
        self.up_fuse = nn.ModuleList([nn.Conv1d(2 * dim, dim, 1) for _ in range(levels)])
        self.up = nn.ModuleList([nn.ModuleList([ResBlock(dim, t_dim) for _ in range(blocks_per_level)]) for _ in range(levels)])
        self.out = nn.Sequential(nn.GroupNorm(8, dim), nn.SiLU(), nn.Conv1d(dim, n_mels, 1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, mask, mu, t, cond: Optional[torch.Tensor] = None):
        T = x.shape[-1]
        mult = 2 ** self.levels
        pad = (mult - T % mult) % mult
        if pad:
            x, mu, mask = (F.pad(a, (0, pad)) for a in (x, mu, mask))
        h = [x, mu]
        if cond is not None:
            h.append(cond.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        h = self.inp(torch.cat(h, dim=1)) * mask
        temb = self.t_mlp(timestep_embedding(t, self.dim))
        skips, masks = [], [mask]
        for blocks, ds in zip(self.down, self.downsample):
            for b in blocks:
                h = b(h, masks[-1], temb)
            skips.append(h)
            h = ds(h * masks[-1])
            masks.append(masks[-1][:, :, ::2])
            h = h * masks[-1]
        for b in self.mid:
            h = b(h, masks[-1], temb)
        for i, (us, fuse, blocks) in enumerate(zip(self.upsample, self.up_fuse, self.up)):
            masks.pop()
            h = us(h * 1.0)[:, :, : masks[-1].shape[-1]] * masks[-1]
            h = fuse(torch.cat([h, skips.pop()], dim=1)) * masks[-1]
            for b in blocks:
                h = b(h, masks[-1], temb)
        v = self.out(h * mask) * mask
        return v[..., :T]


class CFMDecoder(nn.Module):
    def __init__(self, n_mels: int, cond_dim: int = 0, dim: int = 256, levels: int = 2,
                 sigma_min: float = 1e-4, p_uncond: float = 0.0):
        super().__init__()
        self.n_mels = n_mels
        self.sigma_min = sigma_min
        self.p_uncond = p_uncond  # > 0 enables CFG / SPFM (optional, Step 7.3)
        self.estimator = UNetEstimator(n_mels, cond_dim, dim, levels)

    # ---- training ------------------------------------------------------------------------
    def loss(self, x1: torch.Tensor, mask: torch.Tensor, mu: torch.Tensor, cond: Optional[torch.Tensor] = None,
             drop_cond: Optional[torch.Tensor] = None, per_sample: bool = False,
             negatives: Optional[torch.Tensor] = None, neg_weight: float = 0.0):
        """
        x1: target mel (B, n_mels, T); mu: aligned prior mean mu_y (B, n_mels, T).
        drop_cond: (B,) bool, replace mu with zeros for those items (unconditional branch).
        negatives: optional corrupted targets (B, n_mels, T) for contrastive robustness.
        """
        B = x1.shape[0]
        t = torch.rand(B, device=x1.device)
        z = torch.randn_like(x1)
        tt = t[:, None, None]
        y = (1 - (1 - self.sigma_min) * tt) * z + tt * x1
        u = x1 - (1 - self.sigma_min) * z
        if drop_cond is None and self.training and self.p_uncond > 0:
            drop_cond = torch.rand(B, device=x1.device) < self.p_uncond
        if drop_cond is not None:
            mu = mu * (~drop_cond).float()[:, None, None]
        v = self.estimator(y, mask, mu, t, cond)
        denom = mask.sum(dim=(1, 2)) * self.n_mels
        l_pos = ((v - u) ** 2 * mask).sum(dim=(1, 2)) / denom
        out = l_pos
        if negatives is not None and neg_weight > 0:
            u_neg = negatives - (1 - self.sigma_min) * z
            l_neg = ((v - u_neg) ** 2 * mask).sum(dim=(1, 2)) / denom
            out = l_pos - neg_weight * l_neg
        return out if per_sample else out.mean()

    # ---- inference -----------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, mu: torch.Tensor, mask: torch.Tensor, n_steps: int = 10, temperature: float = 1.0,
               cond: Optional[torch.Tensor] = None, guidance: float = 0.0) -> torch.Tensor:
        """Euler from t=0 (noise) to t=1 (data). guidance > 0 needs a model trained with p_uncond > 0."""
        x = torch.randn_like(mu) * temperature
        ts = torch.linspace(0, 1, n_steps + 1, device=mu.device)
        for i in range(n_steps):
            t = ts[i].expand(mu.shape[0])
            v = self.estimator(x, mask, mu, t, cond)
            if guidance > 0:
                v_u = self.estimator(x, mask, torch.zeros_like(mu), t, cond)
                v = v + guidance * (v - v_u)
            x = x + (ts[i + 1] - ts[i]) * v
        return x * mask


# ---- optional robustness helpers --------------------------------------------------------

def repeat_skip_negatives(x1: torch.Tensor, lengths: torch.Tensor, floor: float = 0.0,
                          repeat_frac=(0.2, 0.4)) -> torch.Tensor:
    """
    Length-preserving corruptions of the target that look like TTS failures
    (RobustSpeechFlow, May 2026): half the batch gets a 'repeat' (a 20-40% span overwritten
    with content from another span), half a 'skip' (the tail shifted forward, end padded
    with silence `floor`, i.e. the normalised value of a silent frame).
    """
    neg = x1.clone()
    for b in range(x1.shape[0]):
        n = int(lengths[b])
        span = max(1, int(n * torch.empty(1).uniform_(*repeat_frac).item()))
        if n <= span + 1:
            continue
        if b % 2 == 0:  # repeat
            src = torch.randint(0, n - span, (1,)).item()
            dst = torch.randint(0, n - span, (1,)).item()
            neg[b, :, dst:dst + span] = x1[b, :, src:src + span]
        else:           # skip
            start = torch.randint(0, n - span, (1,)).item()
            tail = x1[b, :, start + span:n]
            neg[b, :, start:start + tail.shape[-1]] = tail
            neg[b, :, start + tail.shape[-1]:n] = floor
    return neg


def spfm_route(loss_cond: torch.Tensor, loss_uncond: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """
    Self-Purifying Flow Matching (arXiv 2509.19091): items whose conditional loss is not better
    than their unconditional loss look mislabeled (text does not match audio) -> train them
    unconditionally. Returns a (B,) bool mask of items to route to the unconditional branch.
    """
    return loss_cond > loss_uncond - margin
```

## 7.3 Optional robustness objectives (enable after the baseline passes Step 14)

Enable each of these as an ablation and keep it only if it lowers the failure metrics in Step 14:

- **Repeat/skip negatives** (RobustSpeechFlow, Supertone, May 2026). The paper's loss is `ℒ = ℒ_pos − 0.2·ℒ_rand − 0.2·ℒ_aug`: random other-utterance negatives plus augmented negatives. The augmented negatives are length-preserving corruptions of the target: a 20–40% span repeated, or the tail shifted forward and padded with silence. The paper reports English CER falling from 0.48% to 0.35% and Korean from 0.81% to 0.57%. There is no official code; `repeat_skip_negatives` re-implements the augmented term from the paper's description. Usage: `model(batch, negatives=repeat_skip_negatives(mels, mel_lengths, floor=silence), negatives_weight=0.2)`, with `silence = (log(1e-5) − mel_mean) / mel_std`.
- **Condition dropout** (`p_uncond: 0.1–0.2`). It trains an unconditional branch, which enables:
  - classifier-free guidance at inference (`guidance > 0`);
  - **Self-Purifying Flow Matching** (arXiv 2509.19091, Sep 2025). Compare each item's conditional and unconditional loss; items where the text does not help (`spfm_route`) are trained unconditionally. Audiobook takes where the narrator departed from the text are exactly those items. An SPFM-based entry had the lowest WER in the WildSpoof 2026 TTS track.
- **Few-step distillation:** skip it. Offline rendering on H200/B200 does not need it, and distilled students risk flatter prosody.

## Test 5–7: Model Component Tests (tested, 22 tests)

```python
# tests/test_model_components.py
"""
Model tests that would have caught the v1 defects:
  - length regulator dropping short tokens (defect 1)
  - flow-matching sampler integrating the wrong way (defect 3)
  - duration predictor LayerNorm crash, unmasked encoder convs, untrained aligner (found while fixing)
"""
import itertools
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.model.duration import (DurationPredictor, durations_from_logw, gaussian_log_prior,
                                maximum_path, path_from_durations)
from src.model.flow_matching import CFMDecoder, repeat_skip_negatives, spfm_route
from src.model.text_encoder import TextEncoder, sequence_mask

torch.manual_seed(0)


class TestTextEncoder:
    def make(self):
        return TextEncoder(n_vocab=60, n_mels=20, dim=64, n_heads=2, ff_dim=128, n_layers=2, dropout=0.0).eval()

    def test_shapes(self):
        enc = self.make()
        h, mu, mask = enc(torch.randint(1, 60, (2, 17)), torch.zeros(2, 17, dtype=torch.long), torch.tensor([17, 9]))
        assert h.shape == (2, 64, 17) and mu.shape == (2, 20, 17) and mask.shape == (2, 1, 17)
        assert (mu[1, :, 9:] == 0).all()

    def test_padding_invariance_is_exact(self):
        """v1 leaked padding through unmasked convs (its test tolerated a 0.5 mean difference)."""
        enc = self.make()
        ids = torch.randint(1, 60, (1, 12))
        langs = torch.zeros(1, 12, dtype=torch.long)
        h1, mu1, _ = enc(ids, langs, torch.tensor([12]))
        ids_p = torch.cat([ids, torch.zeros(1, 20, dtype=torch.long)], dim=1)
        h2, mu2, _ = enc(ids_p, torch.zeros(1, 32, dtype=torch.long), torch.tensor([12]))
        assert torch.allclose(mu1, mu2[:, :, :12], atol=1e-5)

    def test_language_embedding_changes_encoding(self):
        enc = self.make()
        ids = torch.randint(1, 60, (1, 8))
        a, _, _ = enc(ids, torch.zeros(1, 8, dtype=torch.long), torch.tensor([8]))
        b, _, _ = enc(ids, torch.full((1, 8), 2, dtype=torch.long), torch.tensor([8]))
        assert not torch.allclose(a, b)


class TestDurationPredictor:
    def test_any_length(self):
        """v1: nn.LayerNorm(hidden) on (B, hidden, T) crashed whenever T != hidden."""
        dp = DurationPredictor(in_dim=64, hidden=32)
        for T in (1, 7, 31, 200):
            out = dp(torch.randn(2, 64, T), torch.ones(2, 1, T))
            assert out.shape == (2, 1, T)

    def test_masked_positions_are_zero(self):
        dp = DurationPredictor(in_dim=64, hidden=32)
        mask = sequence_mask(torch.tensor([10, 4]), 10)
        out = dp(torch.randn(2, 64, 10), mask)
        assert (out[1, :, 4:] == 0).all()

    def test_encoder_input_is_detached(self):
        dp = DurationPredictor(in_dim=16, hidden=16)
        h = torch.randn(1, 16, 5, requires_grad=True)
        dp(h, torch.ones(1, 1, 5)).sum().backward()
        assert h.grad is None

    def test_tempo_conditioning_moves_durations(self):
        dp = DurationPredictor(in_dim=16, hidden=16, cond_dim=4)
        for film in dp.films:
            nn.init.normal_(film.weight, std=0.5)
        h = torch.randn(1, 16, 6)
        a = dp(h, torch.ones(1, 1, 6), torch.tensor([[1.0, 0, 0, 0]]))
        b = dp(h, torch.ones(1, 1, 6), torch.tensor([[-1.0, 0, 0, 0]]))
        assert not torch.allclose(a, b)


def brute_force_best(value: np.ndarray):
    tx, ty = value.shape
    best, best_path = -np.inf, None
    for cuts in itertools.combinations(range(1, ty), tx - 1):  # token boundaries
        bounds = (0, *cuts, ty)
        score = sum(value[i, bounds[i]:bounds[i + 1]].sum() for i in range(tx))
        if score > best:
            best, best_path = score, bounds
    return best, best_path


class TestMAS:
    @pytest.mark.parametrize("tx,ty,seed", [(1, 5, 0), (3, 3, 1), (3, 7, 2), (4, 9, 3), (5, 8, 4)])
    def test_matches_exhaustive_search(self, tx, ty, seed):
        rng = np.random.default_rng(seed)
        value = rng.standard_normal((tx, ty)).astype(np.float32)
        path = maximum_path(torch.from_numpy(value)[None], torch.tensor([tx]), torch.tensor([ty]))[0].numpy()
        best, _ = brute_force_best(value)
        assert math.isclose(float((path * value).sum()), best, rel_tol=1e-5, abs_tol=1e-5)
        assert (path.sum(0) == 1).all()          # every frame -> exactly one token
        assert (path.sum(1) >= 1).all()          # every token >= 1 frame
        firsts = path.argmax(1)
        assert (np.diff(firsts) > 0).all()       # monotonic

    def test_rejects_fewer_frames_than_tokens(self):
        with pytest.raises(ValueError):
            maximum_path(torch.randn(1, 5, 3), torch.tensor([5]), torch.tensor([3]))

    def test_log_prior_matches_direct_gaussian(self):
        mu, y = torch.randn(1, 4, 3), torch.randn(1, 4, 5)
        lp = gaussian_log_prior(mu, y)
        direct = torch.distributions.Normal(mu.transpose(1, 2).unsqueeze(2), 1.0).log_prob(
            y.transpose(1, 2).unsqueeze(1)).sum(-1)
        assert torch.allclose(lp, direct, atol=1e-4)


class TestLengthRegulation:
    def test_no_token_is_dropped(self):
        """v1: round() + dur > 0 filter made 0.49, 0.5 and tiny predictions vanish."""
        frames = torch.tensor([[3.2, 0.49, 0.5, 0.51, 1.5, 2.5, 0.01]])
        logw = torch.log(frames).unsqueeze(1)
        mask = torch.ones(1, 1, 7)
        for scale in (1.0, 0.5, 0.25, 1.7):
            d = durations_from_logw(logw, mask, scale)
            assert (d >= 1).all(), (scale, d)

    def test_padding_tokens_get_zero_frames(self):
        mask = sequence_mask(torch.tensor([3]), 5)
        d = durations_from_logw(torch.zeros(1, 1, 5), mask)
        assert d[0, 0].tolist() == [1, 1, 1, 0, 0]

    def test_path_is_monotonic_and_complete(self):
        d = torch.tensor([[[2.0, 1.0, 3.0, 0.0]]])
        mask = torch.tensor([[[1.0, 1.0, 1.0, 0.0]]])
        p = path_from_durations(d, mask)[0]
        assert p.shape == (4, 6)
        assert p.sum(1).tolist() == [2, 1, 3, 0] and (p.sum(0) == 1).all()
        assert p[0, :2].all() and p[1, 2] == 1 and p[2, 3:].all()


class GaussianOracle(nn.Module):
    """Exact optimal velocity for 1-D data N(m, s^2) under the Matcha convention."""

    def __init__(self, m, s, sigma_min):
        super().__init__()
        self.m, self.s, self.k = m, s, 1 - sigma_min

    def forward(self, x, mask, mu, t, cond=None):
        t = t[:, None, None]
        a = 1 - self.k * t
        var = a ** 2 + (t * self.s) ** 2
        return self.m + (t * self.s ** 2 - self.k * a) / var * (x - t * self.m)


class TestFlowMatchingDirection:
    def test_sampler_transports_noise_to_data(self):
        """With the exact optimal field, sampling must recover the data distribution.
        v1's sampler, fed the same kind of field, produced N(+2.5, 0.5^2) from N(-5, 2^2) data."""
        dec = CFMDecoder(n_mels=1, dim=8, levels=1)
        dec.estimator = GaussianOracle(-5.0, 2.0, dec.sigma_min)
        torch.manual_seed(0)
        x = dec.sample(torch.zeros(1, 1, 20000), torch.ones(1, 1, 20000), n_steps=200, temperature=1.0)
        assert abs(x.mean().item() + 5.0) < 0.1 and abs(x.std().item() - 2.0) < 0.1

    def test_training_target_matches_sampler(self):
        """The field the sampler needs must be the minimiser of the training loss (same convention)."""
        dec = CFMDecoder(n_mels=1, dim=8, levels=1)
        x1 = -5.0 + 2.0 * torch.randn(1, 1, 50000)
        mask = torch.ones(1, 1, 50000)

        class Shifted(nn.Module):
            def __init__(self, base, delta):
                super().__init__(); self.base, self.delta = base, delta

            def forward(self, *a):
                return self.base(*a) + self.delta

        def loss_with(field):
            dec.estimator = field
            torch.manual_seed(1)
            return dec.loss(x1, mask, torch.zeros_like(x1)).item()

        oracle = GaussianOracle(-5.0, 2.0, dec.sigma_min)
        l_oracle = loss_with(oracle)
        assert l_oracle < loss_with(GaussianOracle(+5.0, 2.0, dec.sigma_min))  # mirrored field
        assert l_oracle < loss_with(Shifted(oracle, 0.3)) and l_oracle < loss_with(Shifted(oracle, -0.3))

    def test_overfits_one_utterance(self):
        """A tiny decoder must learn to reconstruct a fixed mel from its prior: catches any sign/direction bug."""
        torch.manual_seed(0)
        n_mels, T = 16, 32
        target = torch.sin(torch.linspace(0, 6, T))[None, None, :] * torch.linspace(0.5, 1.5, n_mels)[None, :, None]
        mu = target + 0.3 * torch.randn_like(target)  # a noisy prior, like an early encoder
        mask = torch.ones(1, 1, T)
        dec = CFMDecoder(n_mels, dim=64, levels=2)
        opt = torch.optim.Adam(dec.parameters(), lr=2e-3)
        for _ in range(400):
            opt.zero_grad()
            dec.loss(target.expand(8, -1, -1), mask.expand(8, -1, -1), mu.expand(8, -1, -1)).backward()
            opt.step()
        dec.eval()
        out = dec.sample(mu, mask, n_steps=16, temperature=0.5)
        err = ((out - target) ** 2).mean().item()
        assert err < 0.05 * target.var().item() + 0.02, err


class TestRobustnessHelpers:
    def test_negatives_preserve_length_and_differ(self):
        x = torch.randn(4, 10, 40)
        neg = repeat_skip_negatives(x, torch.tensor([40, 40, 30, 30]))
        assert neg.shape == x.shape
        assert all(not torch.allclose(neg[b], x[b]) for b in range(4))

    def test_spfm_routing(self):
        route = spfm_route(torch.tensor([0.2, 0.9]), torch.tensor([0.5, 0.6]))
        assert route.tolist() == [False, True]
```

---

# Step 8: Model Assembly and Narrator/Tempo Conditioning

v1 referenced `MatchaTTS` and `compute_total_loss` in its tests but never specified them, and nothing trained its aligner. The assembly below is Matcha-TTS with three conditioning inputs:

| Input | Enters | Why |
|---|---|---|
| **Narrator** (learned speaker table; one row per pretraining speaker plus the narrator) | Encoder, duration predictor, decoder | Replaces per-utterance ECAPA vectors. In a single-narrator setting those encode channel, session and language rather than voice. They would also force a choice of reference vector at inference, another source of drift. |
| **Tempo** (articulation rate + pause ratio) | Duration predictor **only** | Tempo is an explicit input, fixed at inference |
| **Boundary** (mid / paragraph-final / chapter-final) | Duration predictor + decoder | Paragraph cadence comes from the flag, not from "end of input" |

The losses are Matcha-TTS's three:

- **Duration:** MSE between the predicted and the MAS log-durations.
- **Prior:** Gaussian NLL of the mel frames under the aligned `mu_y`.
- **CFM:** the flow-matching loss.

```text
Claude Code Prompt 8.1: Create src/model/matcha_tts.py exactly as the reference below, and
tests/test_integration.py (the end-to-end synthetic test in the Appendix).
Speaker evaluation (not conditioning): ReDimNet2-B6 (MIT, 0.29% VoxCeleb1-O EER) or a
w2v-BERT 2.0 verifier (0.12%) instead of SpeechBrain ECAPA (0.80%). Compare similarity only
within one language and with one extractor, and pair it with rhythm metrics (Step 14).
```

### Reference: `src/model/matcha_tts.py` (tested)

```python
# src/model/matcha_tts.py
"""
MatchaTTS assembly: encoder -> (MAS | duration predictor) -> CFM decoder, plus the three
Matcha-TTS losses. v1 referenced `MatchaTTS` and `compute_total_loss` in its tests but never
specified them, and nothing trained its alignment.

Conditioning (v2):
  * narrator: a learned speaker table (one row per pretraining speaker + the narrator);
    replaces per-utterance ECAPA vectors, which carry session/language information
  * tempo: utterance articulation rate + pause ratio -> DURATION PREDICTOR ONLY, so tempo is
    an explicit input that is fixed at inference (the ElevenLabs slow-mode problem)
  * boundary: mid-paragraph / paragraph-final / chapter-final sentence flag -> durations and
    decoder, so paragraph cadence comes from the flag, never from "end of input"
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .duration import DurationPredictor, durations_from_logw, gaussian_log_prior, maximum_path, path_from_durations
from .flow_matching import CFMDecoder
from .text_encoder import TextEncoder, sequence_mask


@dataclass
class MatchaConfig:
    n_vocab: int
    n_langs: int = 3
    n_speakers: int = 1
    n_boundaries: int = 3
    n_mels: int = 100
    enc_dim: int = 192
    enc_layers: int = 6
    enc_heads: int = 2
    enc_ff: int = 768
    spk_dim: int = 64
    cond_dim: int = 64
    dur_hidden: int = 256
    dec_dim: int = 256
    dec_levels: int = 2
    sigma_min: float = 1e-4
    p_uncond: float = 0.0
    dropout: float = 0.1
    rate_mean: float = 12.0   # phones/s, set from the corpus (Step 4.4)
    rate_std: float = 2.0


class MatchaTTS(nn.Module):
    def __init__(self, cfg: MatchaConfig):
        super().__init__()
        self.cfg = cfg
        self.spk = nn.Embedding(cfg.n_speakers, cfg.spk_dim)
        self.boundary = nn.Embedding(cfg.n_boundaries, cfg.cond_dim)
        self.rate_mlp = nn.Sequential(nn.Linear(2, cfg.cond_dim), nn.SiLU(), nn.Linear(cfg.cond_dim, cfg.cond_dim))
        self.encoder = TextEncoder(cfg.n_vocab, cfg.n_langs, cfg.n_mels, cfg.enc_dim, cfg.enc_heads, cfg.enc_ff,
                                   cfg.enc_layers, cond_dim=cfg.spk_dim, dropout=cfg.dropout)
        self.duration = DurationPredictor(cfg.enc_dim, cfg.dur_hidden, cond_dim=cfg.spk_dim + 2 * cfg.cond_dim,
                                          dropout=cfg.dropout)
        self.decoder = CFMDecoder(cfg.n_mels, cond_dim=cfg.spk_dim + cfg.cond_dim, dim=cfg.dec_dim,
                                  levels=cfg.dec_levels, sigma_min=cfg.sigma_min, p_uncond=cfg.p_uncond)

    # ---- conditioning --------------------------------------------------------------------
    def _conds(self, speaker_ids, rates, boundary_ids):
        spk = self.spk(speaker_ids)
        rate_n = torch.stack([(rates[:, 0] - self.cfg.rate_mean) / self.cfg.rate_std, rates[:, 1]], dim=1)
        dur_cond = torch.cat([spk, self.rate_mlp(rate_n), self.boundary(boundary_ids)], dim=1)
        dec_cond = torch.cat([spk, self.boundary(boundary_ids)], dim=1)
        return spk, dur_cond, dec_cond

    # ---- training ------------------------------------------------------------------------
    def forward(self, batch: Dict[str, torch.Tensor], negatives_weight: float = 0.0,
                negatives: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x, xl, y, yl = batch["phoneme_ids"], batch["phoneme_lengths"], batch["mels"], batch["mel_lengths"]
        spk, dur_cond, dec_cond = self._conds(batch["speaker_ids"], batch["rates"], batch["boundary_ids"])
        h, mu_x, x_mask = self.encoder(x, batch["lang_ids"], xl, spk)
        y_mask = sequence_mask(yl, y.shape[-1])

        with torch.no_grad():
            attn = maximum_path(gaussian_log_prior(mu_x, y), xl, yl)  # (B, T_x, T_y)

        logw = self.duration(h, x_mask, dur_cond)
        logw_target = torch.log(1e-8 + attn.sum(-1)).unsqueeze(1) * x_mask
        dur_loss = torch.sum((logw - logw_target) ** 2) / torch.sum(xl)

        mu_y = torch.matmul(attn.transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)  # (B, n_mels, T_y)
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask) / (
            torch.sum(y_mask) * self.cfg.n_mels)

        cfm_loss = self.decoder.loss(y, y_mask, mu_y, dec_cond, negatives=negatives, neg_weight=negatives_weight)
        return {"dur_loss": dur_loss, "prior_loss": prior_loss, "cfm_loss": cfm_loss,
                "loss": dur_loss + prior_loss + cfm_loss, "attn": attn}

    # ---- inference -----------------------------------------------------------------------
    @torch.no_grad()
    def synthesize(self, phoneme_ids, lang_ids, lengths, speaker_ids, rates, boundary_ids,
                   n_steps: int = 10, temperature: float = 0.667, length_scale: float = 1.0,
                   guidance: float = 0.0, token_scale: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        `rates` is the FIXED target tempo for every sentence of a book (narrator median),
        not something inferred per sentence. `length_scale` is one scalar per language/book.
        """
        spk, dur_cond, dec_cond = self._conds(speaker_ids, rates, boundary_ids)
        h, mu_x, x_mask = self.encoder(phoneme_ids, lang_ids, lengths, spk)
        logw = self.duration(h, x_mask, dur_cond)
        durations = durations_from_logw(logw, x_mask, length_scale, token_scale)  # >= 1 frame per token
        y_lengths = durations.sum(dim=(1, 2)).long()
        path = path_from_durations(durations, x_mask)                       # (B, T_x, T_y)
        y_mask = sequence_mask(y_lengths, path.shape[-1])
        mu_y = torch.matmul(path.transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)
        mel = self.decoder.sample(mu_y, y_mask, n_steps, temperature, dec_cond, guidance)
        return {"mel": mel, "mel_lengths": y_lengths, "durations": durations.squeeze(1), "path": path}
```

---

# Step 9: Multilingual SSL Discriminator

StyleTTS2's SLM discriminator is kept as an idea, but its backbone changes. WavLM Base+/Large were pretrained on 94k hours of **English** audio (Libri-Light, GigaSpeech and English VoxPopuli; the paper says "our focus is English-only audio"). SSL models "typically degrade on unfamiliar languages".

| Backbone | Pretraining data | Licence | Fit |
|---|---|---|---|
| **XLS-R-300M** (default) | 436k h, 128 languages incl. cs/sk | Apache-2.0 | Drop-in: same 16 kHz waveform input as WavLM |
| Omnilingual wav2vec 2.0 (300M/1B, Meta, Nov 2025) | 4.3M h, 1,600+ languages | Apache-2.0 | Drop-in; loads through fairseq2 |
| w2v-BERT 2.0 | 4.5M h, 143+ languages | MIT | Strongest encoder, but takes 80-bin fbank features: compute them in differentiable code (e.g. `torchaudio.compliance.kaldi.fbank(..., dither=0.0)`, which passes gradients) so generator gradients reach it |
| mHuBERT-147 | 90k h, 147 languages | CC-BY-NC-SA-4.0 | Small; Czech/Slovak hours unconfirmed |

No paper has yet swapped WavLM for a multilingual encoder in this discriminator. Treat the choice as an ablation: no SSL discriminator vs WavLM vs XLS-R vs w2v-BERT 2.0. Score each on Czech/Slovak CER, the Step 14 metrics and a small native-listener panel.

```text
Claude Code Prompt 9.1: Create src/model/discriminators.py (reference below) plus the MPD and
MRD discriminators imported from third_party/BigVGAN (discriminators.py, loss.py).
Wiring audit before the first adversarial run. When a team ported StyleTTS2 to German/Polish
(Kikiri-TTS, Apr 2026), they found bugs of exactly these kinds, so check each one:
  [ ] the real-audio branch of the SSL feature loss is actually computed (not deleted/None)
  [ ] GAN losses switch on at the intended step (not gated on a wrong epoch variable)
  [ ] the discriminator never sees garbage from a disabled module (e.g. an unused sampler)
  [ ] SSL backbone frozen and in eval mode; only the head trains
  [ ] gradient reaches the generator: generated waveform -> resample 24k->16k -> SSL -> head
How the flow decoder meets waveform discriminators (Step 11, stage 4): at a random t, take the
one-step endpoint estimate x1_hat = y_t + (1 - t) * v (it equals x1 + sigma_min*z at the optimum),
de-normalise, vocode with the FROZEN narrator-tuned BigVGAN, crop 1-2 s, then apply MPD + MRD +
SSL with LSGAN + feature-matching losses. Keep this phase short and late.
```

### Reference: `src/model/discriminators.py` (tested with a random-initialised XLS-R-architecture stand-in)

```python
# src/model/discriminators.py
"""
Multilingual SSL discriminator (replaces the English-only WavLM-Large backbone).

WavLM Base+/Large were pretrained on 94k hours of ENGLISH audio (Libri-Light, GigaSpeech,
English VoxPopuli; the paper: "our focus is English-only audio"). For Czech/Slovak the
default backbone is XLS-R-300M (436k h, 128 languages incl. cs/sk, Apache-2.0), which takes
the same 16 kHz waveform input, so it is a drop-in. Alternatives for the ablation (Step 9):
Omnilingual wav2vec 2.0 (Meta, Nov 2025, 1,600+ languages, Apache-2.0; loads via fairseq2)
and w2v-BERT 2.0 (4.5M h, MIT; needs a DIFFERENTIABLE 80-bin fbank front end).

StyleTTS2 recipe: frozen SSL encoder, trainable head over ALL hidden layers, LSGAN losses,
plus an L1 feature-matching term between real/generated SSL features for the generator.
Gradients must flow: generated waveform -> resample(24k->16k) -> SSL encoder -> head.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF


class SSLDiscriminator(nn.Module):
    def __init__(self, ssl_model: nn.Module, n_layers: int, hidden: int, head_dim: int = 256,
                 in_sr: int = 24000, ssl_sr: int = 16000, normalize_input: bool = True):
        """
        ssl_model: a HF Wav2Vec2Model-compatible module, e.g.
            Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")  # n_layers=25, hidden=1024
        n_layers: number of hidden states returned (transformer layers + 1)
        """
        super().__init__()
        self.ssl = ssl_model.eval()
        for p in self.ssl.parameters():
            p.requires_grad_(False)
        self.in_sr, self.ssl_sr, self.normalize_input = in_sr, ssl_sr, normalize_input
        self.head = nn.Sequential(
            nn.Conv1d(n_layers * hidden, head_dim, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(head_dim, head_dim, 5, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(head_dim, 1, 3, padding=1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.ssl.eval()  # the frozen encoder stays in eval mode (no dropout, fixed norms)
        return self

    def features(self, wav: torch.Tensor) -> torch.Tensor:
        """wav (B, T) at in_sr -> stacked hidden states (B, n_layers * hidden, frames)."""
        x = AF.resample(wav, self.in_sr, self.ssl_sr) if self.in_sr != self.ssl_sr else wav
        if self.normalize_input:  # XLS-R / wav2vec2-large feature extractors normalise per utterance
            x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-7)
        hs = self.ssl(x, output_hidden_states=True).hidden_states
        return torch.cat(hs, dim=-1).transpose(1, 2)

    def forward(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.features(wav)
        return self.head(feats), feats

    # ---- losses (LSGAN, as in StyleTTS2) ------------------------------------------------
    def discriminator_loss(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        d_real, _ = self(real)
        d_fake, _ = self(fake.detach())
        return torch.mean((1 - d_real) ** 2) + torch.mean(d_fake ** 2)

    def generator_loss(self, real: torch.Tensor, fake: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            _, f_real = self(real)
        d_fake, f_fake = self(fake)
        adv = torch.mean((1 - d_fake) ** 2)
        fm = F.l1_loss(f_fake, f_real)
        return adv, fm
```

---

# Step 10: BigVGAN Vocoder

The vocoder stays: no NVIDIA successor to BigVGAN v2 has appeared, and vocoders do not cause skipped or repeated words.

```text
Claude Code Prompt 10.1: Create src/vocoder/bigvgan.py:
 - load third_party/BigVGAN: bigvgan.BigVGAN.from_pretrained(
       "nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False); remove weight norm; eval
 - vocode(mel_normalised) = model(mel_normalised * mel_std + mel_mean)   # de-normalise first
 - assert the model's h.num_mels / hop_size / sampling_rate equal configs/model audio settings
 - fine-tuning entry point (Step 11, stage 3): BigVGAN's own train.py on (a) narrator audio and
   (b) "ground-truth-aligned" mels: run the trained acoustic model with MAS alignments
   (teacher-forced durations) on training utterances, so predicted mels line up with real audio.
Optional A/B: Flow2GAN (k2-fsa, ICLR 2026, Apache-2.0) has universal 24 kHz and 44 kHz
checkpoints. Its authors report PESQ 4.203 at 4 steps against BigVGAN-v2's 3.945 on their 24 kHz
universal test set. Verify that its mel settings match before comparing; decide by listening.
```

---

# Step 11: Training Pipeline

| Stage | What | Data | Exit criterion |
|---|---|---|---|
| 0 | Data preparation and audit (Step 4.2), G2P review (Step 3.2), vocabulary (Step 3.3) | Narrator + pretraining corpora | Manifest filtered; narrator CER baseline known; 0 unknown symbols |
| 1 | Multilingual pretraining: speaker table, language IDs, boundary flags; tempo labels bootstrapped (Step 4.4) | Filtered SloPalSpeech + ParlaSpeech-CZ/ParCzech4Speech + LibriTTS-R | Step 14 failure metrics flat on held-out parliament and LibriVox sets |
| 2 | Narrator fine-tune: add the narrator row to the speaker table; tempo labels on; lower LR | 10–40 h narrator | Pace gates M1–M4 pass; ASR failure rate at the narrator baseline |
| 3 | Vocoder fine-tune (Step 10) | Narrator audio + ground-truth-aligned mels | No audible artifacts on held-out narration |
| 4 | Short, late adversarial phase (Step 9): MPD/MRD + SSL discriminator through the frozen vocoder | Narrator | Keep only if a listening test prefers it and no failure metric regresses |
| 5 | RL last: GRPO on the duration predictor (DMOSpeech 2 pattern: AAAI 2026, inference code and checkpoints released, training code "under construction"), then optionally advantage-weighted decoder RL (GROW, Aug 2026; FlowTTS-GRPO, Jun 2026) | Narrator | Rewards must include a **rate-consistency term** (distance from the target rate, and the slope of rate against paragraph position) next to CER and similarity, or RL can drift toward slower "safer" durations |

```text
Claude Code Prompt 11.1: Create src/training/trainer.py and stages.py for stages 1-4, the entry
point scripts/train.py --config <stage config>, and one config per stage in configs/training/:
pretrain.yaml (1), finetune_narrator.yaml (2), vocoder.yaml (3, for the Prompt 10.1 fine-tuning
entry point) and adversarial.yaml (4). A stage config names its data manifests, the checkpoint it
starts from and only the training settings it overrides from configs/model/matcha_base.yaml
(stage 2: learning_rate 3e-5).
 - batches by total mel frames (bucketed by length), bf16, DDP, grad clip 1.0, EMA of weights
 - loss = duration + prior + CFM (Step 8); optional negatives / p_uncond from Step 7.3
 - log every N steps: losses, MAS duration histogram, % tokens with 1 frame, and every eval
   interval run Step 14's fast checks on 50 held-out sentences
 - stage 1 -> 2: load weights, extend the speaker embedding table by one row for the narrator,
   LR 3e-5, keep language/boundary embeddings trainable
 - stage 4 uses the endpoint-estimate trick from Step 9 and BigVGAN's discriminators
Checkpoint selection: failure metrics and pace metrics first, then TTSDS2 or MOS predictors,
which are unvalidated for Czech/Slovak, used only for relative same-text comparisons.
Rewards for stage 5 come from Step 14 components (ASR CER, speaker similarity, pace).
```

---

# Step 12: Inference Pipeline with ASR Verification

Rules that make pace consistent:

1. Keep the duration predictor deterministic.
2. Fix the tempo inputs to the narrator's median for every sentence of a book.
3. Calibrate one `length_scale` per language and hold it constant within a book.
4. Keep style out of the duration path. (If style diffusion is ever added, smooth it across sentences.)
5. Render sentence by sentence, with the same segmentation rules used for training.
6. Paragraph cadence comes only from the boundary flag.
7. Insert pauses by rule from the narrator's measured medians per boundary type (comma, sentence, paragraph, chapter).

```text
Claude Code Prompt 12.1: Create src/inference/synthesizer.py, scripts/synthesize_book.py and
configs/inference/inference.yaml (n_steps, temperature, the pause per boundary type from the
narrator's medians, the chapter loudness for masters, the ASR judges and the retry order; tempo
and length_scale stay in configs/model/matcha_base.yaml):
 1. normalise -> segment (boundary labels) -> G2P tokens (strict vocabulary) per sentence
 2. MatchaTTS.synthesize(..., rates=[inference_rate, median pause ratio], boundary=label,
    length_scale=config[lang], n_steps=10, temperature=0.667) -> BigVGAN
 3. insert rule-based pauses; loudness-normalise per chapter (-23 LUFS for masters; derive
    delivery formats from it)
 4. VERIFY every sentence with src/eval/asr_check.py against the narrator baseline CER
    (Parakeet-TDT-0.6B-v3 as the primary judge; a second model family as a tie-breaker:
    Canary-1B-v2 for Czech, the SloPal whisper-large-v3-sk fine-tune for Slovak, or Omnilingual
    LLM-ASR). Durations are deterministic, so a retry that only re-seeds the decoder cannot fix
    a duration-caused skip. Retry actions, in order: local token_scale x1.15 on the suspect
    words -> lexicon check -> re-split the sentence -> human review queue.
 5. Write a per-chapter QA report with src/eval/report.py (Prompt 14.1): failures, retries,
    pace metrics.
```

### Reference: `src/eval/asr_check.py` (tested)

```python
# src/eval/asr_check.py
"""
ASR verification for every rendered sentence (Step 12/14).

Why relative thresholds: the best open Czech/Slovak ASR still errs on 5.5-11% of words of
HUMAN speech (FLEURS: Parakeet-TDT-0.6B-v3 cs 11.01%, sk 8.82%), so absolute WER targets like
"2-4%" are unmeasurable. Compare against the SAME ASR on the narrator's real recordings, and
look for failure SHAPES that ASR noise does not produce: long runs of deleted words (skips),
long runs of inserted/substituted words (hallucinations), repeated n-grams, truncated endings.
Run-length definitions follow Argmax, "Taming Long-form TTS" (Sep 2026): skip = >= 10
contiguous deleted words, hallucination = >= 20 contiguous inserted/substituted words; for
sentence-level audiobook QA we use much stricter defaults (see CheckConfig).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

import jiwer


def normalize_for_asr(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"<[^>]+>", " ", text)              # drop <en> span tags
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class CheckConfig:
    max_cer_over_baseline: float = 0.05   # absolute CER margin over the narrator's own median
    max_deleted_run: int = 2              # words; a 3-word hole in one sentence is a skip
    max_inserted_run: int = 3
    tail_words: int = 2                   # the last words must be present (cut-off check)


@dataclass
class CheckResult:
    ok: bool
    cer: float
    wer: float
    reasons: List[str] = field(default_factory=list)
    suspect_word_indices: List[int] = field(default_factory=list)  # indices into the reference words


def check_sentence(reference: str, hypothesis: str, narrator_baseline_cer: float,
                   cfg: CheckConfig = CheckConfig()) -> CheckResult:
    ref, hyp = normalize_for_asr(reference), normalize_for_asr(hypothesis)
    cer = jiwer.cer(ref, hyp) if ref else 0.0
    out = jiwer.process_words(ref, hyp)
    reasons, suspects = [], []
    del_run = ins_run = 0
    max_del = max_ins = 0
    for chunk in out.alignments[0]:
        n_ref = chunk.ref_end_idx - chunk.ref_start_idx
        n_hyp = chunk.hyp_end_idx - chunk.hyp_start_idx
        if chunk.type == "delete":
            del_run += n_ref; max_del = max(max_del, del_run)
            suspects.extend(range(chunk.ref_start_idx, chunk.ref_end_idx))
        else:
            del_run = 0
        if chunk.type in ("insert", "substitute"):
            ins_run += n_hyp; max_ins = max(max_ins, ins_run)
            if chunk.type == "substitute":
                suspects.extend(range(chunk.ref_start_idx, chunk.ref_end_idx))
        else:
            ins_run = 0
    if cer > narrator_baseline_cer + cfg.max_cer_over_baseline:
        reasons.append(f"CER {cer:.3f} > baseline {narrator_baseline_cer:.3f} + {cfg.max_cer_over_baseline}")
    if max_del > cfg.max_deleted_run:
        reasons.append(f"skip: {max_del} contiguous words missing")
    if max_ins > cfg.max_inserted_run:
        reasons.append(f"hallucination: {max_ins} contiguous inserted/substituted words")
    ref_words = ref.split()
    hyp_words = hyp.split()
    if ref_words and ref_words[-cfg.tail_words:] != hyp_words[-cfg.tail_words:]:
        tail_ok = all(w in hyp_words[-(cfg.tail_words + 2):] for w in ref_words[-cfg.tail_words:])
        if not tail_ok:
            reasons.append("ending missing or cut off")
            suspects.extend(range(max(0, len(ref_words) - cfg.tail_words), len(ref_words)))
    rep = _repeated_ngram(hyp_words)
    if rep and rep not in " ".join(ref_words):
        reasons.append(f"repeated phrase not in text: '{rep}'")
    return CheckResult(not reasons, cer, out.wer, reasons, sorted(set(suspects)))


def _repeated_ngram(words: List[str], n: int = 3) -> Optional[str]:
    for i in range(len(words) - 2 * n + 1):
        if words[i:i + n] == words[i + n:i + 2 * n]:
            return " ".join(words[i:i + n])
    return None


def retry_actions(result: CheckResult) -> List[str]:
    """
    Durations are deterministic, so re-seeding the decoder cannot fix a duration-caused skip.
    Retries must change something that matters, in this order.
    """
    actions = []
    if any(r.startswith("skip") or r.startswith("ending") for r in result.reasons):
        actions.append("raise local length_scale (x1.15) on the suspect words")
    if result.suspect_word_indices:
        actions.append("check G2P of suspect words; add a lexicon entry if wrong")
    actions.append("re-split the sentence at a comma / render clauses separately")
    actions.append("flag for human listening if still failing")
    return actions
```

---

# Step 13: Production API

```text
Claude Code Prompt 13.1: Create src/inference/api.py (FastAPI):
 - POST /synthesize {text, lang, voice} -> audio + per-sentence QA result
 - POST /books (long jobs, queued) -> job id; GET /books/{id} -> status, audio, QA report
 - streaming by sentence for previews; rate limiting; /health
 - never return a sentence that failed verification without marking it in the QA report
```

---

# Step 14: Evaluation and QA

The binding constraint is measurement, not modelling. No MOS predictor is validated for Czech or Slovak. The best open ASR still errs on 5.5–11% of words of human speech (FLEURS WER: Parakeet-TDT-0.6B-v3 11.01% cs and 8.82% sk; Canary-1B-v2 7.86% cs; SloPal whisper-large-v3-sk 5.5% sk). No arena covers either language. So:

1. **Relative ASR thresholds.** Run the same ASR on the narrator's real recordings of similar text and normalise both sides to one spoken form. Then detect failure *shapes*:
   - runs of deleted words (skips);
   - runs of inserted or substituted words (hallucinations);
   - repeated n-grams;
   - missing endings.

   For chapter-level regression runs, also report Argmax's counts (Sep 2026): skips of ≥ 10 contiguous deleted words and hallucinations of ≥ 20 contiguous inserted or substituted words. Add worst-of-N WER across seeds, the catastrophic-failure rate, and windowed speaker similarity (wSIM, 8 s windows at a 4 s stride).
2. **Pace metrics M1–M6.** Compute them with an aligner independent of the model (Parakeet word/character timestamps). The model's own durations are circular.
3. **Speaker similarity.** Measure within one language with one extractor, and never compare numbers across extractors: SupertonicTTS scores 0.472 with WavLM-TDNN and 0.915 with WavLM Base+. Pair it with rhythm metrics, because verification embeddings ignore rhythm.
4. **TTSDS2** (the metric that best tracks human ratings, Spearman 0.67) is a relative checkpoint signal only. Its multilingual mode covers 14 languages; Polish and Russian are the nearest, and Czech and Slovak are not included. Recalibrate it against native ratings before trusting it.
5. **Native listening.** A few hundred judgments to calibrate the automatic metrics. Include minimal pairs for Czech vowel length, ř and consonant clusters, and Slovak diphthongs and ľ.

| Metric | Measures | Proposed gate |
|---|---|---|
| M1 sentence articulation rate | Phones/s excluding pauses ≥ 100 ms | Flag any sentence > 10–12% from target |
| M2 end-of-text slowdown | Median rate of the last two sentences ÷ the preceding ones, plus the slope against position | 0.95–1.05 (catches the ElevenLabs symptom) |
| M3 position A/B | Same paragraph rendered as final vs followed by more text | ≤ 3–5% |
| M4 cross-text consistency | Coefficient of variation of chapter-median rates | ≤ the narrator's own CV on held-out recordings; book median within ±5% of target |
| M5 probe sentence | One fixed sentence embedded in many texts | Identical under sentence-level rendering |
| M6 distribution match | Wasserstein distance between synthetic and human sentence-rate distributions | Track |

The thresholds are proposals anchored to the roughly 5% just-noticeable difference for tempo; no industry standard exists.

```text
Claude Code Prompt 14.1: Create src/eval/report.py, scripts/evaluate.py and tests/test_report.py.
 - report.py builds the per-chapter QA report of Prompt 12.1 as JSON (the API in Prompt 13.1
   returns it): each sentence's check result from src/eval/asr_check.py with its retries, the
   narrator baseline CER it was judged against, and M1-M6 from src/eval/pace.py next to the
   proposed gates in the table above.
 - scripts/evaluate.py scores a checkpoint on held-out sets in two modes:
     fast: what the trainer runs at every eval interval (Prompt 11.1) on 50 held-out sentences:
           the sentence-level failure shapes of src/eval/asr_check.py and M1
     full: a chapter-level regression run: Argmax's counts (skips of >= 10 contiguous deleted
           words, hallucinations of >= 20 contiguous inserted or substituted words), worst-of-N
           WER across seeds, the catastrophic-failure rate, wSIM (8 s windows at a 4 s stride),
           M1-M6, and TTSDS2 as a relative same-text signal only
 - rates for M1-M6 come from an aligner independent of the model (Parakeet word/char
   timestamps), never from the model's own durations; speaker similarity uses one extractor
   (Prompt 8.1) and compares within one language
 - the ASR model, the aligner and the speaker extractor sit behind small interfaces, so
   tests/test_report.py runs on CPU with stubs: gates flag the right sentences, the Argmax
   counts match hand-built alignments, and wSIM uses the right windows
Native listening (item 5 above) stays manual.
```

### Reference: `src/eval/pace.py` (tested)

```python
# src/eval/pace.py
"""
Pace metrics M1-M6 (Step 14). Thresholds are PROPOSALS anchored to the ~5% just-noticeable
difference for tempo; no industry standard exists. Compute rates with an aligner that is
independent of the model (e.g. Parakeet-TDT-0.6B-v3 word/char timestamps), never from the
model's own durations (circular).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class Sentence:
    n_phones: int
    speech_s: float   # duration without pauses >= 100-150 ms
    position: int     # index within its paragraph (0-based)
    paragraph_len: int


def articulation_rate(s: Sentence) -> float:
    return s.n_phones / max(s.speech_s, 1e-6)


def m1_sentence_outliers(sentences: Sequence[Sentence], target_rate: float, tol: float = 0.12) -> List[int]:
    """M1: sentences more than tol (10-12%) away from the target articulation rate."""
    return [i for i, s in enumerate(sentences) if abs(articulation_rate(s) / target_rate - 1) > tol]


def m2_end_slowdown(sentences: Sequence[Sentence]) -> Dict[str, float]:
    """
    M2: median rate of the last two sentences of each text / median of the preceding ones,
    plus the least-squares slope of rate against relative position. Pass: 0.95 <= ratio <= 1.05.
    This is the metric for the ElevenLabs "slows down over the last sentences" symptom.
    """
    rates = np.array([articulation_rate(s) for s in sentences])
    if len(rates) < 4:
        return {"ratio": float("nan"), "slope": float("nan")}
    ratio = float(np.median(rates[-2:]) / np.median(rates[:-2]))
    pos = np.linspace(0, 1, len(rates))
    slope = float(np.polyfit(pos, rates / np.median(rates), 1)[0])
    return {"ratio": ratio, "slope": slope}


def m3_position_ab(rate_as_final: float, rate_with_continuation: float) -> float:
    """M3: same paragraph rendered as the final one vs followed by more text. Pass: <= 0.03-0.05."""
    return abs(rate_as_final / rate_with_continuation - 1)


def m4_cross_text_cv(chapter_medians: Sequence[float]) -> float:
    """M4: coefficient of variation of chapter-median rates. Pass: <= the narrator's own CV."""
    x = np.asarray(chapter_medians, dtype=float)
    return float(x.std() / x.mean())


def m5_probe_spread(probe_rates: Sequence[float]) -> float:
    """M5: one fixed probe sentence embedded in many texts; max relative spread of its rate."""
    x = np.asarray(probe_rates, dtype=float)
    return float((x.max() - x.min()) / np.median(x))


def m6_distribution_distance(synthetic_rates: Sequence[float], human_rates: Sequence[float]) -> float:
    """M6: 1-D Wasserstein distance between synthetic and human sentence-rate distributions."""
    a, b = np.sort(np.asarray(synthetic_rates, float)), np.sort(np.asarray(human_rates, float))
    q = np.linspace(0, 1, 200)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))
```

### Test 9, 12, 14: Discriminator, tempo, pace and ASR-check tests (tested)

```python
# tests/test_eval_and_discriminator.py
"""Tests for the multilingual SSL discriminator wrapper, tempo labels, pace metrics and ASR checks."""
import pytest
import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model

from src.data.tempo import flag_rate_outliers, tempo_from_alignment
from src.eval.asr_check import check_sentence, retry_actions
from src.eval.pace import Sentence, m1_sentence_outliers, m2_end_slowdown, m4_cross_text_cv, m6_distribution_distance
from src.model.discriminators import SSLDiscriminator
from src.model.duration import durations_from_logw


@pytest.fixture(scope="module")
def tiny_ssl():
    # random-initialised stand-in with the XLS-R architecture (no download in tests)
    cfg = Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                         conv_dim=(32, 32), conv_kernel=(10, 3), conv_stride=(5, 2), num_feat_extract_layers=2,
                         feat_extract_norm="layer", do_stable_layer_norm=True)
    return Wav2Vec2Model(cfg)


class TestSSLDiscriminator:
    def test_encoder_frozen_and_generator_gets_gradients(self, tiny_ssl):
        d = SSLDiscriminator(tiny_ssl, n_layers=3, hidden=32, head_dim=16)
        real = torch.randn(2, 24000) * 0.1
        fake = (torch.randn(2, 24000) * 0.1).requires_grad_(True)
        adv, fm = d.generator_loss(real, fake)
        (adv + fm).backward()
        assert fake.grad is not None and fake.grad.abs().sum() > 0      # 24k -> 16k resample is differentiable
        assert all(p.grad is None for p in d.ssl.parameters())           # frozen backbone

    def test_discriminator_step_updates_head_only(self, tiny_ssl):
        d = SSLDiscriminator(tiny_ssl, n_layers=3, hidden=32, head_dim=16).train()
        assert not d.ssl.training                                        # stays in eval mode
        loss = d.discriminator_loss(torch.randn(2, 12000), torch.randn(2, 12000))
        loss.backward()
        assert any(p.grad is not None for p in d.head.parameters())


class TestTempo:
    def test_articulation_rate_excludes_pauses(self):
        kinds = ["phone"] * 10 + ["punct"] + ["phone"] * 10
        frames = [5] * 10 + [47] + [5] * 10   # 47 frames * 10.67 ms = 0.5 s pause
        lab = tempo_from_alignment(kinds, frames, hop_s=256 / 24000)
        speech = 100 * 256 / 24000
        assert abs(lab.articulation_rate - 20 / speech) < 1e-6
        assert 0.3 < lab.pause_ratio < 0.4

    def test_outlier_flags(self):
        rates = {f"u{i}": 12.0 for i in range(20)} | {"slow": 9.5}
        assert flag_rate_outliers(rates) == ["slow"]


class TestPace:
    def test_m2_catches_end_of_text_slowdown(self):
        s = [Sentence(60, 5.0, i, 8) for i in range(6)] + [Sentence(60, 5.9, 6, 8), Sentence(60, 5.9, 7, 8)]
        m = m2_end_slowdown(s)
        assert m["ratio"] < 0.95 and m["slope"] < 0                       # ElevenLabs-style wind-down fails
        flat = [Sentence(60, 5.0, i, 8) for i in range(8)]
        assert abs(m2_end_slowdown(flat)["ratio"] - 1.0) < 1e-9

    def test_m1_m4_m6(self):
        s = [Sentence(60, 5.0, 0, 3), Sentence(60, 6.0, 1, 3), Sentence(60, 5.1, 2, 3)]
        assert m1_sentence_outliers(s, target_rate=12.0) == [1]
        assert m4_cross_text_cv([12.0, 12.0, 12.0]) == 0.0
        assert m6_distribution_distance([12, 12.5, 13], [12, 12.5, 13]) < 1e-9


class TestASRCheck:
    REF = "Včera večer sme s bratom išli do kina a potom domov pešo."

    def test_clean_render_passes(self):
        assert check_sentence(self.REF, "včera večer sme s bratom išli do kina a potom domov pešo", 0.03).ok

    def test_skip_is_caught_by_run_length(self):
        # three contiguous words missing: flagged as a skip even with a generous CER margin
        r = check_sentence(self.REF, "včera večer sme s bratom a potom domov pešo", 0.30)
        assert not r.ok and any(x.startswith("skip") for x in r.reasons)
        assert r.suspect_word_indices == [5, 6, 7]

    def test_cut_off_ending(self):
        r = check_sentence(self.REF, "včera večer sme s bratom išli do kina a potom", 0.03)
        assert not r.ok and any("ending" in x for x in r.reasons)

    def test_repetition_is_caught(self):
        r = check_sentence(self.REF, "včera večer sme s bratom išli do kina do kina a potom domov pešo", 0.03)
        assert not r.ok

    def test_retry_changes_something_that_matters(self):
        r = check_sentence(self.REF, "včera večer sme s bratom išli do kina a potom", 0.03)
        assert "raise local length_scale (x1.15) on the suspect words" in retry_actions(r)

    def test_token_scale_slows_only_suspect_tokens(self):
        logw = torch.log(torch.full((1, 1, 4), 3.0))
        scale = torch.tensor([[1.0, 1.0, 1.5, 1.0]])
        d = durations_from_logw(logw, torch.ones(1, 1, 4), 1.0, scale)
        assert d[0, 0].tolist() == [3, 3, 5, 3]
```

---

# Appendix: Test Suite

```bash
uv run pytest tests/ -q -m "not slow"   # 199 tests, ~15 s on CPU
uv run pytest tests/ -q                 # + end-to-end synthetic training test, ~40 s on CPU
```

| File | Tests | Guards |
|---|---|---|
| test_environment.py | 4 | torch and torchaudio versions; espeak-ng for cs, sk and en; the audio config equals the vocoder's; CUDA (skips without a GPU) |
| test_phonemizer.py | 52 | Defects 4 and 5; espeak-ng corrections; strict vocabulary; digits raise |
| test_audio.py | 10 | Defects 6 and 7; exact BigVGAN mel equality; filterbank pinned to librosa 0.10.2; EBU R128; soft onsets kept |
| test_data_pipeline.py | 5 | Defect 2; drop-don't-crop; strict symbols; conditioning in batches |
| test_model_components.py | 22 | Defects 1 and 3; MAS equals exhaustive search; decoder overfits one utterance; LayerNorm crash; exact padding invariance |
| test_eval_and_discriminator.py | 12 | SSL discriminator gradients and freezing; tempo labels; pace metrics; ASR failure shapes; per-token retry scaling |
| test_integration.py | 1 | The whole model learns alignment, durations and content on a synthetic language, and keeps every token |
| test_num2words_sk.py | 44 | Slovak numerals; expectations quoted from MSJ, PSP, Navrátil, Beliana or JÚĽŠ columns, or decided in native review; no ordinal forms in cardinals |
| test_num2words_cs.py | 50 | Czech numerals; expectations quoted from IJP, cs.wikipedia, ČRo or NK; every output word is a dictionary word |

### End-to-end test (tested)

```python
# tests/test_integration.py
"""
End-to-end: train a tiny MatchaTTS on a synthetic 'language' whose tokens have known spectra and
durations, then check that MAS finds the true segmentation, the duration predictor learns it,
synthesis keeps every token, and the generated mel has the right content.
This is the test v1 lacked: shape tests passed while the model could never have worked.
"""
import pytest
import torch

from src.model.matcha_tts import MatchaConfig, MatchaTTS


def synthetic_batch(n_utts=16, n_mels=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    V = 10
    patterns = torch.randn(V, n_mels, generator=g) * 1.5
    true_dur = {k: 2 + (k % 4) for k in range(1, V)}          # 2..5 frames per token
    seqs, mels, durs = [], [], []
    for _ in range(n_utts):
        L = int(torch.randint(4, 9, (1,), generator=g))
        s = [int(torch.randint(1, V, (1,), generator=g))]
        while len(s) < L:  # no identical neighbours: their boundary would be unobservable
            k = int(torch.randint(1, V, (1,), generator=g))
            if k != s[-1]:
                s.append(k)
        s = torch.tensor(s)
        d = torch.tensor([true_dur[int(k)] for k in s])
        m = torch.cat([patterns[k].unsqueeze(1).repeat(1, true_dur[int(k)]) for k in s], dim=1)
        seqs.append(s); durs.append(d); mels.append(m + 0.05 * torch.randn(m.shape, generator=g))
    B = n_utts
    tx = torch.tensor([len(s) for s in seqs]); ty = torch.tensor([m.shape[1] for m in mels])
    ids = torch.zeros(B, int(tx.max()), dtype=torch.long); y = torch.zeros(B, n_mels, int(ty.max()))
    dd = torch.zeros(B, int(tx.max()))
    for i in range(B):
        ids[i, : tx[i]] = seqs[i]; y[i, :, : ty[i]] = mels[i]; dd[i, : tx[i]] = durs[i]
    batch = {"phoneme_ids": ids, "phoneme_lengths": tx, "mels": y, "mel_lengths": ty,
             "lang_ids": torch.zeros_like(ids), "speaker_ids": torch.zeros(B, dtype=torch.long),
             "boundary_ids": torch.zeros(B, dtype=torch.long), "rates": torch.tensor([[12.0, 0.0]]).repeat(B, 1)}
    return batch, dd, patterns


@pytest.mark.slow
def test_tiny_matcha_learns_alignment_durations_and_content():
    torch.manual_seed(0)
    batch, true_d, patterns = synthetic_batch()
    cfg = MatchaConfig(n_vocab=10, n_mels=16, enc_dim=64, enc_layers=2, enc_heads=2, enc_ff=128,
                       spk_dim=8, cond_dim=8, dur_hidden=64, dec_dim=64, dec_levels=2, dropout=0.0)
    model = MatchaTTS(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for step in range(700):
        opt.zero_grad()
        out = model(batch)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()

    # 1) MAS recovers the true segmentation
    with torch.no_grad():
        attn = model(batch)["attn"]
    mas_d = attn.sum(-1)
    valid = batch["phoneme_ids"] > 0
    assert ((mas_d - true_d).abs()[valid] <= 1).float().mean() > 0.9

    # 2) + 3) synthesis: every token kept, durations learned, content right
    res = model.synthesize(batch["phoneme_ids"], batch["lang_ids"], batch["phoneme_lengths"],
                           batch["speaker_ids"], batch["rates"], batch["boundary_ids"],
                           n_steps=10, temperature=0.3)
    pred_d = res["durations"]
    assert (pred_d[valid] >= 1).all()
    assert (pred_d - true_d).abs()[valid].mean() < 1.0
    cos = []
    for b in range(batch["phoneme_ids"].shape[0]):
        start = 0
        for i in range(int(batch["phoneme_lengths"][b])):
            n = int(pred_d[b, i])
            seg = res["mel"][b, :, start:start + n].mean(dim=1)
            ref = patterns[batch["phoneme_ids"][b, i]]
            cos.append(torch.nn.functional.cosine_similarity(seg, ref, dim=0))
            start += n
    assert torch.stack(cos).mean() > 0.9
```

---

# Quick Start

```bash
# 1. environment (uv: https://docs.astral.sh/uv/)
uv sync
git clone https://github.com/NVIDIA/BigVGAN third_party/BigVGAN
git -C third_party/BigVGAN checkout 7d2b454
uv run pytest tests/ -q

# 2. data (Step 4.2) and G2P review (Step 3.2)
uv run scripts/prepare_corpus.py --audio raw/ --text book.txt --lang sk --out data/narrator.jsonl
uv run scripts/g2p_review.py --manifest data/narrator.jsonl --out review/sk_words.tsv
uv run scripts/build_vocab.py --manifests data/*.jsonl --lexicon lexicon/lexicon.json --out lexicon/vocab.json

# 3. training (Step 11)
uv run scripts/train.py --config configs/training/pretrain.yaml
uv run scripts/tempo_labels.py --checkpoint checkpoints/pretrain.pt --manifest data/narrator.jsonl
uv run scripts/train.py --config configs/training/finetune_narrator.yaml

# 4. a book with verification and a QA report (Step 12)
uv run scripts/synthesize_book.py --checkpoint checkpoints/narrator.pt --book book.txt --lang sk --out out/
```
