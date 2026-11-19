"""Data types of the alignment stage (design D3): the result of `align` and its config."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.data.prepare.asr import AsrConfig

STATUSES = ("ok", "low_conf", "unmatched")
GAP_KINDS = ("extra_speech", "skipped_text")


@dataclass(frozen=True)
class WordSpan:
    text: str
    start: float
    end: float
    score: float


@dataclass
class UnitSpan:
    """An unmatched unit has no span: start_s and end_s are None and words is empty."""
    id: str
    start_s: Optional[float]
    end_s: Optional[float]
    words: List[WordSpan]
    score_mean: float
    score_min: float
    status: str
    flags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class Gap:
    start: float
    end: float
    kind: str


@dataclass
class AlignResult:
    units: List[UnitSpan]
    gaps: List[Gap]
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AlignConfig:
    backend: str = "mms-300m"
    language: str = "cs"
    device: str = "auto"
    longform: str = "auto"
    block_max_s: float = 600.0
    margin_s: float = 3.0
    window_s: float = 30.0
    context_s: float = 2.0
    gap_min_words: int = 3
    gap_min_s: float = 1.0
    ok_sigmas: float = 3.0
    edge_sigmas: float = 4.0
    unmatched_ratio: float = 0.5
    sigma_floor: float = 0.02
    asr: AsrConfig = field(default_factory=AsrConfig)
