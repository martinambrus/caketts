"""Forced alignment of a narrated part against its text (Step 4.2, design D3); see `interface.align`."""
from src.data.prepare.align.interface import align, check_distinct_models
from src.data.prepare.align.types import AlignConfig, AlignResult, Gap, UnitSpan, WordSpan

__all__ = ["align", "AlignConfig", "AlignResult", "UnitSpan", "WordSpan", "Gap", "check_distinct_models"]
