"""VisualBind — multi-observer signal binding via weak supervision.

    >>> from visualbind import HintCollector, AgreementEngine, CrossCheck
    >>> collector = HintCollector({"face.expression": {"signals": ["happy"]}})
    >>> engine = AgreementEngine([CrossCheck("face.expression", "happy", "face.au", "AU12")])
"""

from visualbind.types import (
    HintVector,
    HintFrame,
    CrossCheck,
    AgreementResult,
    CheckResult,
    SourceSpec,
)
from visualbind.collector import HintCollector
from visualbind.agreement import AgreementEngine
from visualbind.pairing import PairMiner, ContrastivePair, PairMiningResult
from visualbind.encoder import TripletEncoder, TrainHistory

__all__ = [
    "HintCollector",
    "AgreementEngine",
    "PairMiner",
    "TripletEncoder",
    "HintVector",
    "HintFrame",
    "CrossCheck",
    "AgreementResult",
    "CheckResult",
    "SourceSpec",
    "ContrastivePair",
    "PairMiningResult",
    "TrainHistory",
]
