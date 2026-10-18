"""Protein Complex Jigsaw (PCJ).

The :mod:`pcjigsaw` namespace is the preferred public API from v4 onward.
The historical :mod:`scijigsaw` package remains installed as a compatibility
layer so existing analyses and published scripts continue to run.
"""
from scijigsaw import *  # noqa: F401,F403
from scijigsaw import __all__ as _legacy_all

from .variant import (
    Perturbation,
    VariantImpactReport,
    analyse_perturbation,
)
from .misassembly import (
    OffPathwayRule,
    MisassemblyAssessment,
    assess_misassembly,
)

__version__ = "4.0.0"
__all__ = list(_legacy_all) + [
    "Perturbation",
    "VariantImpactReport",
    "analyse_perturbation",
    "OffPathwayRule",
    "MisassemblyAssessment",
    "assess_misassembly",
]
