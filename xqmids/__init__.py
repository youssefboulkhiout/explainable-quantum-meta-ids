"""explainable-quantum-meta-ids (package ``xqmids``).

Explainable Quantum-Enhanced Meta-Learning for Few-Shot and Zero-Day
Intrusion Detection in IoT Networks -- unified benchmark framework.

The code is the single-file script ``xai_meta_qml_levelAB.py`` split into
modules without changing any function, class or parameter, so a run gives
the same results as the original script.
"""
from . import _env  # noqa: F401  (headless Matplotlib, seaborn theme, warnings)
from .config import CONFIG, DATASETS, SEED, DEVICE, ALL_MODELS  # noqa: F401
from .pipeline import run_dataset  # noqa: F401
from .cli import main  # noqa: F401

__version__ = "1.0.0"
