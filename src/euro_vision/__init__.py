"""euro-vision — automated Euro coin scanning and rare coin detection."""

from .config import Config, load_config
from .pipeline import Pipeline
from .types import Coin, Detection, ScanResult

__version__ = "0.1.0"

__all__ = [
    "Config",
    "load_config",
    "Pipeline",
    "Coin",
    "Detection",
    "ScanResult",
    "__version__",
]
