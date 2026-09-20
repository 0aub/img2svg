"""img2svg - redraw flat and cel-shaded raster art as clean, verified SVG."""

__version__ = "0.1.0"

from .config import Config
from .pipeline import convert

__all__ = ["Config", "convert", "__version__"]
