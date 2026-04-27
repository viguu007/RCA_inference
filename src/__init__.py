"""Top-level package for RCA inference.

Keep this module lightweight to avoid import-time side effects.
"""

from .metadata import Metadata

__all__ = ["Metadata"]