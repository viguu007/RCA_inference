
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Metadata:
    process_id: Optional[int] = None
    gpu_index: Optional[int] = None

