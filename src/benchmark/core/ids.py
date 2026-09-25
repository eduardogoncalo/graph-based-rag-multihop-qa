from __future__ import annotations

import hashlib
import json
from typing import Any


def deterministic_id(prefix: str, parts: list[Any] | tuple[Any, ...]) -> str:
    """Create a stable ID from JSON-serializable parts."""
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"
