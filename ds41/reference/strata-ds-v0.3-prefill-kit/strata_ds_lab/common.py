from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from typing import Any
GiB = 1 << 30
MiB = 1 << 20
class ContractError(ValueError):
    pass

def require_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{name}: expected integer >= {minimum}")
    return value

def finite_number(value: Any, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name}: expected finite number")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ContractError(f"{name}: expected finite number >= {minimum}")
    return value

def _unique_pairs(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ContractError(f"duplicate JSON key: {k}")
        result[k] = v
    return result

def parse_json(text: str):
    def reject(value):
        raise ContractError(f"non-finite JSON literal: {value}")
    return json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=reject)

def read_json(path: str | Path, max_bytes: int = 64 << 20):
    with open(path, 'rb') as f:
        b = f.read(max_bytes + 1)
    if len(b) > max_bytes:
        raise ContractError('JSON exceeds configured size cap')
    return parse_json(b.decode('utf-8'))

def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def safe_path(root: Path, relative: str) -> Path:
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ContractError('path must be relative and stay inside checkpoint')
    root = root.resolve()
    dest = (root / p).resolve()
    if dest != root and root not in dest.parents:
        raise ContractError('symlink leaves checkpoint root')
    return dest
