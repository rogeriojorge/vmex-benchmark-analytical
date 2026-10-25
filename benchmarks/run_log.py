"""Run-record helpers with no DESC dependency: timestamped stdout capture, solver-table parsing, JSON safety."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import re
import time

import numpy as np


class Tee(io.TextIOBase):
    """Write through to the real stdout and keep (wall time, line) pairs."""

    def __init__(self, stream):
        self.stream, self.lines, self.partial, self.t0 = stream, [], "", time.perf_counter()

    def write(self, text):
        self.stream.write(text)
        self.partial += text
        while "\n" in self.partial:
            line, self.partial = self.partial.split("\n", 1)
            self.lines.append((time.perf_counter()-self.t0, line))
        return len(text)

    def flush(self):
        self.stream.flush()


def parse_table(lines):
    """Iteration rows of DESC's lsq-exact and SciPy's TRF verbose tables."""
    rows = []
    for t, line in lines:
        tok = line.split()
        if len(tok) >= 4 and re.fullmatch(r"\d+", tok[0]) and re.fullmatch(r"\d+", tok[1]):
            try:
                vals = [float(x) for x in tok[2:]]
            except ValueError:
                continue
            rows.append({"iteration": int(tok[0]), "nfev": int(tok[1]), "cost": vals[0], "optimality": vals[-1],
                         "step_norm": vals[2] if len(vals) >= 4 else None, "t": round(t, 2)})
    return rows


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if hasattr(value, "shape") and hasattr(value, "dtype"):  # numpy and JAX arrays
        return json_safe(np.asarray(value).tolist())
    return value


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
