"""Split Beasley OR-Library bqp multi-instance files into per-instance QUBO files.

Source: OR-Library (J.E. Beasley), https://people.brunel.ac.uk/~mastjjb/jeb/orlib/bqpinfo.html
Format of each bqpN.txt: first line = number of instances, then for each instance
a header `n m` followed by `m` lines of `i j v` (1-indexed) — directly compatible
with this project's QUBO loader (`src/problem/qubo/problem.rs::Qubo::load_file`).

Usage: python3 data/instances/scripts/split_bqp.py [SRC_DIR]
  SRC_DIR holds the downloaded orlib_bqp{N}.txt bundles (default: /tmp).
Outputs: data/instances/qubo/bqp/bqp{N}_{k}.txt for k = 1..n_instances
"""
from __future__ import annotations

from pathlib import Path
import sys

SRC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp")
DST_DIR = Path(__file__).resolve().parent.parent / "qubo" / "bqp"
SIZES = [50, 100, 250, 500, 1000]


def tokenize(path: Path) -> list[str]:
    return path.read_text().split()


def split_one(size: int) -> int:
    src = SRC_DIR / f"orlib_bqp{size}.txt"
    if not src.exists():
        print(f"skip: {src} not found")
        return 0
    toks = tokenize(src)
    idx = 0
    n_inst = int(toks[idx]); idx += 1
    written = 0
    for k in range(1, n_inst + 1):
        n = int(toks[idx]); idx += 1
        m = int(toks[idx]); idx += 1
        lines = [f"{n} {m}"]
        for _ in range(m):
            i = toks[idx]; idx += 1
            j = toks[idx]; idx += 1
            v = toks[idx]; idx += 1
            lines.append(f"{i} {j} {v}")
        out = DST_DIR / f"bqp{size}_{k}.txt"
        out.write_text("\n".join(lines) + "\n")
        written += 1
    return written


def main() -> int:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for sz in SIZES:
        w = split_one(sz)
        total += w
        print(f"bqp{sz}: wrote {w} instance(s)")
    print(f"total: {total} files in {DST_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
