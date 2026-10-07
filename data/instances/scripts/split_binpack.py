"""Split OR-Library binpack1.txt ... binpack8.txt into per-instance files.

Source: OR-Library, instances of E. Falkenauer (1996), "A hybrid grouping
genetic algorithm for bin packing", Journal of Heuristics 2, 5-30,
https://people.brunel.ac.uk/~mastjjb/jeb/orlib/binpackinfo.html
Files 1-4 hold the uniform class (u120 ... u1000, capacity 150) and files 5-8
the triplets (t60 ... t501, capacity 100, optimum n/3 by construction).

Each bundle is `P`, then per instance an identifier line, a line
`capacity n best_known`, and `n` item sizes. The output layout is the one
BinPacking::load_file reads, `n capacity` and then the `n` sizes. The best
known bin count is left out on purpose.

The triplet files give sizes with one decimal. Those instances are scaled by
ten, capacity 1000, so every value is an integer and nothing is rounded.

Usage: python3 data/instances/scripts/split_binpack.py SRC_DIR
  SRC_DIR holds the downloaded binpack1.txt ... binpack8.txt.
Outputs: data/instances/bin_packing/falkenauer/<identifier>.txt
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import sys

DST_DIR = Path(__file__).resolve().parent.parent / "bin_packing" / "falkenauer"


def split_file(src: Path, scale: int) -> int:
    tokens = iter(src.read_text().split())
    written = 0
    for _ in range(int(next(tokens))):
        name = next(tokens)
        capacity, n, _best = Decimal(next(tokens)), int(next(tokens)), next(tokens)
        sizes = [Decimal(next(tokens)) for _ in range(n)]
        as_int = lambda v: int(v * scale)
        assert all(v * scale == as_int(v) for v in sizes + [capacity]), name
        body = "\n".join(str(as_int(v)) for v in sizes)
        (DST_DIR / f"{name}.txt").write_text(f"{n} {as_int(capacity)}\n{body}\n")
        written += 1
    return written


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src_dir = Path(sys.argv[1])
    DST_DIR.mkdir(parents=True, exist_ok=True)
    # Files 5-8 are the triplets, whose sizes have one decimal.
    written = sum(
        split_file(src_dir / f"binpack{k}.txt", 10 if k >= 5 else 1) for k in range(1, 9)
    )
    print(f"wrote {written} instances to {DST_DIR}")
    if written != 160:
        sys.exit(f"expected 160 instances, found {written}")


if __name__ == "__main__":
    main()
