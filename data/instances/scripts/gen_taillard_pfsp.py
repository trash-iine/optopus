"""Generate Taillard's 120 permutation flow shop instances ta001-ta120.

Source: OR-Library flowshop2.txt (contributed by Dirk C. Mattfeld and Rob J.M.
Vaessens), https://people.brunel.ac.uk/~mastjjb/jeb/orlib/flowshopinfo.html
It carries the random number generator of Taillard (1993), "Benchmarks for
basic scheduling problems", EJOR 64, 278-285, and the time seed of every
instance. The instances are regenerated from those seeds here rather than
copied, with processing times uniform on 1..99 drawn machine by machine, as
generate_flow_shop in the paper does.

The output layout is the one FlowShop::load_file reads, a header line
`n_jobs n_machines` and then one line of `n_jobs` processing times per
machine. Taillard's upper and lower bounds are left out on purpose.

Usage: python3 data/instances/scripts/gen_taillard_pfsp.py SRC_FILE
  SRC_FILE is the downloaded flowshop2.txt.
Outputs: data/instances/flow_shop/taillard/taNNN.txt
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

DST_DIR = Path(__file__).resolve().parent.parent / "flow_shop" / "taillard"

SIZE_RE = re.compile(r"^\s*(\d+)\s+jobs\s+(\d+)\s+machines\s*$", re.IGNORECASE)
SEED_RE = re.compile(r"^\s*(\d+)\s+(ta\d{3})\s*$")


def unif(seed: int, low: int, high: int) -> tuple[int, int]:
    """Taillard's unif, returning the drawn value and the advanced seed."""
    m, a, b, c = 2147483647, 16807, 127773, 2836
    k = seed // b
    seed = a * (seed % b) - k * c
    if seed < 0:
        seed += m
    value_0_1 = seed / m
    return low + int(value_0_1 * (high - low + 1)), seed


def generate(seed: int, n_jobs: int, n_machines: int) -> list[list[int]]:
    rows = []
    for _ in range(n_machines):
        row = []
        for _ in range(n_jobs):
            value, seed = unif(seed, 1, 99)
            row.append(value)
        rows.append(row)
    return rows


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    text = Path(sys.argv[1]).read_text()
    DST_DIR.mkdir(parents=True, exist_ok=True)
    size = None
    written = 0
    for line in text.splitlines():
        if m := SIZE_RE.match(line):
            size = (int(m.group(1)), int(m.group(2)))
        elif (m := SEED_RE.match(line)) and size is not None:
            seed, name = int(m.group(1)), m.group(2)
            n_jobs, n_machines = size
            rows = generate(seed, n_jobs, n_machines)
            body = "\n".join(" ".join(str(p) for p in row) for row in rows)
            (DST_DIR / f"{name}.txt").write_text(f"{n_jobs} {n_machines}\n{body}\n")
            written += 1
    print(f"wrote {written} instances to {DST_DIR}")
    if written != 120:
        sys.exit(f"expected 120 instances, found {written}")


if __name__ == "__main__":
    main()
