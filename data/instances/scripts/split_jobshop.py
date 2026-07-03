"""Split OR-Library jobshop1.txt / jobshop2.txt into per-instance JSSP files.

Source: OR-Library (compiled by Dirk C. Mattfeld and Rob J.M. Vaessens),
https://people.brunel.ac.uk/~mastjjb/jeb/orlib/jobshopinfo.html
The bundle contains classic instances (ft06/10/20, la01-40, abz5-9, orb, swv, yn, ta)
in a multi-instance text file delimited by `instance <name>` headers, each followed
by descriptive lines, then an `n_jobs n_machines` line and `n_jobs` operation rows
(pairs of machine_id processing_time, machines 0-indexed) — directly compatible
with this project's loader (`src/problem/job_shop_scheduling/problem.rs::load_file`).

Usage: python3 data/instances/scripts/split_jobshop.py [SRC_DIR]
  SRC_DIR holds the downloaded orlib_jobshop{1,2}.txt bundles (default: /tmp).
Outputs: data/instances/jssp/orlib/<name>.txt
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

SRC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp")
SRC_FILES = [SRC_DIR / "orlib_jobshop1.txt", SRC_DIR / "orlib_jobshop2.txt"]
DST_DIR = Path(__file__).resolve().parent.parent / "jssp" / "orlib"

INSTANCE_RE = re.compile(r"^\s*instance\s+(\S+)\s*$")
DIM_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")
INT_TOKEN_RE = re.compile(r"^-?\d+$")


def split_file(src: Path) -> int:
    if not src.exists():
        print(f"skip: {src} not found")
        return 0
    lines = src.read_text().splitlines()
    written = 0
    i = 0
    while i < len(lines):
        m = INSTANCE_RE.match(lines[i])
        if not m:
            i += 1
            continue
        name = m.group(1)
        j = i + 1
        # find first line matching "N M" with integers, all-integer following row
        while j < len(lines):
            dm = DIM_RE.match(lines[j])
            if dm:
                n_jobs = int(dm.group(1))
                n_mach = int(dm.group(2))
                # validate: next n_jobs lines should contain 2*n_mach integers each
                rows = lines[j + 1 : j + 1 + n_jobs]
                if len(rows) == n_jobs and all(
                    len(r.split()) == 2 * n_mach
                    and all(INT_TOKEN_RE.match(t) for t in r.split())
                    for r in rows
                ):
                    out_lines = [f"# {name} (OR-Library)", f"{n_jobs} {n_mach}"]
                    out_lines += [" ".join(r.split()) for r in rows]
                    (DST_DIR / f"{name}.txt").write_text("\n".join(out_lines) + "\n")
                    written += 1
                    i = j + 1 + n_jobs
                    break
            j += 1
        else:
            i = j
            continue
    return written


def main() -> int:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for s in SRC_FILES:
        w = split_file(s)
        total += w
        print(f"{s.name}: wrote {w} instance(s)")
    print(f"total: {total} files in {DST_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
