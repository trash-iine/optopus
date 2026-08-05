"""Render benchmark TOMLs in `docs/benchmarks/data/` into the viewer's data source.

Reads every curated `*.toml` under `docs/benchmarks/data/` (publish-ready outputs
of the Rust benchmark runner) and produces the web viewer's data source: a slim
copy of each TOML with the heavy per-run `runs` arrays stripped (`*.slim.toml`),
plus an `index.json` the browser fetches to discover them. `viewer.html` parses
those slim TOMLs client-side (see `docs/benchmarks/vendor/smol-toml.js`) and is the
single, interactive presentation of the results.

The problem type is the Rust runner's own output: each `[[results]]` entry carries
a `problem` field. `index.json` is a pure derivation of the slim TOMLs — it lists
their paths and mirrors each file's `problem` for the viewer's filters; it is never
hand-maintained. Older curated TOMLs that predate the `problem` field fall back to
`detect_problem` (parent-directory heuristic) so they keep working un-regenerated.

Requires `tomli_w` (`pip install tomli-w`). Run from anywhere:

    python3 docs/benchmarks/render.py
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


# Directory / instance-path names use the Rust `ProblemKind` serde spelling
# (see `src/benchmark/config.rs`) so the fallback below matches the runner's
# native `problem` field exactly.
PROBLEM_DIRS = {
    "maxcut": "MaxCut",
    "qubo": "Qubo",
    "sat": "Sat",
    "tsp": "Tsp",
    "jssp": "JobShop",
    "vertex_cover": "VertexCover",
}


def detect_problem(toml_path: Path, instance_path: str) -> str:
    """Fallback problem type for legacy TOMLs lacking the `problem` field.

    The runner now writes `problem` into each result, so this is only used for
    older curated files. Some problems share the same instance file (e.g.
    VertexCover uses MaxCut GSET graphs), so the result TOML alone is ambiguous;
    the parent directory under `docs/benchmarks/data/<problem>/` disambiguates.
    """
    rel_parts = toml_path.relative_to(DATA_DIR).parts
    if len(rel_parts) >= 2:
        return PROBLEM_DIRS.get(rel_parts[0], rel_parts[0])
    parts = Path(instance_path).parts
    if "max_cut" in parts:
        return "MaxCut"
    if "qubo" in parts:
        return "Qubo"
    if "sat" in parts:
        return "Sat"
    if "tsp" in parts:
        return "Tsp"
    if "jssp" in parts:
        return "JobShop"
    return "Other"


SLIM_SUFFIX = ".slim.toml"
INDEX_PATH = DATA_DIR / "index.json"


def raw_toml_paths() -> list[Path]:
    """Curated source TOMLs under DATA_DIR, excluding generated slim copies."""
    return sorted(
        p
        for p in DATA_DIR.rglob("*.toml")
        if not p.name.endswith(SLIM_SUFFIX)
    )


def slim_document(doc: dict, toml_path: Path) -> dict:
    """Strip the per-run `runs` arrays; keep only what the viewer needs.

    The bulk of each curated TOML is `results[].runs[].solution` (full solution
    vectors). The web viewer only reads `problem`, `instance_path`, `heuristic`,
    and `summary`, so the slim copy drops `runs` entirely. `problem` comes from
    the runner's own field, falling back to `detect_problem` for legacy TOMLs.
    """
    slim: dict = {}
    for key in ("timestamp", "config_file"):
        if key in doc:
            slim[key] = doc[key]
    slim["results"] = [
        {
            "problem": r.get("problem")
            or detect_problem(toml_path, r["instance_path"]),
            "instance_path": r["instance_path"],
            "heuristic": r["heuristic"],
            "summary": r["summary"],
        }
        for r in doc.get("results", [])
    ]
    return slim


def build_site_data() -> None:
    """Write slim TOMLs (viewer data source) + an index derived from them.

    GitHub Pages exposes no directory listing, so the viewer discovers the data
    files through `index.json`. The index is a pure derivation of the slim TOMLs:
    each entry lists a file's path and mirrors its `problem` (read from the slim
    document, itself sourced from the runner's `problem` field). It is never
    hand-maintained.
    """
    import tomli_w

    index: list[dict] = []
    for toml_path in raw_toml_paths():
        with toml_path.open("rb") as f:
            doc = tomllib.load(f)
        slim = slim_document(doc, toml_path)
        if not slim["results"]:
            continue

        slim_path = toml_path.with_name(toml_path.stem + SLIM_SUFFIX)
        with slim_path.open("wb") as f:
            tomli_w.dump(slim, f)

        rel = slim_path.relative_to(DATA_DIR).as_posix()
        index.append({"path": rel, "problem": slim["results"][0]["problem"]})

    index.sort(key=lambda e: e["path"])
    INDEX_PATH.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n"
    )
    total_kb = sum(
        (DATA_DIR / e["path"]).stat().st_size for e in index
    ) / 1024
    print(
        f"wrote {INDEX_PATH.relative_to(HERE.parent.parent)} "
        f"({len(index)} slim files, {total_kb:.1f} KB total)"
    )


def main() -> None:
    if not DATA_DIR.is_dir():
        sys.exit(f"data directory not found: {DATA_DIR}")
    build_site_data()


if __name__ == "__main__":
    main()
