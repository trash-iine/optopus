"""Render benchmark TOMLs in `docs/benchmarks/data/` into the viewer's data source.

Reads every curated `*.toml` under `docs/benchmarks/data/` (publish-ready outputs
of the Rust benchmark runner) and produces the web viewer's data source: a slim
copy of each TOML with the heavy per-run `runs` arrays stripped (`*.slim.toml`),
plus a `manifest.json` the browser fetches to discover them. `viewer.html` parses
those slim TOMLs client-side (see `docs/benchmarks/vendor/smol-toml.js`) and is the
single, interactive presentation of the results.

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


PROBLEM_DIRS = {
    "maxcut": "MaxCut",
    "qubo": "QUBO",
    "sat": "SAT",
    "tsp": "TSP",
    "jssp": "JobShopScheduling",
    "vertex_cover": "VertexCover",
}


def detect_problem(toml_path: Path, instance_path: str) -> str:
    """Resolve the problem type from the curated file's parent directory.

    Some problems share the same instance file (e.g. VertexCover uses MaxCut GSET
    graphs), so the result TOML alone is ambiguous; the parent directory under
    `docs/benchmarks/data/<problem>/` disambiguates.
    """
    rel_parts = toml_path.relative_to(DATA_DIR).parts
    if len(rel_parts) >= 2:
        return PROBLEM_DIRS.get(rel_parts[0], rel_parts[0])
    parts = Path(instance_path).parts
    if "max_cut" in parts:
        return "MaxCut"
    if "qubo" in parts:
        return "QUBO"
    if "sat" in parts:
        return "SAT"
    if "tsp" in parts:
        return "TSP"
    if "jssp" in parts:
        return "JobShopScheduling"
    return "Other"


SLIM_SUFFIX = ".slim.toml"
MANIFEST_PATH = DATA_DIR / "manifest.json"


def raw_toml_paths() -> list[Path]:
    """Curated source TOMLs under DATA_DIR, excluding generated slim copies."""
    return sorted(
        p
        for p in DATA_DIR.rglob("*.toml")
        if not p.name.endswith(SLIM_SUFFIX)
    )


def slim_document(doc: dict) -> dict:
    """Strip the per-run `runs` arrays; keep only what the viewer needs.

    The bulk of each curated TOML is `results[].runs[].solution` (full solution
    vectors). The web viewer only reads `instance_path`, `heuristic`, and
    `summary`, so the slim copy drops `runs` entirely.
    """
    slim: dict = {}
    for key in ("timestamp", "config_file"):
        if key in doc:
            slim[key] = doc[key]
    slim["results"] = [
        {
            "instance_path": r["instance_path"],
            "heuristic": r["heuristic"],
            "summary": r["summary"],
        }
        for r in doc.get("results", [])
    ]
    return slim


def build_site_data() -> None:
    """Write slim TOMLs (viewer data source) + a manifest listing them.

    GitHub Pages exposes no directory listing, so the viewer discovers the data
    files through `manifest.json`. Each entry carries the parent-dir-derived
    `problem` because some problems share instance files (VertexCover reuses
    MaxCut GSET graphs), making the TOML alone ambiguous.
    """
    import tomli_w

    manifest: list[dict] = []
    for toml_path in raw_toml_paths():
        with toml_path.open("rb") as f:
            doc = tomllib.load(f)
        results = doc.get("results", [])
        if not results:
            continue
        problem = detect_problem(toml_path, results[0]["instance_path"])

        slim_path = toml_path.with_name(toml_path.stem + SLIM_SUFFIX)
        with slim_path.open("wb") as f:
            tomli_w.dump(slim_document(doc), f)

        rel = slim_path.relative_to(DATA_DIR).as_posix()
        manifest.append({"path": rel, "problem": problem})

    manifest.sort(key=lambda e: e["path"])
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    total_kb = sum(
        (DATA_DIR / e["path"]).stat().st_size for e in manifest
    ) / 1024
    print(
        f"wrote {MANIFEST_PATH.relative_to(HERE.parent.parent)} "
        f"({len(manifest)} slim files, {total_kb:.1f} KB total)"
    )


def main() -> None:
    if not DATA_DIR.is_dir():
        sys.exit(f"data directory not found: {DATA_DIR}")
    build_site_data()


if __name__ == "__main__":
    main()
