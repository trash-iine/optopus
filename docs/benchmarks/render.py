"""Render benchmark TOMLs in `docs/benchmarks/data/` into docs artifacts.

Reads every curated `*.toml` under `docs/benchmarks/data/` (publish-ready outputs
of the Rust benchmark runner) and produces two things:

1. Per-heuristic Markdown (grouped by `heuristic.kind`), one file per kind
   alongside this script.
2. The web viewer's data source: a slim copy of each TOML with the heavy
   per-run `runs` arrays stripped (`*.slim.toml`), plus a `manifest.json` the
   browser fetches to discover them. `viewer.html` parses those slim TOMLs
   client-side (see `docs/benchmarks/vendor/smol-toml.js`).

Requires `tomli_w` (`pip install tomli-w`). Run from anywhere:

    python3 docs/benchmarks/render.py
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


@dataclass
class Row:
    instance: str
    summary: dict
    source: str
    heuristic: dict


def snake_case(name: str) -> str:
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    return s.lower()


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


def instance_short(instance_path: str) -> str:
    return Path(instance_path).stem


def instance_sort_key(name: str) -> tuple[int, str]:
    m = re.match(r"^G(\d+)$", name)
    if m:
        return (int(m.group(1)), name)
    m = re.match(r"^bqp(\d+)_(\d+)$", name)
    if m:
        return (int(m.group(1)) * 1000 + int(m.group(2)), name)
    m = re.match(r"^uf(\d+)-(\d+)$", name)
    if m:
        return (int(m.group(1)) * 10000 + int(m.group(2)), name)
    m = re.match(r"^la(\d+)$", name)
    if m:
        return (int(m.group(1)), name)
    m = re.match(r"^abz(\d+)$", name)
    if m:
        return (100 + int(m.group(1)), name)
    m = re.match(r"^orb(\d+)$", name)
    if m:
        return (200 + int(m.group(1)), name)
    return (10**9, name)


def fmt_num(x, digits: int = 2) -> str:
    if x is None:
        return "-"
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    if isinstance(x, int):
        return str(x)
    return f"{x:.{digits}f}"


def fmt_int_objective(x) -> str:
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return fmt_num(x)


def fmt_hyperparams(h: dict) -> str:
    parts = []
    for key in ("tabu_tenure", "t", "l0", "p0", "q"):
        if key in h:
            v = h[key]
            if isinstance(v, list):
                v = "(" + ", ".join(str(x) for x in v) + ")"
            parts.append(f"{key}={v}")
    sc = h.get("stop_condition", {})
    for key in ("max_iteration", "max_duration_secs", "max_failed_update"):
        if key in sc:
            parts.append(f"{key}={sc[key]}")
    return ", ".join(parts)


def load_data() -> list[tuple[str, Path, dict, dict]]:
    """Return a flat list of (source_stem, toml_path, heuristic, result_block)."""
    if not DATA_DIR.is_dir():
        sys.exit(f"data directory not found: {DATA_DIR}")
    items: list[tuple[str, Path, dict, dict]] = []
    for toml_path in sorted(DATA_DIR.rglob("*.toml")):
        with toml_path.open("rb") as f:
            doc = tomllib.load(f)
        source = toml_path.stem
        for r in doc.get("results", []):
            items.append((source, toml_path, r["heuristic"], r))
    return items


def group_by_kind(
    items: list[tuple[str, Path, dict, dict]],
) -> dict[str, dict[str, list[Row]]]:
    """kind -> problem -> rows. Later sources overwrite same (kind, instance)."""
    by_kind_problem: dict[str, dict[str, dict[str, Row]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    seen_sources_per_kind: dict[str, set[str]] = defaultdict(set)
    for source, toml_path, heuristic, r in items:
        kind = heuristic["kind"]
        instance_path = r["instance_path"]
        problem = detect_problem(toml_path, instance_path)
        instance = instance_short(instance_path)
        bucket = by_kind_problem[kind][problem]
        neighbor = heuristic.get("neighbor")
        row_key = (instance, neighbor) if neighbor else (instance, None)
        if row_key in bucket:
            print(
                f"warning: duplicate ({kind}, {problem}, {row_key}); "
                f"using {source} over {bucket[row_key].source}",
                file=sys.stderr,
            )
        bucket[row_key] = Row(
            instance=instance,
            summary=r["summary"],
            source=source,
            heuristic=heuristic,
        )
        seen_sources_per_kind[kind].add(source)
    out: dict[str, dict[str, list[Row]]] = {}
    for kind, problems in by_kind_problem.items():
        out[kind] = {}
        for problem, rows_by_key in problems.items():
            out[kind][problem] = sorted(
                rows_by_key.values(),
                key=lambda row: (
                    instance_sort_key(row.instance),
                    row.heuristic.get("neighbor") or "",
                ),
            )
    return out


def render_kind(kind: str, problems: dict[str, list[Row]]) -> str:
    lines: list[str] = []
    lines.append(f"# {kind}")
    lines.append("")
    lines.append(
        "Auto-generated by `docs/benchmarks/render.py` from "
        "`docs/benchmarks/data/*.toml`. Do not edit by hand."
    )
    lines.append("")
    sources_seen: dict[str, dict] = {}
    for problem in sorted(problems):
        rows = problems[problem]
        lines.append(f"## {problem}")
        lines.append("")
        n_runs = {row.summary["num_successful_runs"] for row in rows}
        runs_label = (
            f"{next(iter(n_runs))} runs"
            if len(n_runs) == 1
            else "runs vary per row"
        )
        lines.append(f"{len(rows)} instance(s), {runs_label}.")
        lines.append("")
        has_neighbor = any(row.heuristic.get("neighbor") for row in rows)
        if has_neighbor:
            lines.append(
                "| Instance | Neighbor | Best | Avg | Worst | Std | "
                "Best time-to-best [s] | Avg time-to-best [s] | "
                "Avg total [s] | Runs | Source |"
            )
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        else:
            lines.append(
                "| Instance | Best | Avg | Worst | Std | "
                "Best time-to-best [s] | Avg time-to-best [s] | "
                "Avg total [s] | Runs | Source |"
            )
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for row in rows:
            s = row.summary
            cols = [row.instance]
            if has_neighbor:
                cols.append(row.heuristic.get("neighbor") or "-")
            cols += [
                fmt_int_objective(s["best_objective"]),
                fmt_num(s["avg_objective"]),
                fmt_int_objective(s["worst_objective"]),
                fmt_num(s["std_objective"]),
                fmt_num(s["best_time_to_best_secs"]),
                fmt_num(s["avg_time_to_best_secs"]),
                fmt_num(s["avg_total_time_secs"]),
                str(s["num_successful_runs"]),
                f"`{row.source}`",
            ]
            lines.append("| " + " | ".join(cols) + " |")
            sources_seen.setdefault(row.source, row.heuristic)
        lines.append("")
    lines.append("## Hyperparameters per source")
    lines.append("")
    for source in sorted(sources_seen):
        lines.append(f"- `{source}`: {fmt_hyperparams(sources_seen[source])}")
    lines.append("")
    return "\n".join(lines)


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
    items = load_data()
    if not items:
        sys.exit(f"no TOML data found under {DATA_DIR}")
    grouped = group_by_kind(items)
    for kind, problems in grouped.items():
        out_path = HERE / f"{snake_case(kind)}.md"
        out_path.write_text(render_kind(kind, problems))
        print(f"wrote {out_path.relative_to(HERE.parent.parent)}")
    build_site_data()


if __name__ == "__main__":
    main()
