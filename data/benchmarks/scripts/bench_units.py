"""Run the benchmark sweep as resumable (instance × heuristic) units.

A band config under `data/benchmarks/<problem>/` takes days at the large band,
and a run the machine kills partway loses everything a monolithic run has done.
This splits every config of a band into one unit per (instance, heuristic),
runs the units one at a time, keeps each finished result, and skips it on the
next start. `merge` puts the units of a config back together as the result
file the monolithic run would have written.

Within a unit every run of the one (instance, heuristic) pair still runs in
parallel. A unit holds its heuristic at index 0, so its per-run seeds differ
from those of a monolithic run of the same config. A time-limited run does
not repeat bit for bit anyway.

    python3 data/benchmarks/scripts/bench_units.py freeze
    python3 data/benchmarks/scripts/bench_units.py split small medium large bls
    caffeinate -dis python3 data/benchmarks/scripts/bench_units.py run small medium large bls
    python3 data/benchmarks/scripts/bench_units.py progress
    python3 data/benchmarks/scripts/bench_units.py merge small
    python3 docs/benchmarks/render.py

`run` takes `--problem <dir>` (e.g. `--problem qap`) to run one problem
first, and `--threads <n>` to run the runs of a unit on n threads instead of
one per core. A time limit counts from the start of each run, so fewer threads
than runs keep every budget and lengthen the unit. The first line of each
unit's `log.txt` records the thread count it ran with. `freeze` copies `target/release/optopus` into the work directory, and
`run` uses that copy from then on, so a rebuild during a sweep that takes
weeks does not change the binary halfway, which can move timings by itself. The units live under `$OPTOPUS_BENCH_WORK` (default
`~/optopus_bench_work`), outside the repository and outside `/tmp`, which a
restart clears. Requires `tomli_w` (`pip install tomli-w`).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import tomli_w

REPO = Path(__file__).resolve().parents[3]
CONFIGS = REPO / "data" / "benchmarks"
CURATED = REPO / "docs" / "benchmarks" / "data"
BUILT = REPO / "target" / "release" / "optopus"
WORK = Path(os.environ.get("OPTOPUS_BENCH_WORK", Path.home() / "optopus_bench_work"))
UNITS = WORK / "units"
FROZEN = WORK / "optopus"

BANDS = ("small", "medium", "large", "bls")
# The order problems run in within a band, cheapest first, so whole problems
# finish early and can be merged while the rest runs.
PROBLEM_ORDER = ("tsp", "qap", "flow_shop", "vrp", "graph_coloring", "sat",
                 "qubo", "jssp", "maxcut", "vertex_cover", "bin_packing")
# The instance source in each curated file name, `<track>_<source>_<band>.toml`.
SOURCE = {
    "maxcut": "gset", "vertex_cover": "gset", "graph_coloring": "gset",
    "qubo": "bqp", "sat": "uf", "tsp": "tsplib", "jssp": "orlib",
    "vrp": "cvrplib", "flow_shop": "taillard", "qap": "qaplib",
    "bin_packing": "falkenauer",
}


def band_configs(band: str) -> list[Path]:
    if band == "bls":
        return sorted((CONFIGS / "maxcut" / "gset_bls").glob("n*.toml"))
    found = [p for p in CONFIGS.glob(f"*/*_{band}.toml")
             if p.stem.split("_")[0] in ("general", "specific", "lkh", "hgs")]
    rank = {d: i for i, d in enumerate(PROBLEM_ORDER)}
    return sorted(found, key=lambda p: (rank.get(p.parent.name, len(rank)), p.name))


def problem_dir(config: Path) -> str:
    return "maxcut" if config.parent.name == "gset_bls" else config.parent.name


def curated_name(config: Path, band: str) -> str:
    if band == "bls":
        return f"bls_gset_{config.stem.removeprefix('n')}.toml"
    track = config.stem.removesuffix(f"_{band}")
    return f"{track}_{SOURCE[problem_dir(config)]}_{band}.toml"


def expand(config: dict) -> list[tuple[str, str]]:
    """Every (repo-relative instance path, problem) the config names, in order."""
    out = []
    for inst in config["instances"]:
        for path in sorted(glob.glob(str(REPO / inst["path"]))):
            out.append((str(Path(path).relative_to(REPO)), inst["problem"]))
    return out


def unit_key(band: str, config: Path, instance: str, idx: int) -> str:
    return f"{band}__{problem_dir(config)}__{config.stem}__{Path(instance).stem}__h{idx:02d}"


def units_of(band: str, config: Path) -> list[dict]:
    doc = tomllib.loads(config.read_text())
    return [
        {"key": unit_key(band, config, inst, i), "band": band,
         "config": str(config.relative_to(REPO)), "instance": inst, "problem": problem,
         "heuristic_idx": i, "budget": h.get("stop_condition", {}).get("max_duration_secs")}
        for inst, problem in expand(doc)
        for i, h in enumerate(doc["heuristics"])
    ]


def split(bands: list[str]) -> None:
    for band in bands:
        n = 0
        for config in band_configs(band):
            doc = tomllib.loads(config.read_text())
            for unit in units_of(band, config):
                # A finished unit keeps the unit.toml it ran with.
                if is_done(unit):
                    continue
                d = UNITS / unit["key"]
                d.mkdir(parents=True, exist_ok=True)
                one = {k: v for k, v in doc.items() if k not in ("instances", "heuristics")}
                one["instances"] = [{"path": str(REPO / unit["instance"]), "problem": unit["problem"]}]
                one["heuristics"] = [doc["heuristics"][unit["heuristic_idx"]]]
                (d / "unit.toml").write_text(tomli_w.dumps(one))
                (d / "meta.json").write_text(json.dumps(unit, indent=1))
                n += 1
        print(f"{band}: {n} units under {UNITS}")


def load_units(bands: list[str], problem: str | None = None) -> list[dict]:
    units = []
    for band in bands:
        for config in band_configs(band):
            if problem and problem_dir(config) != problem:
                continue
            units.extend(units_of(band, config))
    return units


def is_done(unit: dict) -> bool:
    return (UNITS / unit["key"] / "res.toml").exists()


def freeze() -> None:
    if not BUILT.exists():
        sys.exit(f"{BUILT} not found, run `cargo build --release` first")
    WORK.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUILT, FROZEN)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    (WORK / "optopus.txt").write_text(f"{head} {time.strftime('%Y-%m-%d %H:%M')}\n")
    print(f"froze {BUILT} as {FROZEN} (HEAD {head})")


def run(bands: list[str], problem: str | None, threads: int | None) -> None:
    binary = FROZEN if FROZEN.exists() else BUILT
    if not binary.exists():
        sys.exit(f"{binary} not found, run `cargo build --release` first")
    env = dict(os.environ)
    if threads:
        env["RAYON_NUM_THREADS"] = str(threads)
    print(f"binary {binary}, threads {threads or 'one per core'}", flush=True)
    units = load_units(bands, problem)
    todo = [u for u in units if not is_done(u)]
    print(f"{len(units) - len(todo)}/{len(units)} done, {len(todo)} to run", flush=True)
    for i, unit in enumerate(todo, 1):
        d = UNITS / unit["key"]
        if not (d / "unit.toml").exists():
            sys.exit(f"{unit['key']} has no unit.toml, run `split` for its band first")
        for stale in (d / "result").glob("*.toml") if (d / "result").exists() else []:
            stale.unlink()
        start = time.time()
        with open(d / "log.txt", "w") as log:
            log.write(f"threads {threads or os.cpu_count()}\n")
            log.flush()
            rc = subprocess.run([str(binary), "unit.toml"], cwd=d, env=env,
                                stdout=log, stderr=log).returncode
        results = sorted((d / "result").glob("*.toml")) if (d / "result").exists() else []
        if rc != 0 or not results:
            print(f"[{i}/{len(todo)}] FAIL {unit['key']} rc={rc}, see {d / 'log.txt'}", flush=True)
            continue
        results[-1].rename(d / "res.toml")
        print(f"[{i}/{len(todo)}] done {unit['key']} {time.time() - start:.0f}s "
              f"{time.strftime('%m-%d %H:%M')}", flush=True)


def progress() -> None:
    for band in BANDS:
        units = load_units([band])
        if not units:
            continue
        done = sum(map(is_done, units))
        left = sum(u["budget"] or 0 for u in units if not is_done(u))
        print(f"{band:6s} {done:5d}/{len(units):5d} done, about {left / 3600:6.1f} h left")
        for config in band_configs(band):
            us = units_of(band, config)
            n = sum(map(is_done, us))
            if 0 < n < len(us):
                print(f"         {config.relative_to(CONFIGS)}: {n}/{len(us)}")


def merge(bands: list[str], partial: bool) -> None:
    for band in bands:
        for config in band_configs(band):
            us = units_of(band, config)
            done = [u for u in us if is_done(u)]
            if not done or (len(done) < len(us) and not partial):
                if done:
                    print(f"skip {config.relative_to(CONFIGS)}: {len(done)}/{len(us)} units")
                continue
            results, stamps = [], []
            for unit in done:
                doc = tomllib.loads((UNITS / unit["key"] / "res.toml").read_text())
                stamps.append(doc["timestamp"])
                for r in doc["results"]:
                    r["instance_path"] = unit["instance"]
                    results.append(r)
            out = CURATED / problem_dir(config) / curated_name(config, band)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(tomli_w.dumps({
                "timestamp": max(stamps),
                "config_file": str(config.relative_to(REPO)),
                "results": results,
            }))
            print(f"wrote {out.relative_to(REPO)} ({len(done)}/{len(us)} units)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("freeze", "split", "run", "progress", "merge"))
    ap.add_argument("bands", nargs="*", choices=BANDS + ("all",))
    ap.add_argument("--problem", help="run only this problem directory, e.g. qap")
    ap.add_argument("--threads", type=int, help="run the runs of a unit on this many threads")
    ap.add_argument("--partial", action="store_true", help="merge configs with units still missing")
    args = ap.parse_args()
    bands = list(BANDS) if not args.bands or "all" in args.bands else args.bands
    if args.command == "freeze":
        freeze()
    elif args.command == "split":
        split(bands)
    elif args.command == "run":
        run(bands, args.problem, args.threads)
    elif args.command == "progress":
        progress()
    else:
        merge(bands, args.partial)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
