"""Generate benchmark TOML configs under `data/benchmarks/<problem>/`.

Drives Phase 3 of the comprehensive benchmark plan (see
`docs/benchmarks/profiles.md`). One TOML per (problem, band, track) where:

    track ∈ {"general", "lkh"}
    band  ∈ {"small", "medium", "large"}

The `general` track always enumerates LocalSearch, TabuSearch,
LateAcceptanceHillClimbing, SimulatedAnnealing (× each applicable neighbor),
plus GeneticAlgorithm, Iterated, and Restart.

Usage: python3 data/scripts/gen_benchmark_configs.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "benchmarks"

NUM_RUNS = 10
DURATION = {"small": 30.0, "medium": 120.0, "large": 600.0}
SA_ITER_CAP = {"small": 10_000_000, "medium": 50_000_000, "large": 200_000_000}
LAHC_HISTORY = {"small": 100, "medium": 500, "large": 2000}
GA_POP = {"small": 50, "medium": 100, "large": 200}
RESTART_FAILED = {"small": 5000, "medium": 10_000, "large": 50_000}

NEIGHBORS = {
    "MaxCut": ["Flip", "Swap"],
    "Qubo": ["Flip", "Swap"],
    "Sat": ["Flip", "Swap"],
    "Tsp": ["TwoOpt", "Relocate"],
    "JobShop": ["Swap", "Relocate"],
    "VertexCover": ["Flip", "Swap"],
}

CROSSOVER = {
    "MaxCut": "Uniform",
    "Qubo": "Uniform",
    "Sat": "Uniform",
    "Tsp": "Order",
    "JobShop": "Ppx",
    "VertexCover": "Uniform",
}

TABU_TENURE = {
    "MaxCut": {"small": [3, 80], "medium": [3, 200], "large": [3, 500]},
    "Qubo": {"small": [3, 30], "medium": [3, 100], "large": [3, 200]},
    "Sat": {"small": [3, 20], "medium": [3, 50], "large": [3, 100]},
    "Tsp": {"small": [3, 30], "medium": [3, 100], "large": [3, 200]},
    "JobShop": {"small": [3, 30], "medium": [3, 80], "large": [3, 200]},
    "VertexCover": {"small": [3, 80], "medium": [3, 200], "large": [3, 500]},
}

SA_T0 = {"MaxCut": 100.0, "Qubo": 100.0, "Sat": 100.0, "Tsp": 1000.0,
         "JobShop": 100.0, "VertexCover": 100.0}
SA_COOLING = 0.9995

# Instance globs per (problem, band). Lists become multiple [[instances]] blocks.
INSTANCES: dict[tuple[str, str], list[str]] = {
    ("MaxCut", "small"):  [f"data/max_cut/G{n}" for n in (
        list(range(1, 22)) + list(range(43, 48)) + list(range(51, 55)))],
    ("MaxCut", "medium"): [f"data/max_cut/G{n}" for n in (
        list(range(22, 43)) + list(range(48, 51)))],
    ("MaxCut", "large"):  [f"data/max_cut/G{n}" for n in (
        list(range(55, 68)) + [70, 72, 77, 81])],
    ("Qubo", "small"):  ["data/qubo/bqp/bqp50_*.txt",  "data/qubo/bqp/bqp100_*.txt"],
    ("Qubo", "medium"): ["data/qubo/bqp/bqp250_*.txt", "data/qubo/bqp/bqp500_*.txt"],
    ("Qubo", "large"):  ["data/qubo/bqp/bqp1000_*.txt"],
    ("Sat", "small"):  ["data/sat/satlib/uf50-218/*.cnf",  "data/sat/satlib/uf75-325/*.cnf"],
    ("Sat", "medium"): ["data/sat/satlib/uf100-430/*.cnf", "data/sat/satlib/uf150-645/*.cnf"],
    ("Sat", "large"):  ["data/sat/satlib/uf200-860/*.cnf"],
    ("Tsp", "small"):  ["data/tsp/burma14.tsp", "data/tsp/eil51.tsp",
                       "data/tsp/att48.tsp", "data/tsp/berlin52.tsp"],
    ("Tsp", "medium"): ["data/tsp/eil101.tsp", "data/tsp/ch150.tsp"],
    ("Tsp", "large"):  ["data/tsp/dsj1000.tsp"],
    ("JobShop", "small"):  ["data/jssp/orlib/ft06.txt",  "data/jssp/orlib/ft10.txt",
                            "data/jssp/orlib/ft20.txt",
                            *[f"data/jssp/orlib/la0{n}.txt" for n in range(1, 6)],
                            "data/jssp/orlib/abz5.txt"],
    ("JobShop", "medium"): [*[f"data/jssp/orlib/la{n:02d}.txt" for n in range(6, 26)],
                            *[f"data/jssp/orlib/abz{n}.txt" for n in range(6, 10)],
                            *[f"data/jssp/orlib/orb{n:02d}.txt" for n in range(1, 11)]],
    ("JobShop", "large"):  [*[f"data/jssp/orlib/la{n}.txt" for n in range(26, 41)],
                            "data/jssp/orlib/swv*.txt",
                            "data/jssp/orlib/yn*.txt"],
    ("VertexCover", "small"):  [f"data/max_cut/G{n}" for n in
                                (list(range(1, 22)) + list(range(43, 48)) + list(range(51, 55)))],
    ("VertexCover", "medium"): [f"data/max_cut/G{n}" for n in
                                (list(range(22, 43)) + list(range(48, 51)))],
    ("VertexCover", "large"):  [f"data/max_cut/G{n}" for n in
                                (list(range(55, 68)) + [70, 72, 77, 81])],
}

PROBLEM_DIR = {
    "MaxCut": "maxcut", "Qubo": "qubo", "Sat": "sat", "Tsp": "tsp",
    "JobShop": "jssp", "VertexCover": "vertex_cover",
}


def fmt_array(xs: Iterable[int]) -> str:
    return "[" + ", ".join(str(x) for x in xs) + "]"


def instance_blocks(problem: str, band: str) -> str:
    out = []
    for glob in INSTANCES[(problem, band)]:
        out.append(f'[[instances]]\npath = "{glob}"\nproblem = "{problem}"\n')
    return "\n".join(out)


def stop_duration(band: str) -> str:
    return f"[heuristics.stop_condition]\nmax_duration_secs = {DURATION[band]}\n"


def stop_duration_with_iter_cap(band: str) -> str:
    return (f"[heuristics.stop_condition]\n"
            f"max_duration_secs = {DURATION[band]}\n"
            f"max_iteration = {SA_ITER_CAP[band]}\n")


def heuristic_ls(neighbor: str, band: str) -> str:
    return (f'[[heuristics]]\nkind = "LocalSearch"\nneighbor = "{neighbor}"\n'
            f"{stop_duration(band)}")


def heuristic_ts(problem: str, neighbor: str, band: str) -> str:
    t = TABU_TENURE[problem][band]
    return (f'[[heuristics]]\nkind = "TabuSearch"\nneighbor = "{neighbor}"\n'
            f"tabu_tenure = {fmt_array(t)}\n{stop_duration(band)}")


def heuristic_lahc(neighbor: str, band: str) -> str:
    return (f'[[heuristics]]\nkind = "LateAcceptanceHillClimbing"\nneighbor = "{neighbor}"\n'
            f"history_length = {LAHC_HISTORY[band]}\n{stop_duration(band)}")


def heuristic_sa(problem: str, neighbor: str, band: str) -> str:
    return (f'[[heuristics]]\nkind = "SimulatedAnnealing"\nneighbor = "{neighbor}"\n'
            f"initial_temperature = {SA_T0[problem]}\ncooling_rate = {SA_COOLING}\n"
            f"{stop_duration_with_iter_cap(band)}")


def heuristic_ga(problem: str, neighbor: str, band: str) -> str:
    t = TABU_TENURE[problem][band]
    return (f'[[heuristics]]\nkind = "GeneticAlgorithm"\n'
            f"population_size = {GA_POP[band]}\n"
            f'crossover_kind = "{CROSSOVER[problem]}"\n'
            f'parent_selection = "Tournament"\n'
            f"{stop_duration(band)}"
            f'\n[[heuristics.steps]]\nkind = "TabuSearch"\nneighbor = "{neighbor}"\n'
            f"tabu_tenure = {fmt_array(t)}\n"
            f"[heuristics.steps.stop_condition]\nmax_failed_update = 200\n")


def heuristic_iterated(neighbor: str, band: str) -> str:
    return (f'[[heuristics]]\nkind = "Iterated"\n{stop_duration(band)}'
            f'\n[[heuristics.steps]]\nkind = "LocalSearch"\nneighbor = "{neighbor}"\n'
            f"[heuristics.steps.stop_condition]\nmax_failed_update = 1\n"
            f'\n[[heuristics.steps]]\nkind = "SimulatedAnnealing"\nneighbor = "{neighbor}"\n'
            f"initial_temperature = 5.0\ncooling_rate = 0.99\n"
            f"[heuristics.steps.stop_condition]\nmax_iteration = 200\n")


def heuristic_restart(problem: str, neighbor: str, band: str) -> str:
    t = TABU_TENURE[problem][band]
    return (f'[[heuristics]]\nkind = "Restart"\n{stop_duration(band)}'
            f"\n[heuristics.restart_condition]\nmax_failed_update = {RESTART_FAILED[band]}\n"
            f'\n[[heuristics.steps]]\nkind = "TabuSearch"\nneighbor = "{neighbor}"\n'
            f"tabu_tenure = {fmt_array(t)}\n"
            f"[heuristics.steps.stop_condition]\nmax_iteration = 50000\n")


def general_toml(problem: str, band: str) -> str:
    parts = [f"# Auto-generated by data/scripts/gen_benchmark_configs.py — do not edit by hand.",
             f"# Profile: {problem} / {band} (general heuristics)",
             f"# See docs/benchmarks/profiles.md",
             f"",
             f"num_runs = {NUM_RUNS}",
             f"",
             instance_blocks(problem, band)]
    nbrs = NEIGHBORS[problem]
    for n in nbrs:
        parts.append(heuristic_ls(n, band))
    for n in nbrs:
        parts.append(heuristic_ts(problem, n, band))
    for n in nbrs:
        parts.append(heuristic_lahc(n, band))
    for n in nbrs:
        parts.append(heuristic_sa(problem, n, band))
    # GA / Iterated / Restart use the first neighbor for the inner step
    inner = nbrs[0]
    parts.append(heuristic_ga(problem, inner, band))
    parts.append(heuristic_iterated(inner, band))
    parts.append(heuristic_restart(problem, inner, band))
    return "\n".join(parts) + "\n"


def lkh_toml(band: str) -> str:
    return (f"# Auto-generated. Profile: Tsp / {band} (LinKernighanHelsgott).\n"
            f"# See docs/benchmarks/profiles.md\n\n"
            f"num_runs = {NUM_RUNS}\n\n"
            f"{instance_blocks('Tsp', band)}\n"
            f'[[heuristics]]\nkind = "LinKernighanHelsgott"\n'
            f"num_neighbors = 5\nmax_depth = 5\n"
            f"{stop_duration(band)}")


def main() -> int:
    written = 0
    for problem in NEIGHBORS:
        out_dir = OUT / PROBLEM_DIR[problem]
        out_dir.mkdir(parents=True, exist_ok=True)
        for band in ("small", "medium", "large"):
            path = out_dir / f"general_{band}.toml"
            path.write_text(general_toml(problem, band))
            written += 1
    # LKH applies to TSP
    for band in ("small", "medium", "large"):
        path = OUT / "tsp" / f"lkh_{band}.toml"
        path.write_text(lkh_toml(band))
        written += 1
    print(f"wrote {written} TOML configs under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
