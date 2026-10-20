"""Generate benchmark TOML configs under `data/benchmarks/<problem>/`.

Drives the comprehensive benchmark sweep (see
`docs/benchmarks/README.md#profiles`). One TOML per (problem, band, track) where:

    track ∈ {"general", "lkh", "hgs", "specific"}
    band  ∈ {"small", "medium", "large"}

The `general` track enumerates LocalSearch, TabuSearch,
LateAcceptanceHillClimbing, SimulatedAnnealing (× each applicable neighbor),
plus GeneticAlgorithm, Iterated and Restart, then VariableNeighborhoodSearch,
PopulationAnnealing, ParallelTempering and ReinforcementLearningSearch on the
first neighbor. The later kinds are appended after the
original eleven so the earlier kinds keep their position in the list.

The `specific` track holds the heuristics that only some problems implement
(ALNS, Iterated Greedy, GRASP, WalkSAT). `lkh` and `hgs` keep their names
from the earlier sweeps.

Usage: python3 data/benchmarks/scripts/gen_benchmark_configs.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

OUT = Path(__file__).resolve().parent.parent

BANDS = ("small", "medium", "large")
# Five runs at the large band, where a run takes ten minutes.
NUM_RUNS = {"small": 10, "medium": 10, "large": 5}
# Master seed: makes every run reproducible (per-run seeds are derived from it).
SEED = 42
DURATION = {"small": 30.0, "medium": 120.0, "large": 600.0}
SA_ITER_CAP = {"small": 10_000_000, "medium": 50_000_000, "large": 200_000_000}
LAHC_HISTORY = {"small": 100, "medium": 500, "large": 2000}
GA_POP = {"small": 50, "medium": 100, "large": 200}
RESTART_FAILED = {"small": 5000, "medium": 10_000, "large": 50_000}
REPLICAS = 20
# Shake lengths of the VariableNeighborhoodSearch, one RandomWalk each.
VNS_SHAKES = (5, 10, 20)
# Ratio between the coldest and the hottest temperature of the replica annealers.
BETA_RATIO = 50.0
# PopulationAnnealing's default reset period, over which beta climbs the whole range.
PA_RESET_PERIOD = 400

NEIGHBORS = {
    "MaxCut": ["Flip", "Swap"],
    "Qubo": ["Flip", "Swap"],
    "Sat": ["Flip", "Swap"],
    "Tsp": ["TwoOpt", "Relocate"],
    "JobShop": ["Swap", "Relocate"],
    "VertexCover": ["Flip", "Swap"],
    "Vrp": ["Relocate", "Swap", "TwoOpt"],
    "FlowShop": ["Relocate", "Swap"],
    "Qap": ["Swap"],
    "BinPacking": ["Relocate", "Swap"],
    "GraphColoring": ["Flip", "Swap"],
}

CROSSOVER = {
    "MaxCut": "Uniform",
    "Qubo": "Uniform",
    "Sat": "Uniform",
    "Tsp": "Order",
    "JobShop": "Ppx",
    "VertexCover": "Uniform",
    "Vrp": "Order",
    "FlowShop": "Order",
    "Qap": "Order",
    "BinPacking": "Group",
    "GraphColoring": "Uniform",
}

TABU_TENURE = {
    "MaxCut": {"small": [3, 80], "medium": [3, 200], "large": [3, 500]},
    "Qubo": {"small": [3, 30], "medium": [3, 100], "large": [3, 200]},
    "Sat": {"small": [3, 20], "medium": [3, 50], "large": [3, 100]},
    "Tsp": {"small": [3, 30], "medium": [3, 100], "large": [3, 200]},
    "JobShop": {"small": [3, 30], "medium": [3, 80], "large": [3, 200]},
    "VertexCover": {"small": [3, 80], "medium": [3, 200], "large": [3, 500]},
    "Vrp": {"small": [3, 30], "medium": [3, 80], "large": [3, 200]},
    "FlowShop": {"small": [3, 15], "medium": [3, 30], "large": [3, 60]},
    "Qap": {"small": [3, 30], "medium": [20, 60], "large": [20, 100]},
    "BinPacking": {"small": [3, 15], "medium": [3, 30], "large": [3, 60]},
    "GraphColoring": {"small": [3, 80], "medium": [3, 200], "large": [3, 500]},
}

SA_T0 = {"MaxCut": 100.0, "Qubo": 100.0, "Sat": 100.0, "Tsp": 1000.0,
         "JobShop": 100.0, "VertexCover": 100.0, "Vrp": 100.0,
         "FlowShop": 10.0, "Qap": 1000.0, "BinPacking": 10.0, "GraphColoring": 3.0}
SA_COOLING = 0.9995

# Hottest temperature of PopulationAnnealing and ParallelTempering. The coldest
# is this divided by BETA_RATIO. Each value is the best of SA_T0 times
# 0.01, 0.1, 1, 10 and 100 on two small-band instances at 5 s.
REPLICA_T_HOT = {"MaxCut": 10.0, "Qubo": 100.0, "Sat": 10.0, "Tsp": 10.0,
                 "JobShop": 100.0, "VertexCover": 10.0, "Vrp": 100.0,
                 "FlowShop": 10.0, "Qap": 10000.0, "BinPacking": 10.0,
                 "GraphColoring": 0.03}

GSET_SMALL = list(range(1, 22)) + list(range(43, 48)) + list(range(51, 55))
GSET_MEDIUM = list(range(22, 43)) + list(range(48, 51))
GSET_LARGE = [55, 58, 60, 63, 64, 70, 72, 77, 81]


def gset(ns: Iterable[int]) -> list[str]:
    return [f"data/instances/max_cut/G{n}" for n in ns]


def taillard_pfsp(firsts: Iterable[int], per_class: int = 3) -> list[str]:
    """The first `per_class` instances of each Taillard class starting at `firsts`."""
    return [f"data/instances/flow_shop/taillard/ta{f + i:03d}.txt"
            for f in firsts for i in range(per_class)]


def qaplib(names: Iterable[str]) -> list[str]:
    return [f"data/instances/qap/{n}.dat" for n in names]


def falkenauer(classes: Iterable[str], per_class: int = 5) -> list[str]:
    return [f"data/instances/bin_packing/falkenauer/{c}_{i:02d}.txt"
            for c in classes for i in range(per_class)]


# Instance globs per (problem, band). Lists become multiple [[instances]] blocks.
INSTANCES: dict[tuple[str, str], list[str]] = {
    ("MaxCut", "small"): gset(GSET_SMALL),
    ("MaxCut", "medium"): gset(GSET_MEDIUM),
    ("MaxCut", "large"): gset(GSET_LARGE),
    ("Qubo", "small"):  ["data/instances/qubo/bqp/bqp50_*.txt",  "data/instances/qubo/bqp/bqp100_*.txt"],
    ("Qubo", "medium"): ["data/instances/qubo/bqp/bqp250_*.txt", "data/instances/qubo/bqp/bqp500_*.txt"],
    ("Qubo", "large"):  ["data/instances/qubo/bqp/bqp1000_*.txt"],
    ("Sat", "small"):  ["data/instances/sat/satlib/uf50-218/*.cnf",  "data/instances/sat/satlib/uf75-325/*.cnf"],
    ("Sat", "medium"): ["data/instances/sat/satlib/uf100-430/*.cnf", "data/instances/sat/satlib/uf150-645/*.cnf"],
    ("Sat", "large"):  ["data/instances/sat/satlib/uf200-860/*.cnf"],
    ("Tsp", "small"):  ["data/instances/tsp/burma14.tsp", "data/instances/tsp/eil51.tsp",
                       "data/instances/tsp/att48.tsp", "data/instances/tsp/berlin52.tsp"],
    ("Tsp", "medium"): ["data/instances/tsp/eil101.tsp", "data/instances/tsp/ch150.tsp"],
    ("Tsp", "large"):  ["data/instances/tsp/dsj1000.tsp"],
    ("JobShop", "small"):  ["data/instances/jssp/orlib/ft06.txt",  "data/instances/jssp/orlib/ft10.txt",
                            "data/instances/jssp/orlib/ft20.txt",
                            *[f"data/instances/jssp/orlib/la0{n}.txt" for n in range(1, 6)],
                            "data/instances/jssp/orlib/abz5.txt"],
    ("JobShop", "medium"): [*[f"data/instances/jssp/orlib/la{n:02d}.txt" for n in range(6, 26)],
                            *[f"data/instances/jssp/orlib/abz{n}.txt" for n in range(6, 10)],
                            *[f"data/instances/jssp/orlib/orb{n:02d}.txt" for n in range(1, 11)]],
    # One or two of each shape among la26-40, swv01-20 and yn1-4.
    ("JobShop", "large"):  [*[f"data/instances/jssp/orlib/la{n}.txt" for n in (26, 29, 32, 36, 38, 40)],
                            *[f"data/instances/jssp/orlib/swv{n:02d}.txt" for n in (1, 6, 11, 16)],
                            *[f"data/instances/jssp/orlib/yn{n}.txt" for n in (1, 3)]],
    ("VertexCover", "small"): gset(GSET_SMALL),
    ("VertexCover", "medium"): gset(GSET_MEDIUM),
    ("VertexCover", "large"): gset(GSET_LARGE),
    # CVRPLIB "X" set, banded by customer count (see data/instances/README.md).
    ("Vrp", "small"):  ["data/instances/vrp/demo16.vrp",
                        *[f"data/instances/vrp/X-n{n}-k{k}.vrp" for n, k in (
                            (101, 25), (110, 13), (125, 30), (139, 10),
                            (153, 22), (176, 26), (195, 51))]],
    ("Vrp", "medium"): [f"data/instances/vrp/X-n{n}-k{k}.vrp" for n, k in (
                            (214, 11), (242, 48), (270, 35), (303, 21),
                            (344, 43), (401, 29), (459, 26))],
    ("Vrp", "large"):  [f"data/instances/vrp/X-n{n}-k{k}.vrp" for n, k in (
                            (502, 39), (561, 42), (627, 43), (701, 44),
                            (801, 40), (895, 37), (1001, 43))],
    # Taillard classes of ten: 20 jobs (ta001-030), 50 and 100 jobs (ta031-090),
    # 200 and 500 jobs (ta091-120).
    ("FlowShop", "small"):  taillard_pfsp((1, 11, 21)),
    ("FlowShop", "medium"): taillard_pfsp((31, 41, 51, 61, 71, 81)),
    ("FlowShop", "large"):  taillard_pfsp((91, 101, 111)),
    # QAPLIB, banded by n. Not bundled, fetch with data/instances/scripts/fetch_qaplib.sh.
    ("Qap", "small"):  qaplib(("nug12", "nug20", "nug30", "tai12a", "tai15a",
                               "tai17a", "tai20a", "tai25a", "tai30a")),
    ("Qap", "medium"): qaplib(("tai35a", "tai40a", "sko42", "tai50a", "sko56", "tai60a")),
    ("Qap", "large"):  qaplib(("sko72", "tai80a", "sko100a", "tai100a")),
    # Falkenauer's uniform (u) and triplet (t) classes, the first five of each
    # (three at the large band).
    ("BinPacking", "small"):  falkenauer(("u120", "t60")),
    ("BinPacking", "medium"): falkenauer(("u250", "t120", "t249")),
    ("BinPacking", "large"):  falkenauer(("u500", "u1000", "t501"), per_class=3),
    # A subset of the G-set covering its random, toroidal and planar graphs.
    ("GraphColoring", "small"):  gset((1, 2, 3, 11, 12, 13, 14, 15, 16, 43)),
    ("GraphColoring", "medium"): gset((22, 23, 32, 33, 35, 36, 48, 49)),
    ("GraphColoring", "large"):  gset((55, 56, 57, 60, 62, 70)),
}

PROBLEM_DIR = {
    "MaxCut": "maxcut", "Qubo": "qubo", "Sat": "sat", "Tsp": "tsp",
    "JobShop": "jssp", "VertexCover": "vertex_cover", "Vrp": "vrp",
    "FlowShop": "flow_shop", "Qap": "qap", "BinPacking": "bin_packing",
    "GraphColoring": "graph_coloring",
}

# Problems with a `specific_<band>.toml`, and the problem-specific heuristics in it.
SPECIFIC = {
    "Tsp": ("alns", "grasp"),
    "FlowShop": ("iterated_greedy", "alns", "grasp"),
    "BinPacking": ("alns", "grasp"),
    "Sat": ("walksat", "walksat_adaptive"),
}


def fmt_array(xs: Iterable[int]) -> str:
    return "[" + ", ".join(str(x) for x in xs) + "]"


def header(title: str, band: str) -> str:
    return (f"# Auto-generated by data/benchmarks/scripts/gen_benchmark_configs.py, do not edit by hand.\n"
            f"# Profile: {title}\n"
            f"# See docs/benchmarks/README.md#profiles\n\n"
            f"num_runs = {NUM_RUNS[band]}\nseed = {SEED}\n")


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


def heuristic_vns(neighbor: str, band: str) -> str:
    shakes = "".join(
        f'\n[[heuristics.steps]]\nkind = "RandomWalk"\nneighbor = "{neighbor}"\n'
        f"[heuristics.steps.stop_condition]\nmax_iteration = {k}\n"
        for k in VNS_SHAKES)
    return (f'[[heuristics]]\nkind = "VariableNeighborhoodSearch"\n{stop_duration(band)}'
            f'\n[[heuristics.steps]]\nkind = "LocalSearch"\nneighbor = "{neighbor}"\n'
            f"[heuristics.steps.stop_condition]\nmax_failed_update = 1\n"
            f"{shakes}")


def beta_range(problem: str) -> tuple[float, float]:
    t_hot = REPLICA_T_HOT[problem]
    return 1.0 / t_hot, BETA_RATIO / t_hot


def heuristic_pa(problem: str, neighbor: str, band: str) -> str:
    beta_min, beta_max = beta_range(problem)
    delta = (beta_max - beta_min) / PA_RESET_PERIOD
    return (f'[[heuristics]]\nkind = "PopulationAnnealing"\nneighbor = "{neighbor}"\n'
            f"population_size = {REPLICAS}\n"
            f"initial_beta = {beta_min:.6g}\ndelta_beta = {delta:.6g}\n"
            f"{stop_duration(band)}")


def heuristic_pt(problem: str, neighbor: str, band: str) -> str:
    beta_min, beta_max = beta_range(problem)
    return (f'[[heuristics]]\nkind = "ParallelTempering"\nneighbor = "{neighbor}"\n'
            f"num_replicas = {REPLICAS}\n"
            f"beta_min = {beta_min:.6g}\nbeta_max = {beta_max:.6g}\n"
            f"{stop_duration(band)}")


def heuristic_rl(neighbor: str, band: str) -> str:
    return (f'[[heuristics]]\nkind = "ReinforcementLearningSearch"\nneighbor = "{neighbor}"\n'
            f"{stop_duration(band)}")


def heuristic_alns(band: str) -> str:
    return (f'[[heuristics]]\nkind = "AdaptiveLargeNeighborhoodSearch"\n'
            f"removal_fraction = 0.15\ncooling_rate = 0.9995\n"
            f"{stop_duration(band)}")


def heuristic_grasp(neighbor: str, band: str) -> str:
    """GRASP, a randomized greedy construction then a descent, in a loop."""
    return (f'[[heuristics]]\nkind = "Iterated"\n{stop_duration(band)}'
            f'\n[[heuristics.steps]]\nkind = "LocalSearch"\nneighbor = "{neighbor}"\n'
            f'\n[[heuristics.steps]]\nkind = "GreedyRandomizedConstruction"\nalpha = 0.2\n')


def heuristic_iterated_greedy(band: str) -> str:
    return (f'[[heuristics]]\nkind = "IteratedGreedy"\n'
            f"removal_count = 4\ntemperature_factor = 0.4\n"
            f"{stop_duration(band)}")


def heuristic_walksat(band: str, adaptive: bool) -> str:
    flag = "adaptive_noise = true\n" if adaptive else ""
    return (f'[[heuristics]]\nkind = "WalkSat"\nnoise = 0.3\n{flag}'
            f"{stop_duration(band)}")


def general_toml(problem: str, band: str) -> str:
    parts = [header(f"{problem} / {band} (general heuristics)", band),
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
    # The composite and replica kinds use the first neighbor only.
    inner = nbrs[0]
    parts.append(heuristic_ga(problem, inner, band))
    parts.append(heuristic_iterated(inner, band))
    parts.append(heuristic_restart(problem, inner, band))
    parts.append(heuristic_vns(inner, band))
    parts.append(heuristic_pa(problem, inner, band))
    parts.append(heuristic_pt(problem, inner, band))
    parts.append(heuristic_rl(inner, band))
    return "\n".join(parts) + "\n"


def specific_toml(problem: str, band: str) -> str:
    inner = NEIGHBORS[problem][0]
    builders = {
        "alns": lambda: heuristic_alns(band),
        "grasp": lambda: heuristic_grasp(inner, band),
        "iterated_greedy": lambda: heuristic_iterated_greedy(band),
        "walksat": lambda: heuristic_walksat(band, adaptive=False),
        "walksat_adaptive": lambda: heuristic_walksat(band, adaptive=True),
    }
    parts = [header(f"{problem} / {band} (problem-specific heuristics)", band),
             instance_blocks(problem, band)]
    parts.extend(builders[name]() for name in SPECIFIC[problem])
    return "\n".join(parts) + "\n"


def lkh_toml(band: str) -> str:
    return (f"{header(f'Tsp / {band} (LinKernighanHelsgaun)', band)}\n"
            f"{instance_blocks('Tsp', band)}\n"
            f'[[heuristics]]\nkind = "LinKernighanHelsgaun"\n'
            f"num_neighbors = 5\nmax_depth = 5\n"
            f"{stop_duration(band)}")


def vrp_special_toml(band: str) -> str:
    """The bespoke CVRP heuristics, on equal budget so they can be compared."""
    return (f"{header(f'Vrp / {band} (bespoke CVRP heuristics)', band)}\n"
            f"{instance_blocks('Vrp', band)}\n"
            f"{heuristic_alns(band)}"
            f'\n[[heuristics]]\nkind = "HybridGeneticSearch"\n'
            f"min_population_size = 25\ngeneration_size = 40\ngranularity = 20\n"
            f"target_feasible = 0.2\nrestart_generations = 20000\n"
            f"{stop_duration(band)}"
            f"\n{heuristic_grasp(NEIGHBORS['Vrp'][0], band)}")


def main() -> int:
    written = 0
    for problem in NEIGHBORS:
        out_dir = OUT / PROBLEM_DIR[problem]
        out_dir.mkdir(parents=True, exist_ok=True)
        for band in BANDS:
            (out_dir / f"general_{band}.toml").write_text(general_toml(problem, band))
            written += 1
            if problem in SPECIFIC:
                (out_dir / f"specific_{band}.toml").write_text(specific_toml(problem, band))
                written += 1
    # LKH applies to TSP
    for band in BANDS:
        (OUT / "tsp" / f"lkh_{band}.toml").write_text(lkh_toml(band))
        written += 1
    # ALNS, HGS and GRASP on VRP
    for band in BANDS:
        (OUT / "vrp" / f"hgs_{band}.toml").write_text(vrp_special_toml(band))
        written += 1
    print(f"wrote {written} TOML configs under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
