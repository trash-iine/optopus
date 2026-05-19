# Benchmark Profiles

Stop-condition, hyperparameter, and `num_runs` settings used for the
comprehensive (problem × heuristic) benchmark sweep. The intent is that every
TOML under `data/benchmarks/<problem>/` follows one of the size-band profiles
below, so results from different runs remain comparable.

`num_runs = 10` for every (problem, heuristic, instance) triple, matching the
existing BreakoutLocalSearch reports.

## Size bands per problem

| Problem | small | medium | large |
|---|---|---|---|
| MaxCut (GSET) | G1–G21, G43–G47, G51–G54 (n ≤ 1000) | G22–G42 (n = 2000), G48–G50 (n = 3000) | G55–G81 (n ≥ 5000) |
| QUBO (bqp) | bqp50, bqp100 | bqp250, bqp500 | bqp1000 |
| SAT (uf*) | uf50, uf75 | uf100, uf150 | uf200 |
| TSP (TSPLIB) | burma14, eil51, att48, berlin52 | eil101, ch150 | dsj1000 |
| JSSP (OR-Library) | ft06, ft10, ft20, la01–la05, abz5 | la06–la25, abz6–abz9, orb01–orb10 | la26–la40, swv*, yn*, ta* |
| VertexCover | data/max_cut/sample.txt + GSET small | GSET medium | GSET large |

## Wall-clock budget per `(heuristic, band)`

All heuristics in a single band share the same wall-clock budget so the
comparison is fair. `max_iteration` is left unset for general heuristics
(`max_duration_secs` is the only stop condition); for SA, an iteration cap is
added as a safety net.

| Heuristic | small | medium | large |
|---|---|---|---|
| LocalSearch | duration 30s | duration 120s | duration 600s |
| TabuSearch | duration 30s | duration 120s | duration 600s |
| LateAcceptanceHillClimbing (LAHC) | duration 30s | duration 120s | duration 600s |
| SimulatedAnnealing | duration 30s, iter 1e7 | duration 120s, iter 5e7 | duration 600s, iter 2e8 |
| GeneticAlgorithm | duration 30s | duration 120s | duration 600s |
| Iterated (LS + SA-perturb) | duration 30s | duration 120s | duration 600s |
| Restart (TabuSearch inner) | duration 30s | duration 120s | duration 600s |

### Special-purpose heuristics

| Heuristic | Target problem(s) | small | medium | large |
|---|---|---|---|---|
| BreakoutLocalSearchForMaxCut | MaxCut | (existing: 113s @ G1–G21) | (existing: 505s @ G22–G42) | (existing: 2993s @ G55–G64, 7388s @ G65) |
| LinKernighanHelsgott | TSP | duration 30s | duration 120s | duration 600s |

The BLS budget mirrors the curated runs in `docs/benchmarks/data/bls_maxcut_gset_*.toml`
so existing results carry over; LKH uses the same budget as the general
heuristics for direct comparison.

## Hyperparameters per heuristic (band-invariant unless noted)

Hyperparameters are kept constant across bands to avoid an explosion of
configurations. Where a parameter scales with problem size, the rule is given
in the table.

| Heuristic | Parameter | Value |
|---|---|---|
| TabuSearch | `tabu_tenure` | `[3, max(10, n/10)]` (MaxCut/QUBO/SAT/VC: n = #vars; TSP: n = #cities; JSSP: n = #jobs·#machines) |
| SimulatedAnnealing | `initial_temperature` | 100.0 (MaxCut/QUBO/SAT/VC/JSSP), 1000.0 (TSP) |
| SimulatedAnnealing | `cooling_rate` | 0.9995 |
| LateAcceptanceHillClimbing | `history_length` | small: 100, medium: 500, large: 2000 |
| GeneticAlgorithm | `population_size` | small: 50, medium: 100, large: 200 |
| GeneticAlgorithm | `crossover_kind` | `"Uniform"` (MaxCut/QUBO/SAT/VC), `"Order"` (TSP), `"Ppx"` (JSSP) |
| GeneticAlgorithm | mutation step (steps[0]) | `TabuSearch` with `tabu_tenure=[3, 10]`, `max_failed_update=200` |
| Iterated | search step | `LocalSearch` with `max_failed_update=1` |
| Iterated | perturbation step | `SimulatedAnnealing` with `T0=5.0`, `α=0.99`, `max_iteration=200` |
| Restart | inner | `TabuSearch` with `tabu_tenure=[3, 10]`, `max_iteration=50_000` |
| Restart | `restart_condition` | `max_failed_update = max(5_000, n·5)` |
| BLS (MaxCut) | `tabu_tenure`, `t`, `l0`, `p0`, `q` | Use existing per-size values from `bls_maxcut_gset_*.toml` |
| LKH (TSP) | `num_neighbors`, `max_depth` | 5, 5 (library defaults) |

## Neighbor enumeration policy

Every general heuristic is run on **every applicable neighbor type** for the
target problem, listed as separate entries in `[[heuristics]]`. Each neighbor
becomes a row in the rendered Markdown table.

| Problem | Neighbors |
|---|---|
| MaxCut | `Flip`, `Swap` |
| QUBO | `Flip`, `Swap` |
| SAT | `Flip`, `Swap` |
| TSP | `TwoOpt`, `Relocate` |
| JSSP | `Swap`, `Relocate` |
| VertexCover | `Flip`, `Swap` |

So e.g. MaxCut × LocalSearch produces two rows (Flip + Swap) in
`local_search.md`.

## Why these values

- **Wall-clock-first stop condition**: per-iteration cost varies wildly across
  problems and neighbor types (TSP 2-opt scans O(n²), MaxCut Flip is O(1) with
  cached gains). Using `max_duration_secs` makes results comparable across
  algorithm rows in the rendered tables.
- **30/120/600 s tiers**: matches roughly 2×/4× scaling per band and totals
  about 12 h of CPU time per (problem, heuristic) chain assuming serial
  execution (rayon parallelism shrinks this).
- **`num_runs = 10`**: gives reasonable std for stochastic heuristics and
  matches existing BLS reports for cross-table joinability.
- **Band-invariant hyperparameters**: the goal is "out-of-the-box" comparison
  across heuristics, not per-problem tuning. Per-problem tuning is left to
  future work (a separate `tuning/` track).
