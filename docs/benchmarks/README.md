# Benchmarks

Aggregated comparison of every implemented heuristic across every supported
problem type. Each row is `(problem, instance, neighbor, heuristic)` averaged
over 10 runs.

> ⚠️ **Reference values only.** These benchmarks were collected in a
> non-isolated environment (shared machine, background load) and some code
> paths are deliberately generalized, trading raw speed for flexibility. The
> numbers therefore do **not** reflect each algorithm's true performance — treat
> them as indicative reference values, not a rigorous ranking.

## Coverage

Small (≤30s budget), medium (120s), and large (600s) bands are all complete.

| Problem | Instances (small / medium / large) | Neighbors | Heuristics |
|---|---|---|---|
| MaxCut (GSET) | 30 / 24 / 17 | Flip, Swap | LS, TS, LAHC, SA, GA, Iterated, Restart, BLS |
| QUBO (bqp) | 20 / 20 / 10 | Flip, Swap | LS, TS, LAHC, SA, GA, Iterated, Restart |
| MaxSAT (uf) | 20 / 20 / 10 | Flip, Swap | LS, TS, LAHC, SA, GA, Iterated, Restart |
| TSP (TSPLIB) | 4 / 2 / 1 | TwoOpt, Relocate | LS, TS, LAHC, SA, GA, Iterated, Restart, LKH |
| JSSP (OR-Library) | 9 / 34 / 39 | Swap, Relocate | LS, TS, LAHC, SA, GA, Iterated, Restart |
| VertexCover (GSET) | 30 / 24 / 17 | Flip, Swap | LS, TS, LAHC, SA, GA, Iterated, Restart |

The BreakoutLocalSearch (BLS) sweep for MaxCut spans the full GSET set
G1–G81 (vertex-size bins 800–20000), reported separately as
`bls_maxcut_gset_<bin>` — the specialized solver's numbers are not directly
comparable to the wall-clock-budgeted general heuristics.

## Browse interactively

Published on GitHub Pages: **https://trash-iine.github.io/optopus/** (Benchmark
viewer card). The viewer loads its data via `fetch` from
`data/index.json` + `data/**/*.slim.toml`, so it must be served over HTTP —
locally run `cd docs && python3 -m http.server` and open
`http://localhost:8000/benchmarks/viewer.html` (opening the `file://` path
directly no longer works). It offers a
filterable, sortable cross-heuristic comparison. Switch between
**Comparison** (problem × instance rows × heuristic columns, best cell
highlighted per row) and **Flat** views. Filters: problem, heuristic,
neighbor, size band, instance substring; metric selector covers best /
avg / worst / std / time-to-best / total time. Filter state is encoded
in the URL hash so a view can be shared by copying the URL.

## Profiles

Stop-condition, hyperparameter, and `num_runs` settings used for the
comprehensive (problem × heuristic) benchmark sweep. The intent is that every
TOML under `data/benchmarks/<problem>/` follows one of the size-band profiles
below, so results from different runs remain comparable.

`num_runs = 10` for every (problem, heuristic, instance) triple, matching the
existing BreakoutLocalSearch reports.

### Size bands per problem

| Problem | small | medium | large |
|---|---|---|---|
| MaxCut (GSET) | G1–G21, G43–G47, G51–G54 (n ≤ 1000) | G22–G42 (n = 2000), G48–G50 (n = 3000) | G55–G81 (n ≥ 5000) |
| QUBO (bqp) | bqp50, bqp100 | bqp250, bqp500 | bqp1000 |
| SAT (uf*) | uf50, uf75 | uf100, uf150 | uf200 |
| TSP (TSPLIB) | burma14, eil51, att48, berlin52 | eil101, ch150 | dsj1000 |
| JSSP (OR-Library) | ft06, ft10, ft20, la01–la05, abz5 | la06–la25, abz6–abz9, orb01–orb10 | la26–la40, swv*, yn*, ta* |
| VertexCover | data/instances/max_cut/sample.txt + GSET small | GSET medium | GSET large |

### Wall-clock budget per `(heuristic, band)`

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

#### Special-purpose heuristics

| Heuristic | Target problem(s) | small | medium | large |
|---|---|---|---|---|
| BreakoutLocalSearchForMaxCut | MaxCut | 160 M-iter budget @ G1–G54 | 160 M-iter budget @ G22–G50 | per-bin iter budget (160 M → 4 B) @ G55–G81, up to 20000 vertices |
| LinKernighanHelsgaun | TSP | duration 30s | duration 120s | duration 600s |

The BLS budget mirrors the curated runs in `docs/benchmarks/data/bls_maxcut_gset_*.toml`
so existing results carry over; LKH uses the same budget as the general
heuristics for direct comparison.

### Hyperparameters per heuristic (band-invariant unless noted)

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

### Neighbor enumeration policy

Every general heuristic is run on **every applicable neighbor type** for the
target problem, listed as separate entries in `[[heuristics]]`. Each neighbor
becomes a row in the benchmark viewer.

| Problem | Neighbors |
|---|---|
| MaxCut | `Flip`, `Swap` |
| QUBO | `Flip`, `Swap` |
| SAT | `Flip`, `Swap` |
| TSP | `TwoOpt`, `Relocate` |
| JSSP | `Swap`, `Relocate` |
| VertexCover | `Flip`, `Swap` |

So e.g. MaxCut × LocalSearch produces two rows (Flip + Swap) in the
[benchmark viewer](viewer.html).

### Why these values

- **Wall-clock-first stop condition**: per-iteration cost varies wildly across
  problems and neighbor types (TSP 2-opt scans O(n²), MaxCut Flip is O(1) with
  cached gains). Using `max_duration_secs` makes results comparable across
  algorithm rows in the benchmark viewer.
- **30/120/600 s tiers**: matches roughly 2×/4× scaling per band and totals
  about 12 h of CPU time per (problem, heuristic) chain assuming serial
  execution (rayon parallelism shrinks this).
- **`num_runs = 10`**: gives reasonable std for stochastic heuristics and
  matches existing BLS reports for cross-table joinability.
- **Band-invariant hyperparameters**: the goal is "out-of-the-box" comparison
  across heuristics, not per-problem tuning. Per-problem tuning is left to
  future work (a separate `tuning/` track).

## Pipeline & reproduce

- **Run configs**: TOMLs under `data/benchmarks/<problem>/{general,lkh}_{small,medium,large}.toml`
  are generated by `data/benchmarks/scripts/gen_benchmark_configs.py` from the
  profiles above.
- **Curated results**: TOMLs under `docs/benchmarks/data/<problem>/` are
  publish-ready outputs of `optopus <config>` runs. These raw TOMLs carry the
  full per-run solution vectors (100s of MB per large-band file) and are
  **gitignored — kept local only**, never committed.
- **`render.py`** (`pip install tomli-w` first) aggregates the curated raw TOMLs
  into the viewer's data source — a slim copy of each with the heavy per-run
  `runs` arrays stripped (`*.slim.toml`, ~2.8 MB total) plus an `index.json` the
  browser fetches to discover them. Each result carries its own `problem` field
  (emitted by the runner); `index.json` is derived from the slim TOMLs, not
  hand-maintained. **The `*.slim.toml` + `index.json` are what you commit.** The
  viewer parses those slim TOMLs client-side with the vendored parser in
  `vendor/smol-toml.js` (BSD-3-Clause).
- **GitHub Pages**: `.github/workflows/deploy-pages.yml` deploys the committed
  `docs/` (the slim TOMLs + index) on every push to `main`. It does **not**
  re-run `render.py` — run it locally and commit its output. Requires
  Settings → Pages → Source = "GitHub Actions".

```bash
cargo build --release

# Re-run any band/problem; output lands in result/<timestamp>.toml
./target/release/optopus data/benchmarks/maxcut/general_small.toml
mv result/<timestamp>.toml docs/benchmarks/data/maxcut/general_gset_small.toml

# Regenerate viewer data (slim TOML + index)
python3 docs/benchmarks/render.py
```

The full small+medium sweep takes roughly 40 wall-clock hours on an 11-core
machine; see `data/benchmarks/scripts/run_small_band.sh` and `run_medium_band.sh`.
