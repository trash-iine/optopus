# Benchmarks

Aggregated comparison of the general heuristics, and the problem-specific solvers. 
Each row is `(problem, instance, neighbor, heuristic)` averaged over five runs (ten in the small and medium bands of the 2026-10 sweep, and in its BLS runs on the G-set graphs of 1000 and 10000 vertices).

The numbers themselves (and the exact configuration behind each one) live in
the [benchmark viewer](viewer.html); click any cell. This file covers how those
numbers were produced: what is covered, how the size bands are drawn, why the
settings are what they are, and how to reproduce a run.

> ⚠️ Reference values only. These benchmarks were collected in a
> non-isolated environment (shared machine, background load) and some code
> paths are deliberately generalized, trading raw speed for flexibility. The
> numbers therefore do not reflect each algorithm's true performance, treat
> them as indicative reference values, not a rigorous ranking.

## Coverage

Small (≤30s budget), medium (120s), and large (600s) bands.

Every problem runs the eleven general heuristics, LS, TS, LAHC, SA, GA,
Iterated, Restart, VNS, PA, PT and RL. The last column lists the
problem-specific solvers that run beside them.

| Problem | Instances (small / medium / large) | Neighbors | Problem-specific |
|---|---|---|---|
| MaxCut (GSET) | 30 / 24 / 17 | Flip, Swap | BLS |
| QUBO (bqp) | 6 / 6 / 3 | Flip, Swap | |
| MaxSAT (uf) | 6 / 6 / 3 | Flip, Swap | WalkSat, WalkSat adaptive |
| TSP (TSPLIB) | 4 / 2 / 1 | TwoOpt, Relocate | LKH, ALNS, GRASP |
| JSSP (OR-Library) | 5 / 8 / 6 | Swap, Relocate | |
| VertexCover (GSET) | 5 / 4 / 4 | Flip, Swap | |
| GraphColoring (GSET) | 4 / 4 / 3 | Flip, Swap | |
| QAP (QAPLIB) | 3 / 3 / 2 | Swap | |
| FlowShop (Taillard) | 3 / 6 / 3 | Relocate, Swap | IG, ALNS, GRASP |
| BinPacking (Falkenauer) | 4 / 6 / 3 | Relocate, Swap | ALNS, GRASP |
| CVRP (CVRPLIB X) | 4 / 3 / 3 | Relocate, Swap, TwoOpt | HGS, ALNS, GRASP |

`Restart` and `PopulationAnnealing` check the clock only between whole inner
steps, so on the CVRP, the flow shop, graph coloring and dsj1000 a run can
overshoot its budget by as much as 60%.

## HGS against ALNS on CVRP

The CVRP rows of the sweep put [HGS](../heuristics/hgs.md)
(`HybridGeneticSearch`) and [ALNS](../heuristics/alns.md)
(`AdaptiveLargeNeighborhoodSearch`) side by side at the same budget. The gaps
below are to the CVRPLIB best-known solutions
(`data/instances/scripts/fetch_cvrp.sh`), over ten runs at the small and medium
bands and five at the large one.

| Instance | Budget | BKS | ALNS mean gap | HGS mean gap | ALNS best gap | HGS best gap |
|---|---|---|---|---|---|---|
| X-n101-k25 | 30 s | 27591 | +0.34% | +0.14% | +0.00% | +0.02% |
| X-n139-k10 | 30 s | 13590 | +0.14% | +0.11% | +0.00% | +0.00% |
| X-n176-k26 | 30 s | 47812 | +1.47% | +1.11% | +0.98% | +0.83% |
| X-n195-k51 | 30 s | 44225 | +1.91% | +0.64% | +1.55% | +0.46% |
| X-n214-k11 | 120 s | 10856 | +1.30% | +1.22% | +0.43% | +0.79% |
| X-n303-k21 | 120 s | 21736 | +1.23% | +1.66% | +0.63% | +1.32% |
| X-n401-k29 | 120 s | 66154 | +1.51% | +1.80% | +0.96% | +1.57% |
| X-n502-k39 | 600 s | 69226 | +0.25% | +0.76% | +0.19% | +0.64% |
| X-n701-k44 | 600 s | 81923 | +2.20% | +3.17% | +2.09% | +2.91% |
| X-n1001-k43 | 600 s | 72355 | +3.07% | +5.11% | +2.95% | +4.62% |

HGS is ahead on the small instances and ALNS from about 300 customers up, and
the lead of ALNS widens on the largest two. GRASP, the third solver in these
configs, ends 4% to 22% above the best-known solution.

Reproduce with `data/benchmarks/vrp/hgs_{small,medium,large}.toml`.

## Browse interactively

Published on GitHub Pages: https://trash-iine.github.io/optopus/ (Benchmark
viewer card).

Locally the viewer must be served over HTTP, it loads its data with `fetch`
from `data/index.json` + `data/**/*.slim.toml`, so opening the `file://` path
directly does not work:

```bash
cd docs && python3 -m http.server
# then open http://localhost:8000/benchmarks/viewer.html
```

## Profiles

Stop-condition, hyperparameter, and `num_runs` settings used for the
comprehensive (problem × heuristic) benchmark sweep. The intent is that every
TOML under `data/benchmarks/<problem>/` follows one of the size-band profiles
below, so results from different runs remain comparable.

`num_runs = 5` for every (problem, heuristic, instance) triple, and `seed = 42` on every config so per-run
seeds are derived deterministically. The small and medium bands of the 2026-10 sweep
ran ten runs, the count before this profile, and so did its BLS runs on the G-set
graphs of 1000 and 10000 vertices. That sweep ran the runs of one
(instance, heuristic) pair on ten threads through the small band and most of
the medium one, and on five threads from there on.

### Size bands per problem

| Problem | small | medium | large |
|---|---|---|---|
| MaxCut (GSET) | G1–G21, G43–G47, G51–G54 (n ≤ 1000) | G22–G42 (n = 2000), G48–G50 (n = 3000) | G55–G67, G70, G72, G77, G81 (n ≥ 5000) |
| QUBO (bqp) | bqp50, bqp100 (three of each) | bqp250, bqp500 (three of each) | bqp1000 (three) |
| SAT (uf*) | uf50, uf75 (three of each) | uf100, uf150 (three of each) | uf200 (three) |
| TSP (TSPLIB) | burma14, eil51, att48, berlin52 | eil101, ch150 | dsj1000 |
| JSSP (OR-Library) | ft06, ft10, ft20, la01, abz5 | la06, la11, la16, la19, la21, la24, abz7, orb01 | la26, la32, la36, swv01, swv11, yn1 |
| VertexCover | G1, G11, G14, G43, G51 | G22, G32, G35, G48 | G55, G58, G70, G81 |
| GraphColoring | G1, G11, G14, G43 | G22, G32, G35, G48 | G55, G62, G70 |
| QAP (QAPLIB) | nug20, tai20a, tai30a | sko42, tai40a, tai60a | sko100a, tai100a |
| FlowShop (Taillard) | ta001, ta011, ta021 | ta031, ta041, ta051, ta061, ta071, ta081 | ta091, ta101, ta111 |
| BinPacking (Falkenauer) | t60, u120 (two of each) | t120, t249, u250 (two of each) | t501, u500, u1000 (one of each) |
| CVRP (CVRPLIB X) | X-n101, X-n139, X-n176, X-n195 | X-n214, X-n303, X-n401 | X-n502, X-n701, X-n1001 |

## Pipeline & reproduce

- Run TOMLs under `data/benchmarks/<problem>/` generated by `data/benchmarks/scripts/gen_benchmark_configs.py` from the profiles above.
- Place output TOMLs to `docs/benchmarks/data/<problem>/`.
  These raw TOMLs hold the full per-run solution vectors and heavy, therefore gitignored.
- Run `render.py` (`pip install tomli-w` first), which aggregates the curated raw TOMLs
  into the viewer's data source, a slim copy of each with the heavy per-run
  `runs` arrays stripped plus an `index.json` the browser fetches to discover them. 
  The `*.slim.toml` + `index.json` are what you commit. 
  The viewer parses those slim TOMLs client-side with the vendored parser in
  `vendor/smol-toml.js`.
- Deploy GitHub Pages. `.github/workflows/deploy-pages.yml` deploys the committed
  `docs/` (the slim TOMLs + index) on every push to `main`.

```bash
cargo build --release

# Re-run one config; output lands in result/<timestamp>.toml
./target/release/optopus data/benchmarks/maxcut/general_small.toml
mv result/<timestamp>.toml docs/benchmarks/data/maxcut/general_gset_small.toml

# Regenerate viewer data (slim TOML + index)
python3 docs/benchmarks/render.py
```

A whole band takes from several hours (small) to about four days (large), so the
published sweep runs through `data/benchmarks/scripts/bench_units.py`. It
splits every config of a band into one unit per (instance, heuristic), runs
the units one after another, and skips a finished unit when started again, so
a run the machine stops loses at most the unit in progress. `merge` writes each
config's units back as the file a single run of that config would have
written. A unit holds its instance and its heuristic at position 0, so its
per-run seeds differ from those of a single run of the whole config, and the
units of one master seed all draw the same seeds. A run stopped by the clock
does not repeat bit for bit in any case. `freeze` copies the release binary
into the work directory and the sweep runs that copy, so rebuilding the
repository partway through does not change what the later units run.

```bash
cargo build --release
python3 data/benchmarks/scripts/bench_units.py freeze
python3 data/benchmarks/scripts/bench_units.py split small
caffeinate -dis python3 data/benchmarks/scripts/bench_units.py run small
python3 data/benchmarks/scripts/bench_units.py progress
python3 data/benchmarks/scripts/bench_units.py merge small
python3 docs/benchmarks/render.py
```

