# Optopus Documentation

A metaheuristic optimization library for combinatorial problems, written in Rust.

- **[API reference (rustdoc)](api/optopus/index.html)** — full signatures and
  doc comments for every public type, trait and method.
- **[Benchmark viewer](benchmarks/viewer.html)** — filterable, sortable
  cross-heuristic comparison across every supported problem type.

## Getting started

- [Quickstart](quickstart.md) — minimal end-to-end example, file loaders.
- [Concepts](concepts.md) — design philosophy, the three use cases, and key patterns.
- [SearchState](search_state.md) — what the state holds, how a heuristic advances
  it, sub-run clone variants, and crossing a reduction.
- [Core traits](traits.md) — minimum traits + per-heuristic extras.

## Guides

- [Stop conditions](guide/stop_conditions.md)
- [Benchmarking](guide/benchmarking.md) — TOML schema and CLI
- [Error handling](guide/error_handling.md)
- [Defining a custom problem](guide/custom_problem.md)
- [Defining a custom heuristic](guide/custom_heuristic.md)

## Reference

### Problems

- [Overview](problems/README.md)
- [MaxCut](problems/max_cut.md), and its exact [kernelization](problems/max_cut_kernel.md)
- [QUBO](problems/qubo.md)
- [MaxSAT](problems/sat.md)
- [TSP](problems/tsp.md)
- [Vertex Cover](problems/vertex_cover.md)
- [Job Shop Scheduling](problems/job_shop_scheduling.md)
- [CVRP](problems/vrp.md)
- [Formula](problems/formula.md)

### Heuristics

- [Overview](heuristics/README.md)
- [Local Search](heuristics/local_search.md)
- [Simulated Annealing](heuristics/simulated_annealing.md) (including Bang-Bang variant)
- [Late Acceptance Hill Climbing](heuristics/late_acceptance.md)
- [Tabu Search](heuristics/tabu_search.md)
- [Random Walk](heuristics/random_walk.md)
- [Beam Search](heuristics/beam_search.md)
- [RL Search](heuristics/rl_search.md)
- [Genetic Algorithm](heuristics/genetic_algorithm.md) (including `Crossover` trait)
- [Meta-heuristics](heuristics/meta.md) — Sequential, Iterated (ILS), VNS, Restart
- [Breakout Local Search (MaxCut)](heuristics/breakout_local_search.md)
- [RL Breakout Local Search (MaxCut)](heuristics/rl_breakout_local_search.md)
- [Population Annealing (MaxCut)](heuristics/population_annealing.md)
- [Lin-Kernighan-Helsgaun (TSP)](heuristics/lkh.md)
- [WalkSAT (MaxSAT)](heuristics/walksat.md)
- [Hybrid Genetic Search (CVRP)](heuristics/hgs.md)
- [Adaptive Large Neighborhood Search (CVRP)](heuristics/alns.md)

