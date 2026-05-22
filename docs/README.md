# Optopus Documentation

Start here, then dive into the section that matches your task.

## Getting started

- [Quickstart](quickstart.md) — minimal end-to-end example, file loaders.
- [Concepts](concepts.md) — design philosophy, the three use cases, and key patterns.
- [SearchState API](search_state.md) — fields, methods, sub-run clone variants.
- [Core traits](traits.md) — minimum traits + per-heuristic extras.

## Reference

### Problems

- [Overview](problems/README.md)
- [MaxCut](problems/max_cut.md)
- [QUBO](problems/qubo.md)
- [MaxSAT](problems/sat.md)
- [TSP](problems/tsp.md)
- [Vertex Cover](problems/vertex_cover.md)
- [Job Shop Scheduling](problems/job_shop_scheduling.md)
- [Formula](problems/formula.md)

### Heuristics

- [Overview](heuristics/README.md)
- [Local Search](heuristics/local_search.md)
- [Simulated Annealing](heuristics/simulated_annealing.md) (incl. Bang-Bang variant)
- [Late Acceptance Hill Climbing](heuristics/late_acceptance.md)
- [Tabu Search](heuristics/tabu_search.md)
- [Random Walk](heuristics/random_walk.md)
- [Beam Search](heuristics/beam_search.md)
- [RL Search](heuristics/rl_search.md)
- [Genetic Algorithm](heuristics/genetic_algorithm.md) (incl. `Crossover` trait)
- [Meta-heuristics](heuristics/meta.md) — Sequential, Iterated (ILS), Restart
- [Breakout Local Search (MaxCut)](heuristics/breakout_local_search.md)
- [Lin-Kernighan-Helsgaun (TSP)](heuristics/lkh.md)

## Guides

- [Stop conditions](guide/stop_conditions.md)
- [Composing heuristics](guide/composing.md)
- [Benchmarking](guide/benchmarking.md) — TOML schema and CLI
- [Error handling](guide/error_handling.md)
- [Defining a custom problem](guide/custom_problem.md)
- [Defining a custom heuristic](guide/custom_heuristic.md)

## Benchmarks

- [Reports](benchmarks/) — per-heuristic results on standard instance sets.
