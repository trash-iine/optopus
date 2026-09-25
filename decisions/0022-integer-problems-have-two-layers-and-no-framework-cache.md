# Integer problems have two layers, and the framework keeps no gain cache

- Status: adopted
- Area: integer
- Date: 2026-09-26
- Code: src/problem/integer/

## Decision
`IntegerProblem` (ranges plus objective, optional `delta` / `swap_delta` / `reverse_delta`) sits on `IntAssignment`
(read and write one value of a solution of the problem's own type). The three moves are written once against the lower
layer. A problem that needs a cache to price moves keeps it in its own solution. The framework offers no switchable
gain cache and no term model.

## Measurement
Per-iteration time against the native move, median of 5, same seed and start, fat LTO. Every variant reached the
same best as the others. MaxCut G1 / G22 / G43, LocalSearch.
- `IntegerProblem` with a hand-written O(degree) `delta`: x28 / x12 / x11.
- The same with a framework gain table (`CACHE_GAINS` plus a `dependents` list): x5.5 / x4.2 / x4.4. But SA at
  T0=20 went from x2.5 to x12 to x30, since every accept recomputes O(degree^2).
- `IntAssignment` with the gain cache in the user's own solution: x1.07 / x1.10 / x1.10. SA at T0=20 was x1.4 to x1.6.
- Term model (objective as a sum of terms, deltas and cache derived): x78 / x31 / x28 uncached, x7.0 / x5.2 / x5.1
  cached. The cost is the indirection from term index to term data.

TSP eil101 / ch150 / dsj1000 as a permutation with `IntReverseNeighbor`, against `TspTwoOptNeighbor`.
- `IntegerProblem` with a four-lookup `reverse_delta`: x0.8 to x1.3 across LS, Tabu and SA.
- Objective only: x8 to x127, growing with n. Term model: x13 to x196, since a reversal re-evaluates every term inside.

## Do not retry
- A gain cache switched on the problem. Whether it pays depends on the search (LS gains, SA loses), and a missing
  `dependents` entry leaves stale gains with no error.
- The term model as the main entry. It is slower than one hand-written delta, and it cannot express makespan or a
  cheap 2-opt. It may still serve as the internal form of an `Expr` DSL over integers.
- Replacing the `flat_map` scan of `IntChangeNeighbor` with anything but the flat `Changes` walk. The walk halved the
  framework overhead on binary variables.
