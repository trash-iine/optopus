# Integer problems have two layers, the framework keeps no gain cache, and FormulaProblem sits on the same layer

- Status: adopted
- Area: integer
- Date: 2026-09-26
- Code: src/problem/integer/

## Decision

`IntegerProblem` is a struct built from the ranges and an objective closure,
with optional `with_delta` / `with_swap_delta` / `with_reverse_delta` closures.
It implements `IntAssignment`, reading and writing one value of a solution of
the problem's own type, and the three moves are written once against that trait.
A problem that needs a cache to price moves keeps it in its own solution. The
framework offers no switchable gain cache and no term model. A delta not given
is the type `NoDelta`, not an `Option`, because the `Option` check cost 11 to
24% on Tabu and 2-opt scans.

`FormulaProblem` takes `IntVars` and implements `IntAssignment` too, so its
moves are the same three, and `src/problem/binary_optimization/` is deleted.
Monomials keep powers, since `x^2 = x` holds only on binary variables. Its
solution keeps a delta per (variable, value), which on binary variables is the
old per-variable gain. `maximize` / `minimize` constructors replace
`OptDirection`. `IntCrossover` and `Distance` bring `GeneticAlgorithm` to both.
`SubProblemExtractable` stays on `FormulaProblem` only, because substituting
into an expression yields the same type and a closure does not.

## Do not retry

- A gain cache switched on the problem. Whether it pays depends on the search
  (LS gains, SA loses), and a missing `dependents` entry leaves stale gains
  with no error.
- The term model as the main entry, slower than one hand-written delta and
  unable to express makespan or a cheap 2-opt (an `Expr` DSL may still use it).
- Any scan of `IntChangeNeighbor` but the flat `Changes` walk, which halved
  the framework overhead on binary variables.
- `powi` in the pricing path. It became a `__powidf2` call even for power one.
- Swapping or reversing values between variables of different ranges without
  checking both fit.

## Appendix

`IntegerProblem` against the native move, per-iteration time, median of 5, same
seed and start, fat LTO, every variant reaching the same best. MaxCut G1 / G22 /
G43, LocalSearch.
- A hand-written O(degree) `delta`, x28 / x12 / x11.
- The same with a framework gain table (`CACHE_GAINS` plus a `dependents`
  list), x5.5 / x4.2 / x4.4, but SA at T0=20 went from x2.5 to x12 to x30,
  every accept recomputing O(degree²).
- `IntAssignment` with the cache in the user's solution, x1.07 / x1.10 / x1.10,
  SA at T0=20 x1.4 to x1.6.
- Term model, x78 / x31 / x28 uncached, x7.0 / x5.2 / x5.1 cached.

TSP eil101 / ch150 / dsj1000 with `IntReverseNeighbor` against
`TspTwoOptNeighbor`, a four-lookup `reverse_delta` x0.8 to x1.3 across LS,
Tabu and SA, objective only x8 to x127, term model x13 to x196.

Against the old trait, built aligned (0002), five of six LS/Tabu cases on G1 /
G22 / G43 and dsj1000 were within 1.02. G43 LocalSearch stayed 7 to 13% slower
in both builds, not attributed.

`FormulaProblem` old against new, same start and seed, median of 3,
interleaved. MaxCut G43 and G22 written as `w (x_i + x_j - 2 x_i x_j)`, and a
random 500-variable quadratic with one capacity constraint. LS 2k, Tabu 20k and
SA 1M iterations. Best value, best iteration and the solution's hash were
identical in all nine cases. New over old per iteration, G22 / G43 LS 1.10 /
1.13, SA 1.19 / 1.19, Tabu 1.76 / 1.53, the same aligned. Constrained
quadratic LS 0.59, SA 0.56, Tabu 0.60. The unconstrained gap is the generic
move reading the variable's value to build its target, where the old flip read
one gain. The constrained gain is not attributed. Storing a move's cost as a
minimized `f64` rather than an `Evaluable` took LocalSearch from 1.5 to 1.1, and
equal ranges locate a change by arithmetic rather than by the cumulative table.
