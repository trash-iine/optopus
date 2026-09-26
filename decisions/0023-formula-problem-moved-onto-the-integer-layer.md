# FormulaProblem moved onto the integer layer, and the binary module is gone

- Status: adopted
- Area: integer
- Date: 2026-09-26
- Code: src/problem/integer/formula.rs, src/problem/integer/crossover.rs

## Decision
`FormulaProblem` now takes `IntVars` and implements `IntAssignment`, so its moves are `IntChangeNeighbor`,
`IntSwapNeighbor` and `IntReverseNeighbor`, and `src/problem/binary_optimization/` is deleted. Monomials keep powers,
since `x^2 = x` holds only on binary variables. The solution keeps a delta per (variable, value), which on binary
variables is the old per-variable gain. `OptDirection` gave way to `maximize` / `minimize` constructors.
`IntCrossover` and `Distance` bring `GeneticAlgorithm` to both integer problems. `SubProblemExtractable` stays on
`FormulaProblem` only, because substituting into an expression yields the same type and a closure does not.

## Measurement
Old against new from the same start and seed, fat LTO, median of 3, runs interleaved. Binary formulas: MaxCut G43 and
G22 written as a sum of `w (x_i + x_j - 2 x_i x_j)`, and a random 500-variable quadratic with one capacity constraint.
LS 2k, Tabu 20k and SA 1M iterations. Best value, best iteration and the best solution's hash were identical in all
nine cases, SA included, so the arithmetic was kept operand for operand. New over old, per iteration:
- MaxCut G22 / G43: LS 1.10 / 1.13, SA 1.19 / 1.19, Tabu 1.76 / 1.53. The same with `-align-all-functions=6`.
- Constrained quadratic: LS 0.59, SA 0.56, Tabu 0.60.
The unconstrained gap is the generic move reading the variable's value to build its target, where the old flip read
one gain. The constrained gain is not attributed yet.

Three fixes were needed on the way, each measured. `f64::powi` folded into a `__powidf2` call even for power one,
which was most of an apply. A move's cost is stored as a minimized `f64` rather than an `Evaluable`, which took
LocalSearch from 1.5 to 1.1. Equal ranges locate a change by arithmetic rather than by the cumulative table.

## Do not retry
- `powi` in the pricing path. Repeated multiplication is what keeps a power of one free.
- Swapping or reversing values between variables of different ranges without checking both fit. It wrote
  out-of-range values until the integer tests caught it.
