# A test for returning to the same solution compares solutions through Distance, not their objectives

- Status: adopted
- Area: heuristic, max_cut
- Date: 2026-08-08
- Code: src/heuristic/bls.rs (`AdaptivePerturbation::prev_local_optimum`)

## Decision

Where a heuristic asks whether the search came back to a solution it had
before, it compares the two solutions through `Distance::distance == 0`, and
keeps the earlier solution itself rather than its objective value. Two distinct
solutions with the same objective are not the same solution, and on an
instance with wide plateaus they often share one, so the objective test
fires on rounds that did move.

Breakout Local Search is where this is decided. Benlic and Hao's
`if C = Cp then L <- L+1 else L <- L0` is read this way, which is why
`AdaptivePerturbation` keeps the previous local optimum. The evidence is the
G-set only, where every edge weighs ±1 and cut values are small integers.

## Do not retry

Do not store the objective to save a solution clone. In BLS the clone is one
per round that escapes its previous optimum, against a descent of many moves.

`PartialEq` on the solution is not the comparison to use either. `Distance` is
what a problem already defines for this, and it reads the assignment without
the caches beside it, which a derived `PartialEq` would compare too.

## Appendix

On G11 with the paper's `l0 = 8`, the objective test fired on 82.7% of rounds.
That pushed the median `l` to 12 and its maximum to 80, a perturbation an order
of magnitude stronger than the paper asks for, applied to exactly the instances
with the widest plateaus.
