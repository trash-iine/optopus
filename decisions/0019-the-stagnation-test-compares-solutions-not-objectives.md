# The BLS stagnation test compares solutions, not their objectives

- Status: adopted
- Area: max_cut, heuristic
- Date: 2026-08-08
- Code: src/heuristic/bls.rs (`AdaptivePerturbation::prev_local_optimum`)

## Decision

Benlic & Hao's `if C = Cp then L <- L+1 else L <- L0` is read as a comparison of
the two solutions, through `Distance::distance == 0`. The schedule therefore
keeps the previous local optimum itself rather than its objective value.

## Measurement

Comparing objectives instead is not an equivalent reading. On the G-set every
edge weighs ±1, so cut values are small integers and distinct local optima
collide on the same objective constantly.

On G11 with the paper's `l0 = 8`, the objective test fired on 82.7% of rounds.
That pushed the median `l` to 12 and its maximum to 80, a perturbation an order
of magnitude stronger than the paper asks for, applied to exactly the instances
with the widest plateaus.

## Do not retry

Do not store the objective to save a solution clone. The clone is one per round
that escapes its previous optimum, against a descent of many moves.

`PartialEq` on the solution is not the comparison to use either. `Distance` is
what a problem already defines for this, and it reads the assignment without the
caches beside it, which a derived `PartialEq` would compare too.
