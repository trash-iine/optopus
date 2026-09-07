# The learned controller gave up two plateau operators that were paying

- Status: adopted
- Area: max_cut
- Date: 2026-08-08
- Code: examples/rl_bls.rs

## Decision

`examples/rl_bls.rs` drives Breakout Local Search with a contextual bandit over
perturbation type and strength. Two objective-preserving plateau operators, one
flipping connected clusters and one flipping an independent set of zero-gain
vertices, used to be a fourth and fifth action, together with a plateau-width
context feature. They were removed.

The trade was a smaller action space, one operator vocabulary and no second
scratch structure, in exchange for objective. It was made knowing the cost.

## Measurement

At 30s by five runs the removal costs 96.2 and 62.6 on G55 across two seeds,
110.4 on G60 and 92.2 on G63, against standard deviations of 9 to 28. It is
neutral on G70, G11 and G1. With the operators the controller beat plain BLS on
G55, 10200.4 against 10168.0. Without them it does not.

## Do not retry

Take them back if this controller becomes the thing that has to win. The
mechanism used to survive in `PopulationAnnealing`, which owned its own
implementation of the cluster move; that is gone too, see 0016.
