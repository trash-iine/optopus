# A relaxation solved by steps certifies its bound before it prunes and floors the incumbent it aims at

- Status: adopted
- Area: heuristic, max_cut
- Date: 2026-10-01
- Code: src/trait_defs/branch.rs (`Relaxation::bound_against`), src/heuristic/branch_and_bound.rs, src/problem/max_cut/eigen.rs

## Decision

`BranchAndBound` searches the root before bounding anything, so the first bounds
see a searched incumbent, and hands that incumbent to the relaxation through
`Relaxation::bound_against`. A relaxation solved by steps, such as a subgradient
bound, uses it in three ways that hold whatever the problem.

- It prunes only on a certified bound. An iterative estimate can sit on the
  wrong side of the true value, and pruning on it can cut the optimum.
- It aims its steps at the incumbent as the Polyak target, with a floor below
  its own best estimate. Without the floor, a target far below the bound makes
  the steps overshoot.
- It stops once the estimate falls to the incumbent, and not earlier on an
  extrapolation of its recent descent. The bound of a node that is not pruned
  orders the search and is reported, so leaving it loose costs both.

The evidence is MaxCut's `EigenvalueRelaxation` only. It bounds a node by
Delorme and Poljak's `(n/4) λ_max(L + diag(u)) − ¼ Σu` on the folded graph,
improves `u` by subgradient steps, certifies `λ_max` by Rump's floating-point
Cholesky test (BIT 2006, Corollary 2.4, bound II), and floors the target at 5%
below its best estimate. Against the trivial bound of positive weights it takes
the median G-set gap from 34% to 6%.

## Do not retry

- Stopping the steps on an extrapolated descent. On MaxCut it left G11's root
  bound at 640 against 634. BiqBin extrapolates only its cutting-plane phase,
  after the basic bound is solved.
- The incumbent as the target without a floor. On G6 it gave 2725 against
  2679 at the root, and 2658 with the floor.
- Pruning on an uncertified eigenvalue. A Lanczos value is never above the
  true one, so the bound built on it can fall below the valid bound.

## Appendix

G1 to G54, 60 s each, 8 in parallel, BLS of 100k iterations per node, gap =
(upper − lower) / upper.

- Median gap 34.0% with the positive weights, 6.1% with the eigenvalue bound.
  By group, G1-5 39.3 to 4.3, G6-10 (±1) 78.5 to 19.5, G22-26 33.4 to 6.1,
  G39-42 58.3 to 20.5. G48 and G49 are proven by both, G50 stays at 2%.
- The positive weights hardly move with depth. With 25 to 90 times the nodes
  (1k BLS iterations per node) G1 went from 19145 to 19102 only.
- Certified root bound in 0.15 to 0.25 s at n = 800 to 1000, about 1.1 s at
  2000, 3.3 s at 3000.
- The lower bound loses a little to the time the bounds take, 14 of 54 at the
  reference against 17 before, and up to −47 on G39-42.
- Random graphs (p = 0.3) are proven up to 64 vertices in 19 s, where the
  positive weights take 4.3 s at 32.
