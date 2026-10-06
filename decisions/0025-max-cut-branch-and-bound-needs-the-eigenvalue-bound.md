# Branch-and-bound on MaxCut needs the eigenvalue bound, and spends the incumbent on its steps

- Status: adopted
- Area: max_cut, heuristic
- Date: 2026-10-01
- Code: src/problem/max_cut/eigen.rs, src/heuristic/branch_and_bound.rs

## Decision
`EigenvalueRelaxation` bounds a MaxCut node by Delorme and Poljak's `(n/4) λ_max(L + diag(u)) − ¼ Σu` on the folded
graph, `u` improved by subgradient steps, and certifies `λ_max` by Rump's floating-point Cholesky test (BIT 2006,
Corollary 2.4, bound II). The incumbent reaches the relaxation through `Relaxation::bound_against`. It is the Polyak
target, floored at 5% below the best estimate, and the steps stop once the estimate falls to it. `BranchAndBound`
searches the root before bounding anything, so the first bounds see a searched incumbent.

## Measurement
G1 to G54, 60 s each, 8 in parallel, BLS of 100k iterations per node, gap = (upper − lower) / upper.
- Median gap 34.0% with the positive weights, 6.1% with the eigenvalue bound. By group, G1-5 39.3 to 4.3, G6-10 (±1)
  78.5 to 19.5, G22-26 33.4 to 6.1, G39-42 58.3 to 20.5. G48 and G49 are proven by both, G50 stays at 2%.
- The positive weights hardly move with depth. With 25 to 90 times the nodes (1k BLS iterations per node) G1 went
  from 19145 to 19102 only.
- Certified root bound in 0.15 to 0.25 s at n = 800 to 1000, about 1.1 s at 2000, 3.3 s at 3000.
- The lower bound loses a little to the time the bounds take: 14 of 54 at the reference against 17 before, and up to
  −47 on G39-42.
- Random graphs (p = 0.3) are proven up to 64 vertices in 19 s, where the positive weights take 4.3 s at 32.

## Do not retry
- Stopping the steps when the recent descent, extrapolated, would not reach the incumbent. It left the bound of nodes
  that are not pruned loose (G11 640 against 634 at the root), and that bound orders the nodes and is reported.
  BiqBin extrapolates only the cutting-plane phase, after the basic bound is solved.
- The incumbent as the Polyak target without a floor. Far below the bound it overshoots, G6 2725 against 2679 at the
  root and 2658 with the floor.
- Pruning on an uncertified eigenvalue. A Lanczos value is never above the true one.
