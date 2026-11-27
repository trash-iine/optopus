# Weighted local search follows the three modes of the integer programming local search over a constraint-weights trait, and leaves a feasible optimum by its best priced escape rather than a random one

- Status: adopted
- Area: heuristic
- Date: 2026-11-14, rebuilt on the paper's modes 2026-11-26
- Code: src/heuristic/weighted_local_search.rs, src/building_blocks/search/constraint_weights.rs, src/trait_defs/weighted.rs, src/problem/{sat,vertex_cover,graph_coloring}/weights.rs

## Decision

`WeightedLocalSearch<P: WeightedConstraints>` follows the local search for
integer programs of arXiv 2305.00188, read in the original. A step searches
for a first solution that violates nothing, betters the objective of one that
violates nothing by the move that keeps it so, or repairs toward one better
than the best. A move of a violated or satisfied constraint is a tight move,
priced by the weight of what it satisfies less what it violates, plus the
objective's weight if it leads below the best objective and less it if not.
The candidates are the tight moves of three violated constraints, then of
thirty satisfied ones, each sampled down to 2000 and 350 moves. At a weight
change, with probability `1 - smoothing` every violated constraint gains one
and the objective gains one if the solution is no better than the best,
otherwise the satisfied constraints and a better solution's objective lose one.
The step then takes the best tight move of one violated constraint, sampled to
150. The caps, the 0.0003, the tenure of 3 to 12 and the restart after
1,500,000 steps without a better best, from the best and a fresh solution with
even odds, are the paper's. Its partial reward for a constraint that moves
closer to satisfied has no case here, since every constraint of the three
problems is satisfied or not by one move.

One step differs. Where the paper takes a random unit step toward a better
objective from a feasible optimum, this takes the best priced of the problem's
objective escapes. The random step costs 8 colors on G1, a third color on the
bipartite graphs and 120 vertices on G48 (appendix).

The heuristic owns the run, the objective weight and the sampling.
`ConstraintWeights` in `building_blocks/search` owns the constraint weights,
the violated set and their raise and lower rule. The problem owns its prices,
kept with the solution by `apply_weighted`, and gives `weight_gain`,
`reweight`, `tight_moves`, `objective_escapes`, the objective apart from the
constraints and `restart_from`.

## Do not retry

Do not put the weights in the solution or in `SearchState`, which clones them
at every sub-run and ranks a solution that has to ignore part of itself. Do not
rebuild it as `Iterated` over `LocalSearch` and a weight step, since a descent
lasts a few moves and `Iterated` clones the state per phase. Do not draw the
objective escape at random without measuring graph coloring. Drawing every
violated constraint instead of three changes nothing measurable (appendix).

## Appendix

`data/benchmarks/graph_coloring/weighted_gset.toml`,
`data/benchmarks/vertex_cover/weighted_gset.toml` and
`data/benchmarks/sat/weighted_satcomp.toml` as committed, five runs, seed 42,
20 s per run on the two graph problems and 15 s on MaxSAT, Apple M4, the five
runs of one heuristic in parallel and nothing else started on the machine.
Average of five runs, parameters at their defaults and not tuned on these
instances. Every graph coloring and vertex cover run ended feasible. "Full
scan" is the earlier build, which descended on the cover size or the colors
used plus the violated weight over every move each step and left a feasible
optimum the same way. "Random escape" is the paper's unit step. The other
columns are the general heuristics, measured against the full scan.

Graph coloring, colors used. G11 and G32 are bipartite, so two is optimal.

| Graph | Adopted | Random escape | Full scan | TabuSearch Flip [3, 200] | SimulatedAnnealing |
|---|---|---|---|---|---|
| G1 | 14.0 | 22.4 | 14.0 | 68.0 | 64.6 |
| G11 | 2.0 | 3.0 | 2.0 | 5.0 | 5.0 |
| G14 | 6.4 | 13.4 | 6.8 | 128.6 | 57.8 |
| G43 | 8.0 | 12.0 | 7.8 | 37.0 | 37.0 |
| G22 | 8.0 | 12.0 | 8.0 | 38.0 | 38.0 |
| G32 | 2.0 | 3.0 | 2.0 | 5.0 | 5.0 |

Vertex cover, cover size.

| Graph | Adopted | Random escape | Full scan | PopulationAnnealing | TabuSearch Flip [3, 200] |
|---|---|---|---|---|---|
| G22 | 1575.4 | 1576.0 | 1573.4 | 1570.8 | 1663.8 |
| G32 | 1000.0 | 1075.2 | 1000.0 | 1001.0 | 1235.6 |
| G35 | 1288.6 | 1290.6 | 1282.0 | 1287.8 | 1403.4 |
| G48 | 1500.0 | 1619.6 | 1500.0 | 1517.4 | 1844.8 |
| G55 | 2841.6 | 2854.6 | 2817.8 | 3113.4 | 3130.4 |
| G70 | 3947.2 | 4098.0 | 3923.8 | 5117.2 | 4399.0 |

MaxSAT, satisfied clauses, larger is better. The escape does not arise, since
an assignment that violates nothing ends the run.

| Formula | Adopted | Full scan | WalkSat 0.3 | WalkSat adaptive |
|---|---|---|---|---|
| coloring_4g_5color_170 | 2462407.6 | 2462708.8 | 2462705.0 | 2462687.8 |
| hamiltonian_hcp_470_105 | 253835.6 | 253794.4 | 253556.2 | 253349.0 |
| mycielski10_hints6 | 745706.0 | 745706.0 | 745703.8 | 745702.8 |
| quasigroup_ukn006 | 18472.0 | 18472.0 | 18470.0 | 18460.2 |
| sliding_puzzle57 | 163493.0 | 163438.8 | 163439.0 | 163438.0 |
| station_repacking_49 | 892019.8 | 892033.0 | 892015.0 | 892008.0 |
| waerden_2-3-25_606 | 49087.8 | 49092.8 | 49094.6 | 49094.2 |
| uf200-01 | 860.0 | 860.0 | 860.0 | 860.0 |
| uf200-010 | 860.0 | 860.0 | 860.0 | 860.0 |

The adopted build makes 2 to 12 times the steps of the full scan, 2.3e7 on
vertex cover G55 against 1.6e6 and 2e7 on sliding_puzzle57 against 4.3e5.
The restart did not fire within these runs, the same three vertex cover graphs
at 10 s giving the same covers with it removed.

Drawing every violated constraint instead of three, three runs of 10 s, gave
22.0 colors on G1 against 22.7 and 1615.3 on vertex cover G48 against 1596.0,
inside the spread of the random escape it was measured on.

The weighted flip of `Sat` keeps the solution's gains and the weighted gains
in one walk of each clause, through `Sat::clause_shares`. Against walking the
clauses for each separately, 2e6 steps on station_repacking_49 take 32% fewer
instructions and 17% fewer cycles, and the trajectory is the same. The plain
flip, `add_clause_shares` over the same walk, runs the instructions of main
to within 0.01% on a 2e6-step random walk.
