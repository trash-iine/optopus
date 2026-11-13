# Weighted local search is one heuristic over a constraint-weights trait, and it pays on vertex cover and MaxSAT while scanning every move

- Status: adopted
- Area: heuristic
- Date: 2026-11-13
- Code: src/heuristic/weighted_local_search.rs, src/trait_defs/weighted.rs, src/problem/sat/weights.rs, src/problem/vertex_cover/weights.rs

## Decision

`WeightedLocalSearch<P: WeightedConstraints>` descends on the objective plus
the weights of the violated constraints. At a weighted local optimum it adds
one to every violated constraint with probability `1 - smoothing`, otherwise
takes one from every satisfied constraint above one, then leaves by the best
escape move. The rule, the `smoothing` of 0.0003, the cap of
`max(1000, constraints)` and the tenure of 3 to 12 are those of the integer
programming local search of arXiv 2305.00188, read in the original.

Unlike that source the objective has no weight of its own, and a local optimum
that violates nothing leaves by the best objective-improving move under the
weights rather than by one drawn at random.

The weights and the prices under them live in the problem's `Weights`, owned by
the heuristic. `apply_weighted` changes the solution and the prices in one
call, so no caller can get them apart.

It scans every move each step, as `TabuSearch` does, and still gives the
smallest cover on five of six G-set graphs and beats WalkSat on six of seven
large formulas in a tenth of the steps or fewer.

## Do not retry

Do not put the weights in the solution or in `SearchState`. A solution carrying
them is cloned with them at every sub-run and its rank has to ignore part of
itself.

Do not rebuild it as `Iterated` over `LocalSearch` and a weight step. A
weighted descent lasts a few moves, and `Iterated` clones the state per phase.
Do not narrow the scan to the moves of violated constraints without measuring
vertex cover, where leaving a cover is a move of no violated constraint.
Graph coloring waits for the shared integer assignment moves replacing its own.

## Appendix

`data/benchmarks/vertex_cover/weighted_gset.toml` and
`data/benchmarks/sat/weighted_satcomp.toml` as committed, five runs, seed 42,
20 s per run on vertex cover and 15 s on MaxSAT, Apple M4, the five runs of one
heuristic in parallel and nothing else started on the machine. Average of five
runs. Every vertex cover run ended feasible.

Vertex cover, cover size, smaller is better.

| Graph | WeightedLocalSearch | TabuSearch Flip [3, 200] | SimulatedAnnealing | PopulationAnnealing | Restart of TabuSearch |
|---|---|---|---|---|---|
| G22 | 1571.8 | 1663.8 | 1674.2 | 1570.8 | 1651.2 |
| G32 | 1000.0 | 1235.6 | 1231.6 | 1001.0 | 1207.4 |
| G35 | 1282.0 | 1403.4 | 1395.0 | 1287.8 | 1390.4 |
| G48 | 1500.0 | 1844.8 | 1883.2 | 1518.8 | 1827.6 |
| G55 | 2818.8 | 3130.4 | 3161.6 | 3103.8 | 3103.8 |
| G70 | 3924.6 | 4399.0 | 4479.4 | 4978.8 | 4381.8 |

MaxSAT, satisfied clauses, larger is better.

| Formula | WeightedLocalSearch | WalkSat 0.3 | WalkSat adaptive | TabuSearch Flip [5, 20] |
|---|---|---|---|---|
| coloring_4g_5color_170 | 2462743.4 | 2462715.0 | 2462689.2 | 2460484.4 |
| hamiltonian_hcp_470_105 | 253790.4 | 253558.0 | 253349.0 | 251703.8 |
| mycielski10_hints6 | 745706.0 | 745703.8 | 745702.8 | 745593.8 |
| quasigroup_ukn006 | 18472.0 | 18470.0 | 18460.2 | 18424.0 |
| sliding_puzzle57 | 163447.6 | 163439.0 | 163438.0 | 163029.0 |
| station_repacking_49 | 892033.0 | 892015.0 | 892008.0 | 891941.8 |
| waerden_2-3-25_606 | 49093.0 | 49094.6 | 49094.2 | 49096.8 |
| uf200-01 | 860.0 | 860.0 | 860.0 | 858.2 |
| uf200-010 | 860.0 | 860.0 | 860.0 | 859.0 |

WeightedLocalSearch made between 1.5e5 and 1.5e6 moves per run on the
competition formulas, WalkSat between 1.5e6 and 9.5e7. On the two satisfiable
uf200 formulas it stopped at the satisfying assignment after about 0.03 s.
The parameters were the defaults, not tuned on these instances.
