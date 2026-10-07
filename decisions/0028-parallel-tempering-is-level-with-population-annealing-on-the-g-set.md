# Parallel tempering shares the Metropolis sweep with population annealing, and is level with it on the G-set

- Status: reference
- Area: heuristic, max_cut
- Date: 2026-10-07
- Code: src/heuristic/parallel_tempering.rs, src/building_blocks/search/metropolis.rs

## Decision

`ParallelTempering<P, N>` was added with the same requirement as population
annealing, `Evaluate` on the solution and on the move, so it runs on every
problem. The Metropolis sweep that population annealing kept as a private
function moved to `building_blocks::search::metropolis` with `boltzmann_accept`,
which it calls. `heuristic::boltzmann_accept` stays as a re-export, so no public
path changed. Population annealing's pinned trajectories in
`tests/pa_trajectory.rs` are unchanged.

## Measurement

G1, G11, G14 (n = 800) and G22, G32, G35 (n = 2000), 20 s by five runs, seed 42,
`data/benchmarks/maxcut/replica_annealers.toml`. Parallel tempering with 20
replicas on β from 0.1 to 5, one sweep per exchange. Population annealing at
its config defaults with 20 replicas. Simulated annealing from T = 3 cooling by
0.9999995 per move. Mean cut (best).

| Instance | Parallel tempering | Population annealing | Simulated annealing |
|---|---|---|---|
| G1 | 11624.0 (11624) | 11624.0 (11624) | 11608.6 (11624) |
| G11 | 564.0 (564) | 564.0 (564) | 560.8 (564) |
| G14 | 3062.6 (3063) | 3062.2 (3063) | 3055.4 (3059) |
| G22 | 13358.2 (13359) | 13358.0 (13358) | 13328.2 (13355) |
| G32 | 1406.8 (1408) | 1408.4 (1410) | 1395.2 (1398) |
| G35 | 7672.8 (7675) | 7682.6 (7684) | 7648.6 (7652) |

The two replica methods are level at n = 800 and population annealing is ahead
on the two sparse 2000-vertex graphs, by 1.6 and 9.8. Both are clearly ahead of
simulated annealing everywhere. That is the finding of Wang, Machta and
Katzgraber on spin glasses, population annealing comparable to parallel
tempering and better than simulated annealing. The ladder here was not tuned.

## Do not retry

Do not read the G32 and G35 rows as parallel tempering being weaker before
its ladder has been tuned for 2000 vertices, more replicas being the first
thing to try.

Do not move the sweep back into either heuristic. It is the one piece the two
share, and the isoenergetic cluster move 0016 points to would be the next
component both take from outside.
