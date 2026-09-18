# Ruin-and-recreate on routes needs the local repair, and needs it most on large instances

- Status: reference
- Area: vrp
- Date: 2026-09-19
- Code: src/heuristic/specific/vrp/mod.rs (`alns_for_vrp`)

## Decision

`AdaptiveLargeNeighborhoodSearch` takes its `LocalRepair` as an option, since a
problem with no anchored local search still runs plain ruin-and-recreate. On
CVRP the option is not one, so `alns_for_vrp` names the pair and the benchmark
registration builds through it.

## Measurement

Ten X instances at 30s by five runs, seed 42, the same config for both arms and
the descent switched off through a throwaway patch. The anchored descent wins
10 of 10, by 2.18% mean objective.

The mean hides the shape. Up to about 560 customers the difference is 0.01 to
1.17%, and on X-n344 it is 0.01%, which is nothing. From 700 up it is 4.43%,
6.05% and 8.17%.

Run-to-run spread moves the same way. On X-n801 the standard deviation is 201
with the descent and 1424 without, so the arm without it is both worse and far
less repeatable.

The reading is that greedy re-insertion does not repair the edges its own
choices spoil, and on a small instance the next ruin covers the damage while on
a large one it does not, the same anchors reaching a smaller share of the
routes.

## Do not retry

Do not drop the `LocalRepair` from the VRP registration for simplicity, and do
not quote this as a general result for `Ruinable`. It is measured on routes
only. A problem whose repair does not spoil its neighbours may well not need
one, which is why the trait keeps it optional.

Do not read 0012 as this measurement. That one compares an anchored descent
against a full sweep, and both of its arms have a descent.
