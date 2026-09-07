# HGS no longer beats ALNS at equal budget

- Status: reference
- Area: vrp
- Date: 2026-08
- Code: src/heuristic/specific/vrp/

## Decision

`HybridGeneticSearchForVrp` used to beat `AdaptiveLargeNeighborhoodSearchForVrp`
by 0.6 to 2.5 percentage points at equal budget. It no longer does, so neither
is the default recommendation for CVRP.

## Measurement

At 30s the two are level. ALNS is ahead on the largest instances and HGS on the
mid-sized ones, with every difference under 0.7%.

The 600s band has not been re-measured since.

## Do not retry

Do not quote the old 0.6 to 2.5 point advantage. If a comparison matters, measure
the band you care about, and note that the 600s numbers are stale rather than
absent.
