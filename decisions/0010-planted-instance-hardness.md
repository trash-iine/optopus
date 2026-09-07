# Where the planted suites are hard, and why published parameters do not transfer

- Status: reference
- Area: max_cut
- Date: 2026-08
- Code: examples/generate_hard_maxcut.rs

## Decision

`PlantedMaxCut` builds instances around a chosen solution, so the optimum is
exact by construction and a run's gap is measured against the answer rather than
against another heuristic's best result. The suite's parameters were fixed by a
sweep and are recorded in the generator.

## Measurement

For Wishart planting the hardness boundary sits at a constant kernel dimension
`n - M` of about 32, which is `alpha` of 0.35, 0.50, 0.65 and 0.90 at `n` of 48,
64, 96 and 256. Small `alpha` is the hard side at every size measured, not the
easy one an easy-hard-easy profile would suggest, because the criterion here is
reaching the exact optimum rather than the physics notion of a ground state.

`chook`'s default `alpha = 0.75` is easy at all four sizes.

Both families' peaks move with instance size.

## Do not retry

Do not carry published parameter values over without re-sweeping at the size you
intend to generate. The optima in `hard/manifest.toml` are for analysis only and
nothing in the library reads them, which is the same rule as best-known values.
