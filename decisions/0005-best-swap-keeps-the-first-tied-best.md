# Ties keep the lowest vertex index, and sampling them uniformly was rejected

- Status: adopted
- Area: max_cut
- Date: 2026-08
- Code: src/heuristic/specific/max_cut/best_swap.rs

## Decision

The scan in `best_swap` keeps the first candidate it met among equals, which on
the G-set means the lowest vertex index. The comparisons are `>` and not `>=`
for that reason.

Ties are common rather than exotic. Every G-set weight is plus or minus one, so
gains are small integers, and the degree-4 toroidal instances admit only five
distinct values, putting hundreds of vertices in a single tie.

## Measurement

Sampling the tie uniformly lost 6 cut points over G11, G12, G13 and G32 to G34,
and turned three exact matches of the paper's best into misses, while gaining on
one planar instance only.

Index order on a toroidal grid tracks position, so taking the lowest index walks
the lattice coherently. Randomising it scatters the perturbation instead.

## Do not retry

Flattening the comparisons to `>=` is not a tidy-up. It changes which vertex the
scan selects among equals.
