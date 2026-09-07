# The broken-pairs count is asymmetric at the depot

- Status: adopted
- Area: vrp
- Date: 2026-08
- Code: src/problem/vrp/adjacency.rs

## Decision

`Distance for VrpSolution` counts broken pairs of `RouteAdjacency` and
symmetrizes by taking the larger direction. It does not delegate to one
direction, because the count is not symmetric.

Splitting a customer out of a route costs it a depot departure it did not have,
so the split solution counts a break the joined one does not count back.

HGS is the exception. It ranks diversity on the directional count, because that
is what Vidal defines biased fitness on.

## Measurement

Found by exhaustive search over every partition of three and four customers,
which is also how the claim that the count is otherwise symmetric was rejected.
`the_directional_count_is_asymmetric_at_the_depot` pins the smallest case.

## Do not retry

Do not simplify `Distance` to a single direction. The test above will catch it,
but the reason is easier to read here.
