# LinKernighanHelsgottForTsp

Problem-specific heuristic for [TSP](../problems/tsp.md). Performs a
variable-depth edge-exchange search (up to *k*-opt) starting from each city.

## Algorithm sketch

For each starting city, the algorithm extends a chain of edge swaps:

1. Pick a candidate city near the current chain endpoint.
2. Try to close the move; if the resulting tour is shorter, apply it.
3. Otherwise extend the chain to deeper levels.

Pruning:

- **Candidate lists** — only the `num_neighbors` nearest cities at each
  endpoint are considered.
- **Positive gain criterion** — partial gain must remain positive at every
  step.
- **Maximum depth** — the search stops after `max_depth` levels (k in k-opt).

The first improving move found is applied; the search terminates when no
improving move exists for any starting city, or when the stop condition
fires.

## Constructor

```rust
LinKernighanHelsgottForTsp::new(
    stop_condition: StopCondition,
    num_neighbors: usize,
    max_depth: usize,
) -> Self
```

Reasonable defaults: `num_neighbors = 5`, `max_depth = 5`.

## References

- Lin, S. and Kernighan, B. W. "An Effective Heuristic Algorithm for the
  Traveling-Salesman Problem." *Operations Research*, 21(2), 498-516, 1973.
- Helsgaun, K. "An Effective Implementation of the Lin-Kernighan Traveling
  Salesman Heuristic." *European Journal of Operational Research*, 126(1),
  106-130, 2000.
