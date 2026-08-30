# LinKernighanHelsgaunForTsp

**API:** [`LinKernighanHelsgaunForTsp`](../api/optopus/heuristic/struct.LinKernighanHelsgaunForTsp.html)

Problem-specific heuristic for [TSP](../problems/tsp.md). Performs a
variable-depth edge-exchange search (up to *k*-opt) starting from each city.

## Example

```rust
use optopus::prelude::*;

let tsp = TspWithCoordinates::new(
    "demo".to_string(),
    vec![
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.5),
        (2.0, 1.5),
        (1.0, 2.0),
        (0.0, 1.0),
    ],
);
let mut state = SearchState::new(&tsp);

let mut lkh = LinKernighanHelsgaunForTsp::new(
    StopCondition::iterations(10_000),
    /* num_neighbors = */ 5,
    /* max_depth     = */ 5,
);
lkh.run(&mut state)?;

let sol = &state.best_solution;
println!("tour length = {}", sol.objective);
println!("visiting order = {:?}", sol.tour);
# Ok::<(), optopus::error::OptError>(())
```

Like `LocalSearch`, it stops at a local optimum, so a budget larger than one
descent only pays inside [`Restart` or `Iterated`](meta.md).

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
LinKernighanHelsgaunForTsp::new(
    stop_condition: StopCondition,
    num_neighbors: usize,
    max_depth: usize,
) -> Self
```

Defaults: `num_neighbors = 5`, `max_depth = 5`.

`clear()` drops the local-optimum flag that `is_done` reads, so a second `run`
searches again instead of reporting itself done at once; the instance-derived
candidate lists survive it, since they depend on nothing else.

## Benchmark config

```toml
[[heuristics]]
kind = "LinKernighanHelsgaun"
num_neighbors = 5        # optional (default shown)
max_depth = 5            # optional (default shown)
[heuristics.stop_condition]
max_duration_secs = 30.0
```

Takes no `neighbor` — it owns its move set. It stops at a local optimum, so a
long budget only pays inside [`Restart` or `Iterated`](meta.md).

## References

- Lin, S. and Kernighan, B. W. "An Effective Heuristic Algorithm for the
  Traveling-Salesman Problem." *Operations Research*, 21(2), 498-516, 1973.
- Helsgaun, K. "An Effective Implementation of the Lin-Kernighan Traveling
  Salesman Heuristic." *European Journal of Operational Research*, 126(1),
  106-130, 2000.
