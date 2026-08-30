# TSP 2D

**API:** [`TspWithCoordinates`](../api/optopus/problem/tsp_2d/struct.TspWithCoordinates.html)

Given `n` cities placed at 2D coordinates, with `d(i, j)` the distance between
cities `i` and `j`, find the shortest closed tour that visits every city
exactly once and returns to its start. A tour is a permutation `π` of
`{1, ..., n}`, where `π(k)` names the `k`-th city visited. **Minimize** the
total length of that Hamiltonian tour:

```text
minimize  Σ_{k=1}^{n} d(π(k), π(k mod n + 1))    (π a permutation of the n cities)
```

This crate's `TspWithCoordinates` restricts instances to 2D coordinates
(rather than an arbitrary distance matrix) and supports several standard
distance formulas matching the TSPLIB benchmark set's conventions — see
[Edge-weight types](#edge-weight-types) below.

## Example

Running a search and reading back the visiting order it found:

```rust
use optopus::prelude::*;

let tsp = TspWithCoordinates::new(
    "demo".to_string(),
    vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],  // square placement
);
let mut state = SearchState::new(&tsp);
LocalSearch::<TspTwoOptNeighbor>::new(StopCondition::iterations(10_000))
    .run(&mut state)
    .unwrap();

let sol = &state.best_solution;
println!("tour length = {}", sol.objective);
println!("visiting order = {:?}", sol.tour); // city indices in the order they are visited
```

`TspWithCoordinates::new` defaults to `EdgeWeightType::Continuous`; use
`TspWithCoordinates::with_edge_weight_type` to pick one of the formulas
below.

## Solution

[`TspSolution`](../api/optopus/problem/tsp_2d/struct.TspSolution.html) carries
the permutation `π` from the definition above as `tour` (`tour[k]` is
`π(k)`, the `k`-th city visited), and the tour length `objective`, 
which is `Σ d(π(k), π(k+1))`.

## Neighbors

| Type | Move | Iteration cost |
|---|---|---|
| `TspTwoOptNeighbor` | 2-opt: reverse a tour segment between two edges. | `iter + 1` |
| `TspRelocateNeighbor` | Remove a city and reinsert it at another position. | `iter + 1` |

## Crossover

- `TspOrderCrossover` — Order Crossover (OX): copy a contiguous segment from
  one parent, fill remaining positions in order from the other parent.

## Edge-weight types

`EdgeWeightType` selects the distance formula:

| Variant | Formula | TSPLIB key |
|---|---|---|
| `Continuous` | plain Euclidean (no rounding) | — (default for `new`) |
| `Euc2d` | `nint(sqrt(dx² + dy²))` | `EUC_2D` |
| `Ceil2d` | `ceil(sqrt(dx² + dy²))` | `CEIL_2D` |
| `Att` | TSPLIB pseudo-Euclidean | `ATT` |
| `Geo` | TSPLIB great-circle (DDD.MM → radians, R = 6378.388 km) | `GEO` |

## File format (TSPLIB)

```text
NAME: <name>
TYPE: TSP
COMMENT: ...
DIMENSION: N
EDGE_WEIGHT_TYPE: EUC_2D | CEIL_2D | ATT | GEO
NODE_COORD_SECTION
1 x1 y1
2 x2 y2
...
EOF
```

`TYPE`, `COMMENT`, and other unknown header keys are skipped. Header keys may
appear in any order. Coordinate lines are 1-indexed and converted to 0-indexed
internally. The `EXPLICIT` weight type is not supported.

```rust
use optopus::prelude::*;

let tsp = TspWithCoordinates::load_file("data/instances/tsp/att48.tsp")?;
# Ok::<(), optopus::error::OptError>(())
```

## References

- Reinelt, G. "TSPLIB — A Traveling Salesman Problem Library." *ORSA Journal
  on Computing*, 3(4), 376-384, 1991. (Defines the file format and the
  standard instance set.)

