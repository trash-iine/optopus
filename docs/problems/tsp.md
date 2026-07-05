# TSP 2D

Traveling Salesman Problem on cities given by 2D coordinates. **Minimize**
the total length of a Hamiltonian tour.

## Solution

```rust
pub struct TspSolution {
    pub tour: Vec<usize>,                                  // city indices, 0-indexed
    pub objective: f64,                                    // total tour length
    pub gain: HashMap<(TspEdge, TspEdge), f64>,            // cached 2-opt gains
}
```

`Rankable::is_better_than` returns `self.objective < other.objective`. The
`gain` map is keyed by a normalized pair of directed edges
`((a, b), (c, d))`; the value is the change in tour length when the two edges
are swapped (negative = improving).

## Neighbors

| Type | Move | Iteration cost |
|---|---|---|
| `TspTwoOptNeighbor` | 2-opt: reverse a tour segment between two edges. | `iter + 1` |
| `TspRelocateNeighbor` | Remove a city and reinsert it at another position. | `iter + 1` |

`Distance` (position-based dissimilarity) is implemented for use with
`ParentSelection::DistantTopK`.

## Crossover

- `TspOrderCrossover` — Order Crossover (OX): copy a contiguous segment from
  one parent, fill remaining positions in order from the other parent.

## Construction

```rust
use optopus::prelude::*;

// In-memory (defaults to EdgeWeightType::Continuous):
let tsp = TspWithCoordinates::new(
    "demo".to_string(),
    vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
);

// With a specific edge-weight type:
let tsp = TspWithCoordinates::with_edge_weight_type(
    "demo".to_string(),
    vec![(0.0, 0.0), (1.0, 0.0)],
    EdgeWeightType::Euc2d,
);

// Load from a TSPLIB file:
let tsp = TspWithCoordinates::load_file("data/instances/tsp/att48.tsp")?;
# Ok::<(), optopus::error::OptError>(())
```

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

## Optional traits

- `Distance` — number of positions where the tour differs (diversity proxy).
