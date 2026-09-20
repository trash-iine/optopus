# TSP

**API:** [`Tsp`](../api/optopus/problem/tsp/struct.Tsp.html)

Given `n` cities with `d(i, j)` the distance between cities `i` and `j`,
find the shortest closed tour that visits every city exactly once and
returns to its start. A tour is a permutation `π` of
`{1, ..., n}`, where `π(k)` names the `k`-th city visited. Minimize the
total length of that Hamiltonian tour:

```text
minimize  Σ_{k=1}^{n} d(π(k), π(k mod n + 1))    (π a permutation of the n cities)
```

This crate's `Tsp` holds the distances themselves. An instance is built
from 2D coordinates and one of the standard TSPLIB distance formulas, see
[Edge-weight types](#edge-weight-types) below, or from a distance matrix
given directly. Which distances it keeps in memory is chosen at
construction, see [Distance storage](#distance-storage).

## Example

Running a search and reading back the visiting order it found:

```rust
use optopus::prelude::*;

let tsp = Tsp::new(
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

`Tsp::new` defaults to `EdgeWeightType::Continuous`; use
`Tsp::with_edge_weight_type` to pick one of the formulas
below.

## Distance storage

| Constructor | Keeps | Use it when |
|---|---|---|
| `Tsp::new(name, coords)`, `Tsp::with_edge_weight_type(name, coords, ewt)` | the full `n × n` matrix, `8n²` bytes | `n` is a few thousand at most |
| `Tsp::with_nearest_neighbors(name, coords, ewt, k)` | the `k` nearest neighbours of every city, `n × k` distances | the matrix would not fit |
| `Tsp::from_distance_matrix(name, matrix)` | the matrix as given | the distances are not Euclidean or the coordinates are not available |

A nearest-neighbour instance still answers `distance(i, j)` for every pair.
A pair outside the lists is computed from the coordinates, so the values are
the same as with the full matrix and only the far pairs cost more. The
candidate lists that Lin-Kernighan and the anchored tour descent ask for
come straight from the stored rows when `k` covers them.

`Tsp::from_distance_matrix` takes `Vec<Vec<f64>>` and rejects a matrix that
is not square. Such an instance has no coordinates, so `coordinates()` and
`edge_weight_type()` return `None` on it. Give it a symmetric matrix. The
neighbourhoods price a 2-opt move from the four edges it exchanges, which
holds only when reversing a segment leaves its length unchanged, so an
asymmetric matrix makes the reported gains drift from the tour length.

```rust
use optopus::prelude::*;

let matrix = vec![
    vec![0.0, 3.0, 7.0],
    vec![3.0, 0.0, 5.0],
    vec![7.0, 5.0, 0.0],
];
let tsp = Tsp::from_distance_matrix("triangle".to_string(), matrix)?;
```

`Tsp::load_file` keeps the full matrix for files with at most
`Tsp::DIST_MATRIX_MAX_N` cities (2000) and switches to
`Tsp::NEAREST_NEIGHBORS_ABOVE_CAP` neighbours per city (20) above that.

## Solution

[`TspSolution`](../api/optopus/problem/tsp/struct.TspSolution.html) carries
the permutation `π` from the definition above as `tour` (`tour[k]` is
`π(k)`, the `k`-th city visited), and the tour length `objective`, 
which is `Σ d(π(k), π(k+1))`.

## Neighbors

| Type | Move | Iteration cost |
|---|---|---|
| `TspTwoOptNeighbor` | 2-opt: reverse a tour segment between two edges. | `iter + 1` |
| `TspRelocateNeighbor` | Remove a city and reinsert it at another position. | `iter + 1` |

## Crossover

- `TspOrderCrossover`, Order Crossover (OX): copy a contiguous segment from
  one parent, fill remaining positions in order from the other parent.

## Ruin and recreate

`Tsp` implements `Ruinable` with cities as the elements and the
tour as the one container, so
[AdaptiveLargeNeighborhoodSearch](../heuristics/alns.md) runs on it.
`alns_for_tsp` pairs the search with `AnchoredTourDescent`, an Or-opt and
2-opt descent over the cities a ruin just re-inserted and their nearest
neighbours, whose granularity, ring and pass count are builders described on
that page.

## Edge-weight types

`EdgeWeightType` selects the distance formula:

| Variant | Formula | TSPLIB key |
|---|---|---|
| `Continuous` | plain Euclidean (no rounding) | (default for `new`) |
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

let tsp = Tsp::load_file("data/instances/tsp/att48.tsp")?;
```

## References

- Reinelt, G. "TSPLIB, A Traveling Salesman Problem Library." *ORSA Journal
  on Computing*, 3(4), 376-384, 1991. (Defines the file format and the
  standard instance set.)

