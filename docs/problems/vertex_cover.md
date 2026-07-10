# Vertex Cover

Given an undirected graph, **minimize** the size of a vertex subset that
covers every edge. Hard feasibility is enforced by penalty:

```text
objective = cover_size + penalty_weight * uncovered_edges
penalty_weight = n + 1
```

The penalty weight is large enough that any optimum is feasible (no uncovered
edges).

## Solution

```rust
pub struct VertexCoverSolution {
    pub cover: Vec<bool>,           // cover[v] = true iff v is selected
    pub gain: Vec<i32>,             // change in objective per flip (negative = improving)
    pub objective: i32,             // penalty-augmented objective
    pub cover_size: usize,          // current |cover|
    pub uncovered_edges: usize,     // current uncovered edge count
}
```

`Rankable::is_better_than` returns `self.objective < other.objective`.

## Neighbors

| Type | Move | Iteration cost |
|---|---|---|
| `VertexCoverFlipNeighbor` | Flip a single vertex's membership; gain refresh in O(degree). | `iter + 1` |
| `VertexCoverSwapNeighbor` | Swap a covered vertex with an uncovered one. | `iter + 2` |

Both implement `Rankable`, `Evaluate<f64>`, and `EnabledTabu`.

## Crossover

- `VertexCoverUniformCrossover` — per-vertex random parent selection.
- `VertexCover` implements `SubProblemExtractable`: vertices that agree in
  both parents are fixed; the remaining vertices form a sub-instance.

## Construction

```rust
use optopus::prelude::*;

let vc = VertexCover::new(Graph::from_edges([
    (0, 1, 1.0),
    (1, 2, 1.0),
]));

// Load via the shared graph loader:
let vc = VertexCover::new(Graph::load_from_file("data/instances/max_cut/G1")?);
# Ok::<(), optopus::error::OptError>(())
```

Vertex Cover reuses the [MaxCut graph format](max_cut.md#file-format); edge
weights are ignored (every edge contributes equally to the cover constraint).

## Optional traits

- `Distance` — Hamming distance on `cover`.

## References

- Karp, R. M. "Reducibility Among Combinatorial Problems." In *Complexity of
  Computer Computations*, pp. 85-103. Plenum Press, 1972. (Vertex Cover is
  one of Karp's 21 NP-complete problems.)
- See [`data/instances/README.md`](../../data/instances/README.md) for
  instance sources.
