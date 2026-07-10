# MaxCut

Partition the vertices of a weighted undirected graph into two sets so as to
**maximize** the total weight of edges crossing the partition.

## Solution

```rust
pub struct MaxCutSolution {
    pub cut: Vec<bool>,        // partition assignment per vertex
    pub gain: Vec<f32>,        // change in objective when each vertex is flipped
    pub objective: f32,        // total weight of crossing edges
    // pub(crate) positive_gain_*  — optional advanced index
}
```

`Rankable::is_better_than` returns `self.objective > other.objective`.

## Neighbors

| Type | Move | Iteration cost |
|---|---|---|
| `MaxCutFlipNeighbor` | Flip the side of one vertex; gain refresh in O(degree). | `iter + 1` |
| `MaxCutSwapNeighbor` | Swap two vertices on opposite sides. | `iter + 2` |

Both implement `Rankable`, `Evaluate<f64>`, and `EnabledTabu`.

## Crossover

- `MaxCutUniformCrossover` — per-vertex random parent selection.
- `MaxCut` also implements `SubProblemExtractable`, so `SubProblemBasedCrossover`
  works: vertices that agree in both parents are fixed; the disagreeing
  vertices form the sub-MaxCut instance whose edges include bias terms toward
  the fixed neighborhood.

## Construction

```rust
use optopus::prelude::*;

// Inline edges (1-indexed in the file format, 0-indexed here):
let mc = MaxCut::new(Graph::from_edges([
    (0, 1, 1.0),
    (1, 2, 2.0),
]));

// Convenience wrapper (same semantics):
let mc = MaxCut::from_edges([(0, 1, 1.0), (1, 2, 2.0)]);

// Load from file:
let mc = MaxCut::new(Graph::load_from_file("data/instances/max_cut/G1")?);
# Ok::<(), optopus::error::OptError>(())
```

`Graph::from_edges` and `Graph::load_from_file` use **set semantics** for
duplicate edges — the last write wins.

## File format

`Graph::load_from_file` expects one header line followed by edge lines, with
**1-indexed** vertices:

```text
N M
i j w
i j w
...
```

- `N` — number of vertices, `M` — number of edges.
- `w` is optional; defaults to `1.0` if absent.
- Vertices are converted to 0-indexed internally.

## Optional traits

- `Distance` — Hamming distance on the cut vector (used by `ParentSelection::DistantTopK`).

## Notes

- `MaxCutSolution` carries an optional **`positive_gain` index** that
  enumerates only improving flips in O(|improving|). It is used by problem-
  specific algorithms such as [Breakout Local Search](../heuristics/breakout_local_search.md);
  standard heuristics do not need to enable it.

## References

- Karp, R. M. "Reducibility Among Combinatorial Problems." In *Complexity of
  Computer Computations*, pp. 85-103. Plenum Press, 1972. (Max Cut is one of
  Karp's 21 NP-complete problems.)
- Standard benchmark set: the **Gset** graphs (G1–G81), generated with the
  `rudy` graph generator and distributed by Y. Ye. See
  [`data/instances/README.md`](../../data/instances/README.md) for instance
  sources and download instructions.
