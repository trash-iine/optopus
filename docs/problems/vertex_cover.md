# Vertex Cover

**API:** [`VertexCover`](../api/optopus/problem/vertex_cover/struct.VertexCover.html)

Given an undirected graph `G = (V, E)`, a *vertex cover* is a subset
`S ⊆ V` such that every edge has at least one endpoint in `S`. **Minimize**
the size of such a subset — equivalently, choose a binary membership
`x_v ∈ {0,1}` for each vertex `v` (`x_v = 1` iff `v ∈ S`) subject to every
edge being covered:

```text
minimize  Σ_v x_v   subject to   x_i + x_j ≥ 1  for every edge (i,j) ∈ E
```

Vertex Cover is NP-hard — it is equivalent to Independent Set and Clique via
complementation.

Feasibility is soft: the solver actually optimizes a penalty-augmented
objective, with `penalty_weight` chosen large enough that any optimum of it
is feasible (no uncovered edges):

```text
objective(x) = cover_size(x) + penalty_weight · uncovered_edges(x)
```

## Example

Running a search and reading back which vertices form the cover:

```rust
use optopus::prelude::*;

let vc = VertexCover::new(Graph::from_edges([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]));
let mut state = SearchState::new(&vc);
LocalSearch::<VertexCoverFlipNeighbor>::new(StopCondition::iterations(10_000))
    .run(&mut state)
    .unwrap();

let sol = &state.best_solution;
println!("cover size = {}", sol.cover_size);
let cover: Vec<usize> = sol
    .x
    .iter()
    .enumerate()
    .filter(|&(_, &in_cover)| in_cover)
    .map(|(v, _)| v)
    .collect();
println!("cover = {cover:?}"); // vertices selected to cover every edge
```

## Solution

[`VertexCoverSolution`](../api/optopus/problem/vertex_cover/struct.VertexCoverSolution.html)
represents the membership `x` from the definition above (`x[v]` is `x_v`, the
cover membership of vertex `v`), and the penalty-augmented `objective` 
defined above; `cover_size` is `Σ x_v = |S|`
and `uncovered_edges` is the constraint-violation count.

## Neighbors

| Type | Move | Iteration cost |
|---|---|---|
| [`VertexCoverFlipNeighbor`](../api/optopus/problem/vertex_cover/struct.VertexCoverFlipNeighbor.html) | Flip a single vertex's membership. | `iter + 1` |
| [`VertexCoverSwapNeighbor`](../api/optopus/problem/vertex_cover/struct.VertexCoverSwapNeighbor.html) | Swap a covered vertex with an uncovered one. | `iter + 2` |

## Crossover

- `VertexCoverUniformCrossover` — per-vertex random parent selection.
- `VertexCover` implements `SubProblemExtractable`: vertices that agree in
  both parents are fixed; the remaining vertices form a sub-instance.

## File format

Vertex Cover reuses the [MaxCut graph format](max_cut.md#file-format); edge
weights are ignored (every edge contributes equally to the cover constraint).

```rust
use optopus::prelude::*;

let vc = VertexCover::new(Graph::load_from_file("data/instances/max_cut/G1")?);
# Ok::<(), optopus::error::OptError>(())
```

## References

- Karp, R. M. "Reducibility Among Combinatorial Problems." In *Complexity of
  Computer Computations*, pp. 85-103. Plenum Press, 1972. (Vertex Cover is
  one of Karp's 21 NP-complete problems.)

