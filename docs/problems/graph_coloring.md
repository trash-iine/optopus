# Graph Coloring

**API:** [`GraphColoring`](../api/optopus/problem/graph_coloring/struct.GraphColoring.html)

Given an undirected graph `G = (V, E)`, a proper coloring assigns each vertex
`v` a color `c_v` so that the two endpoints of every edge differ. Minimize the
number of distinct colors used:

```text
minimize  |{c_v : v ∈ V}|   subject to   c_i ≠ c_j  for every edge (i,j) ∈ E
```

The smallest achievable count is the chromatic number `χ(G)`. Deciding whether
a graph is 3-colorable is already NP-complete.

The palette is fixed at `k = max_degree + 1` colors, derived from the graph,
so a proper coloring always exists (Brooks' theorem gives an even tighter
bound for almost every graph). There is no separate `k` input; a run reports
how many of those colors it managed to use.

Feasibility is soft, as for [Vertex Cover](vertex_cover.md). The solver
optimizes a penalty-augmented objective, with `penalty_weight = n + 1` so
that removing one conflict always beats any change in the color count and any
optimum is a proper coloring:

```text
objective(c) = colors_used(c) + penalty_weight · conflicts(c)
```

where `conflicts` is the number of edges whose endpoints share a color.

## Example

Running a search and reading back the coloring:

```rust
use optopus::prelude::*;

let gc = GraphColoring::new(Graph::from_edges([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]));
let mut state = SearchState::new(&gc);
LocalSearch::<GraphColoringRecolorNeighbor>::new(StopCondition::iterations(10_000))
    .run(&mut state)
    .unwrap();

let sol = &state.best_solution;
println!("colors used = {}, conflicts = {}", sol.colors_used, sol.conflicts);
println!("coloring = {:?}", sol.colors); // colors[v] is the color of vertex v
```

## Solution

[`GraphColoringSolution`](../api/optopus/problem/graph_coloring/struct.GraphColoringSolution.html)
holds the assignment `colors` (`colors[v]` is `c_v`, in `0..k`), the
penalty-augmented `objective` defined above, `colors_used`, the number of
non-empty color classes, and `conflicts`, the constraint-violation count. It
also caches, per vertex, how many neighbors carry each color (the Γ matrix
of TabuCol), which is what makes every move's gain O(1).

## Neighbors

| Type | Move | Iteration cost |
|---|---|---|
| [`GraphColoringRecolorNeighbor`](../api/optopus/problem/graph_coloring/struct.GraphColoringRecolorNeighbor.html) | Give a single vertex a different color. Selected by `neighbor = "Flip"` in a config. | `iter + 1` |
| [`GraphColoringSwapNeighbor`](../api/optopus/problem/graph_coloring/struct.GraphColoringSwapNeighbor.html) | Exchange the colors of two differently colored vertices. Leaves `colors_used` unchanged. | `iter + 2` |

Single recolors make the color-count term a flat landscape. A conflict is
repaired strongly, but emptying a color class needs every vertex of that class
to move, one at a time, with no reward until the last one. Expect a search to
reach a proper coloring quickly and then reduce the number of colors slowly.

## Crossover

- `GraphColoringUniformCrossover`, per-vertex random parent selection. Color
  labels are not aligned between the parents, so the offspring's color count
  and conflicts are recomputed from scratch.

## File format

Graph Coloring reuses the [MaxCut graph format](max_cut.md#file-format); edge
weights are ignored (every edge contributes equally to the conflict count).

```rust
use optopus::prelude::*;

let gc = GraphColoring::load_file("data/instances/graph_coloring/example.txt")?;
```

## References

- Hertz, A., and de Werra, D. "Using tabu search techniques for graph
  coloring." *Computing* 39, pp. 345-351, 1987. (TabuCol, the origin of the Γ
  matrix the solution caches.)
- Karp, R. M. "Reducibility Among Combinatorial Problems." In *Complexity of
  Computer Computations*, pp. 85-103. Plenum Press, 1972. (Chromatic Number is
  one of Karp's 21 NP-complete problems.)
