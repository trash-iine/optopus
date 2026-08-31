# MaxCut

**API:** [`MaxCut`](../api/optopus/problem/max_cut/struct.MaxCut.html)

Given a weighted undirected graph `G = (V, E, w)` with vertex set `V`, edge
set `E`, and edge weights `w`, partition `V` into two
disjoint sets so as to **maximize** the total weight of the edges that cross
the partition. Equivalently, assign each vertex `i` a binary label
`x_i ∈ {0, 1}` naming which side it falls on; an edge `(i, j)` is *cut*
exactly when `x_i ≠ x_j`:

```text
maximize  Σ_{(i,j)∈E} w_ij · [x_i ≠ x_j]        (x ∈ {0,1}^|V|)
```

## Example

Running a search and reading back the partition it found:

```rust
use optopus::prelude::*;

let mc = MaxCut::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 2.0)]);
let mut state = SearchState::new(&mc);
LocalSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(10_000))
    .run(&mut state)
    .unwrap();

let sol = &state.best_solution;
println!("cut weight = {}", sol.objective);
for (v, &side) in sol.x.iter().enumerate() {
    println!("vertex {v} is on side {}", side as u8); // which of the two sets `v` ended up in
}
```

`MaxCut::from_edges` is a convenience wrapper around
`MaxCut::new(Graph::from_edges(...))`; both use **set semantics** for
duplicate edges — the last write wins.

## Solution

[`MaxCutSolution`](../api/optopus/problem/max_cut/struct.MaxCutSolution.html)
represents the partition from the definition above: `x[v]` is the side
(`false`/`true`) vertex `v`.

## Neighbors

| Type | TOML `neighbor` | Move |
|---|---|---|
| `MaxCutFlipNeighbor` | `"Flip"` | Flip one vertex to the opposite side. `iter + 1`. |
| `MaxCutSwapNeighbor` | `"Swap"` | Swap two vertices on opposite sides. `iter + 2`. |

## Crossover

- `MaxCutUniformCrossover` — per-vertex random parent selection.
- `MaxCut` also implements `SubProblemExtractable`, so `SubProblemBasedCrossover`
  works: vertices that agree in both parents are fixed; the disagreeing
  vertices form the sub-MaxCut instance whose edges include bias terms toward
  the fixed neighborhood.

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

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::load_from_file("data/instances/max_cut/G1")?);
# Ok::<(), optopus::error::OptError>(())
```

## Instances with a known optimum

[`PlantedMaxCut`](../api/optopus/problem/max_cut/struct.PlantedMaxCut.html)
builds instances *around* a chosen solution, so the optimum is exact by
construction rather than a best-known value. 

```rust
use optopus::common::seeded_rng;
use optopus::problem::{PlantedMaxCut, TileProbs2d};

let planted = PlantedMaxCut::tile_planting_2d(
    40, // 40 x 40 torus, degree 4
    TileProbs2d::new(0.35, 0.0, 0.65),
    &mut seeded_rng(1),
);
planted.verify().unwrap(); // the recorded optimum is what the instance computes
// planted.optimum — no run can exceed this
```

| Constructor | Topology | Hardness knob |
|---|---|---|
| `tile_planting_2d(l, TileProbs2d, rng)` | square lattice torus, degree 4 | class mixture `p1`/`p2`/`p3` |
| `tile_planting_3d(l, TileProbs3d, rng)` | cubic lattice torus, degree 6 | class mixture `p_2fp`/`p_4fp` |
| `wishart(n, alpha, WishartCouplers, rng)` | complete graph | `alpha = M / n`, in `(0, 1)` |

- **Every instance is gauge-transformed.** Each construction natively plants the
  all-aligned state, which is trivially findable; a random gauge moves the
  optimum to an arbitrary partition. This is switching on a signed graph, so it
  relabels the solution while leaving the frustration structure — and therefore
  the difficulty — untouched.
- **Integer weights make "reached the optimum" decidable.** Tile planting and
  `WishartCouplers::Discrete` produce integer weights, so the `f32` objective is
  exact. `WishartCouplers::Gaussian` does not, and there a run can only be
  scored up to a rounding bound. `verify()` enforces the distinction and refuses
  any instance whose optimum no longer round-trips.
- **`alpha` must stay below 1, and the useful value depends on `n`.** The
  planted vector lies in the kernel of the Wishart coupling matrix, whose
  dimension is `n - M`; at `alpha >= 1` that kernel collapses onto the planted
  vector and an eigendecomposition recovers it in polynomial time, so the
  constructor rejects it. Well short of that, a hardness sweep found the
  boundary between "always solved" and "never solved" at a **constant kernel
  dimension `n - M` of about 32** —
  `alpha = 0.35 / 0.50 / 0.65 / 0.90` at `n = 48 / 64 / 96 / 256`. Note that
  small `alpha` is the *hard* side at every size measured, not the easy one the
  original study's easy–hard–easy profile suggests; the criterion here is
  reaching the exact optimum rather than the physics notion of a ground state.
  `chook`'s default `alpha = 0.75` is easy at all four sizes.

Suite generation lives in `examples/generate_hard_maxcut.rs`, which records what
the sweep showed for each parameter it bakes in; see
[`data/instances/README.md`](https://github.com/trash-iine/optopus/blob/main/data/instances/README.md).

## Notes

- `MaxCutSolution`'s optional `positive_gain` / `zero_gain` indexes power
  [Breakout Local Search](../heuristics/breakout_local_search.md) (and the
  [learned-policy controller](../guide/learned_perturbation.md) built on it) and
  [Population Annealing](../heuristics/population_annealing.md); standard
  heuristics need neither. See
  [`MaxCutSolution`](../api/optopus/problem/max_cut/struct.MaxCutSolution.html)
  rustdoc for the implementation details.

## References

- Karp, R. M. "Reducibility Among Combinatorial Problems." In *Complexity of
  Computer Computations*, pp. 85-103. Plenum Press, 1972. (Max Cut is one of
  Karp's 21 NP-complete problems.)
- Standard benchmark set: the **Gset** graphs (G1–G81), generated with the
  `rudy` graph generator and distributed by Y. Ye.
- Perera, D. et al. "Chook — A comprehensive suite for generating binary
  optimization problems with planted solutions."
  [arXiv:2005.14344](https://arxiv.org/abs/2005.14344). The reference
  implementation `PlantedMaxCut` follows.
- Perera, D., Hamze, F., Raymond, J., Weigel, M. and Katzgraber, H. G.
  "Computational hardness of spin-glass problems with tile-planted solutions."
  *Phys. Rev. E* 101, 023316 (2020).
  [arXiv:1907.10809](https://arxiv.org/abs/1907.10809)
- Hamze, F., Raymond, J., Pattison, C. A., Biswas, K. and Katzgraber, H. G.
  "Wishart planted ensemble: A tunably rugged pairwise Ising model with a
  first-order phase transition." *Phys. Rev. E* 101, 052102 (2020).
  [arXiv:1906.00275](https://arxiv.org/abs/1906.00275)
