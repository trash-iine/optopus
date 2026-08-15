# MaxCut

Partition the vertices of a weighted undirected graph into two sets so as to
**maximize** the total weight of edges crossing the partition.

Sparse instances can be shrunk first without giving anything up: see
[MaxCutKernel](max_cut_kernel.md), an exact data reduction any heuristic can
search through.

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

## Instances with a known optimum

`PlantedMaxCut` (`src/problem/max_cut/planted.rs`) builds instances *around* a
chosen solution, so the optimum is exact by construction rather than a
best-known value. That changes what a benchmark can say: on the G-set a gap is
measured against the best result anyone has published — a number that has moved
as recently as 2025 and that different papers report differently by 1 to 4 — and
here it is measured against the answer.

```rust
use optopus::common::seeded_rng;
use optopus::problem::{PlantedMaxCut, TileProbs2d};

let planted = PlantedMaxCut::tile_planting_2d(
    40,                              // 40 x 40 torus, degree 4
    TileProbs2d::new(0.35, 0.0, 0.65),
    &mut seeded_rng(1),
);
planted.verify().unwrap();           // the recorded optimum is what the instance computes
// planted.optimum — no run can exceed this
```

| Constructor | Topology | Hardness knob |
|---|---|---|
| `tile_planting_2d(l, TileProbs2d, rng)` | square lattice torus, degree 4 | class mixture `p1`/`p2`/`p3` |
| `tile_planting_3d(l, TileProbs3d, rng)` | cubic lattice torus, degree 6 | class mixture `p_2fp`/`p_4fp` |
| `wishart(n, alpha, WishartCouplers, rng)` | complete graph | `alpha = M / n`, in `(0, 1)` |

Three things are worth knowing before using them:

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
[`data/instances/README.md`](../../data/instances/README.md).

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
