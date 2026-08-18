//! Planted-solution [`MaxCut`] instances, where the optimum is known by
//! construction rather than by consensus of the literature.
//!
//! Every standard MaxCut benchmark reports a gap against a *best-known* value
//! that is neither an upper nor a lower bound. On the G-set that is not a
//! theoretical worry: published values for `G59`, `G60` and `G61` disagree by
//! 1 to 4 depending on the source, which is the same order as the difference
//! between two competing heuristics. Planting removes the problem at the root
//! — the instance is built *around* a chosen solution in such a way that no
//! better one can exist — so a run's gap is measured against the truth.
//!
//! The two families here are the two that Perera et al. package for pairwise
//! (2-local) Ising problems, which is exactly the class that maps to weighted
//! MaxCut without auxiliary variables:
//!
//! | Family | Topology | Hardness knob | Degree |
//! |---|---|---|---|
//! | [`tile_planting_2d`](PlantedMaxCut::tile_planting_2d) | square lattice torus | `p1`, `p2`, `p3` | 4 |
//! | [`tile_planting_3d`](PlantedMaxCut::tile_planting_3d) | cubic lattice torus | `p_2fp`, `p_4fp` | 6 |
//! | [`wishart`](PlantedMaxCut::wishart) | complete graph | `alpha = M / N` | `N - 1` |
//!
//! Tile planting sits at degree 4, the same band as the G-set `toroidal`
//! group, so it extends an established comparison rather than starting a new
//! one. Wishart planting is dense and has a first-order phase transition,
//! which makes it the sharpest available test of any method that assumes
//! agreement between good solutions implies correctness.
//!
//! # Why the planted state is optimal
//!
//! Both tile-planting constructions partition the edges into small
//! **edge-disjoint** tiles that share only vertices. The energy is therefore a
//! sum of independent per-tile terms, and each tile is drawn from a class whose
//! ground states *all* include the all-aligned state. A configuration that
//! minimizes every term simultaneously minimizes the sum, so the all-aligned
//! state is a global ground state — no search is involved in the claim.
//! Wishart planting instead builds the coupling matrix in the null space of the
//! planted vector, which puts the planted state at the minimum directly.
//!
//! # Gauge transformation
//!
//! All three constructions plant the *ferromagnetic* state, which would be
//! found instantly. Each instance is therefore concealed by a random gauge
//! transformation `J_ij -> s_i s_j J_ij`, which moves the ground state to `s`.
//! This is the switching operation on signed graphs, so it relabels the
//! solution while preserving the frustration structure exactly: **the
//! transformation changes nothing about how hard the instance is**.
//!
//! # Ising to MaxCut
//!
//! The generators produce a zero-field Ising model `E(s) = -1/2 sᵀJs` with a
//! zero diagonal. Setting the edge weight `w_ij = -J_ij` gives
//! `cut(s) = (Σw - E(s)) / 2`, so minimizing the energy maximizes the cut, and
//! the planted ground state is the maximum cut. Weights may be negative, as on
//! the signed G-set instances.
//!
//! # References
//!
//! - Perera, D., Akpabio, I., Hamze, F., Mandrà, S., Rose, N., Aramon, M. and
//!   Katzgraber, H. G. "Chook — A comprehensive suite for generating binary
//!   optimization problems with planted solutions."
//!   [arXiv:2005.14344](https://arxiv.org/abs/2005.14344). The constructions
//!   below follow its reference implementation (`chook/planters/`).
//! - Perera, D., Hamze, F., Raymond, J., Weigel, M. and Katzgraber, H. G.
//!   "Computational hardness of spin-glass problems with tile-planted
//!   solutions." *Phys. Rev. E* 101, 023316 (2020).
//!   [arXiv:1907.10809](https://arxiv.org/abs/1907.10809)
//! - Hamze, F., Raymond, J., Pattison, C. A., Biswas, K. and Katzgraber, H. G.
//!   "Wishart planted ensemble: A tunably rugged pairwise Ising model with a
//!   first-order phase transition." *Phys. Rev. E* 101, 052102 (2020).
//!   [arXiv:1906.00275](https://arxiv.org/abs/1906.00275)
//!
//! # Example
//!
//! ```
//! use optopus::common::seeded_rng;
//! use optopus::problem::{PlantedMaxCut, TileProbs2d};
//!
//! let mut rng = seeded_rng(1);
//! let planted = PlantedMaxCut::tile_planting_2d(8, TileProbs2d::new(0.35, 0.0, 0.65), &mut rng);
//!
//! // 4-regular on l * l vertices, and the recorded optimum is a real cut.
//! assert_eq!(planted.problem.graph.num_vertices(), 64);
//! assert_eq!(planted.problem.graph.num_edges(), 128);
//! assert_eq!(
//!     planted.problem.calculate_cut_size(&planted.planted),
//!     planted.optimum
//! );
//! planted.verify().unwrap();
//! ```

use rand::Rng;
use rand::seq::SliceRandom;

use super::problem::MaxCut;
use crate::common::Graph;
use crate::error::OptError;

/// A MaxCut instance together with the solution it was built around.
///
/// See the [MaxCut guide](https://trash-iine.github.io/optopus/problems/max_cut/)
/// for how each family guarantees that
/// [`planted`](Self::planted) is optimal.
#[derive(Debug, Clone)]
pub struct PlantedMaxCut {
    /// The instance, with the gauge transformation already applied.
    pub problem: MaxCut,
    /// The planted cut. This is an optimal solution, not merely a good one.
    pub planted: Vec<bool>,
    /// The cut weight of [`planted`](Self::planted) — the exact optimum.
    pub optimum: f32,
}

/// Class probabilities for square-lattice tile planting.
///
/// A plaquette of class `i` has exactly `i` ground states modulo spin
/// inversion, one of which is the planted all-aligned state. Higher classes are
/// more degenerate, so the mixture controls how rugged the landscape is. The
/// remaining probability `1 - p1 - p2 - p3` goes to class 4.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileProbs2d {
    /// Probability of a class-1 plaquette (one `-1`, three `+2` couplers).
    pub p1: f64,
    /// Probability of a class-2 plaquette (one `-1`, one `+1`, two `+2`).
    pub p2: f64,
    /// Probability of a class-3 plaquette (one `-1`, two `+1`, one `+2`).
    pub p3: f64,
}

impl TileProbs2d {
    /// Builds the mixture; class 4 takes the remaining probability.
    ///
    /// # Panics
    ///
    /// Panics if any entry is negative or if they sum above 1.
    pub fn new(p1: f64, p2: f64, p3: f64) -> Self {
        assert!(
            p1 >= 0.0 && p2 >= 0.0 && p3 >= 0.0,
            "TileProbs2d requires non-negative probabilities, got ({p1}, {p2}, {p3})"
        );
        assert!(
            p1 + p2 + p3 <= 1.0,
            "TileProbs2d requires p1 + p2 + p3 <= 1, got {}",
            p1 + p2 + p3
        );
        Self { p1, p2, p3 }
    }
}

/// Class probabilities for cubic-lattice tile planting.
///
/// A voxel class is named by how many of its six facet plaquettes are
/// frustrated. The remaining probability `1 - p_2fp - p_4fp` goes to the
/// 6-frustrated-plaquette class, which is the hardest of the three: the
/// reference implementation notes that `p_6fp = 1` is the hardest regime found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileProbs3d {
    /// Probability of a voxel with two frustrated facets (class C2,2).
    pub p_2fp: f64,
    /// Probability of a voxel with four frustrated facets (class C4,2).
    pub p_4fp: f64,
}

impl TileProbs3d {
    /// Builds the mixture; the 6-frustrated-facet class takes the remainder.
    ///
    /// # Panics
    ///
    /// Panics if either entry is negative or if they sum above 1.
    pub fn new(p_2fp: f64, p_4fp: f64) -> Self {
        assert!(
            p_2fp >= 0.0 && p_4fp >= 0.0,
            "TileProbs3d requires non-negative probabilities, got ({p_2fp}, {p_4fp})"
        );
        assert!(
            p_2fp + p_4fp <= 1.0,
            "TileProbs3d requires p_2fp + p_4fp <= 1, got {}",
            p_2fp + p_4fp
        );
        Self { p_2fp, p_4fp }
    }
}

/// Coupler distribution for [`PlantedMaxCut::wishart`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WishartCouplers {
    /// Real-valued couplers from Gaussian variates. The default, and the only
    /// one usable at benchmark sizes.
    Gaussian,
    /// Integer couplers from `{-1, 1}` variates, scaled by `N²(N-1)`.
    ///
    /// Integer weights make the cut value exact, so "did the run reach the
    /// optimum" is a comparison rather than a judgement call about rounding.
    /// The price is size: the scaling grows as `N³`, and past some point the
    /// cut value no longer round-trips through the `f32` objective. Measured at
    /// `alpha = 0.75`, instances survive to `n = 96` and fail from `n = 128` —
    /// but that is where the draws happened to land, not a bound, so
    /// [`PlantedMaxCut::verify`] is what decides. It rejects any instance whose
    /// stored optimum is not exactly what the instance computes.
    Discrete,
}

impl PlantedMaxCut {
    /// Generates a square-lattice tile-planted instance on `l * l` vertices.
    ///
    /// The lattice is the 4-regular torus of [`Graph::grid_torus_2d`], and its
    /// edges are partitioned into `l² / 2` plaquettes in a checkerboard
    /// pattern: the plaquette rooted at `(m, n)` (for `m ≡ n mod 2`) covers the
    /// 4-cycle through `(m, n)`, `(m+1, n)`, `(m+1, n+1)`, `(m, n+1)`. Each
    /// plaquette gets one antiferromagnetic `-1` coupler and three
    /// ferromagnetic ones, `p.class` of which are weak (`+1`) and the rest
    /// strong (`+2`).
    ///
    /// # Panics
    ///
    /// Panics if `l` is odd or below 4. An odd `l` would leave the checkerboard
    /// inconsistent across the periodic seam, so the plaquettes would no longer
    /// partition the edges and the planted state would not be optimal.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::common::seeded_rng;
    /// use optopus::problem::{PlantedMaxCut, TileProbs2d};
    ///
    /// let p = PlantedMaxCut::tile_planting_2d(6, TileProbs2d::new(0.2, 0.5, 0.1), &mut seeded_rng(7));
    /// assert_eq!(p.problem.graph.num_edges(), 2 * 36);
    /// ```
    pub fn tile_planting_2d(l: usize, p: TileProbs2d, rng: &mut impl Rng) -> Self {
        assert!(
            l >= 4 && l.is_multiple_of(2),
            "tile_planting_2d requires an even l >= 4, got {l}"
        );

        let idx = |m: usize, n: usize| m + l * n;
        let mut couplings = Vec::with_capacity(2 * l * l);
        for n in 0..l {
            for m in (n % 2..l).step_by(2) {
                // chook's plaquette adjacency is (0,1), (0,2), (1,3), (2,3)
                // over [(m,n), (m+1,n), (m,n+1), (m+1,n+1)] — the 4-cycle.
                let v = [
                    idx(m, n),
                    idx((m + 1) % l, n),
                    idx(m, (n + 1) % l),
                    idx((m + 1) % l, (n + 1) % l),
                ];
                let ends = [(0, 1), (0, 2), (1, 3), (2, 3)];
                for (&(a, b), j) in ends.iter().zip(sample_plaquette(p, rng)) {
                    couplings.push((v[a], v[b], j));
                }
            }
        }
        Self::from_ising(l * l, couplings, rng)
    }

    /// Generates a cubic-lattice tile-planted instance on `l³` vertices.
    ///
    /// The lattice is the 6-regular torus of [`Graph::grid_torus_3d`], and its
    /// edges are partitioned into `l³ / 4` unit-cube voxels: layer `k` carries
    /// voxels rooted at every `(m, n)` with `m ≡ n ≡ k mod 2`, and each voxel
    /// spans layers `k` and `k+1`. Neighbouring voxels meet at a single vertex,
    /// never along an edge.
    ///
    /// Each voxel starts ferromagnetic on all twelve edges, flips the two or
    /// three edges its class names, and is then relabelled by a uniformly random
    /// element of the octahedral group.
    ///
    /// # Panics
    ///
    /// Panics if `l` is odd or below 4, for the same reason as
    /// [`tile_planting_2d`](Self::tile_planting_2d).
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::common::seeded_rng;
    /// use optopus::problem::{PlantedMaxCut, TileProbs3d};
    ///
    /// let p = PlantedMaxCut::tile_planting_3d(4, TileProbs3d::new(0.0, 0.0), &mut seeded_rng(7));
    /// assert_eq!(p.problem.graph.num_edges(), 3 * 64);
    /// ```
    pub fn tile_planting_3d(l: usize, p: TileProbs3d, rng: &mut impl Rng) -> Self {
        assert!(
            l >= 4 && l.is_multiple_of(2),
            "tile_planting_3d requires an even l >= 4, got {l}"
        );

        let idx = |m: usize, n: usize, k: usize| m + l * n + l * l * k;
        let mut couplings = Vec::with_capacity(3 * l * l * l);
        for k in 0..l {
            let offset = k % 2;
            for n in (offset..l).step_by(2) {
                for m in (offset..l).step_by(2) {
                    // Local voxel index is `dm + 2 dn + 4 dk`, matching chook.
                    let v: Vec<usize> = (0..8)
                        .map(|b| {
                            idx(
                                (m + (b & 1)) % l,
                                (n + ((b >> 1) & 1)) % l,
                                (k + ((b >> 2) & 1)) % l,
                            )
                        })
                        .collect();
                    for (a, b, j) in sample_voxel(p, rng) {
                        couplings.push((v[a], v[b], j));
                    }
                }
            }
        }
        Self::from_ising(l * l * l, couplings, rng)
    }

    /// Generates a Wishart-planted instance on the complete graph `K_n`.
    ///
    /// Draws `M = round(alpha * n)` constraint vectors from a distribution
    /// whose covariance is the projector orthogonal to the planted vector, so
    /// `Wᵀt = 0` holds exactly and `J = -WWᵀ/n` (diagonal removed) has `t` as a
    /// ground state.
    ///
    /// `alpha` traces an easy–hard–easy profile. It must stay below 1: the
    /// planted vector lies in the kernel of `WWᵀ`, whose dimension is `n - M`,
    /// so once `M` approaches `n` the kernel collapses onto the planted vector
    /// and an eigendecomposition recovers it in polynomial time. Where the peak
    /// sits within `(0, 1)` depends on `n`, so it should be swept rather than
    /// taken from a paper.
    ///
    /// # Panics
    ///
    /// Panics if `n < 2`, or if `alpha` is not in `(0, 1)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::common::seeded_rng;
    /// use optopus::problem::{PlantedMaxCut, WishartCouplers};
    ///
    /// let p = PlantedMaxCut::wishart(32, 0.75, WishartCouplers::Gaussian, &mut seeded_rng(7));
    /// assert_eq!(p.problem.graph.num_edges(), 32 * 31 / 2);
    /// ```
    pub fn wishart(n: usize, alpha: f64, couplers: WishartCouplers, rng: &mut impl Rng) -> Self {
        assert!(n >= 2, "wishart requires n >= 2, got {n}");
        assert!(
            alpha > 0.0 && alpha < 1.0,
            "wishart requires alpha in (0, 1) — at alpha >= 1 the planted vector \
             is recoverable by eigendecomposition — got {alpha}"
        );
        let m = ((alpha * n as f64).round() as usize).max(1);

        // W = sqrt(n / (n-1)) * (I - 11ᵀ/n) R: each column of R is recentred,
        // which is what makes it orthogonal to the all-ones planted vector.
        let scale = (n as f64 / (n - 1) as f64).sqrt();
        let mut w = vec![0.0f64; n * m];
        for mu in 0..m {
            let column: Vec<f64> = (0..n)
                .map(|_| match couplers {
                    WishartCouplers::Gaussian => standard_normal(rng),
                    WishartCouplers::Discrete => {
                        if rng.random_bool(0.5) {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                })
                .collect();
            let mean = column.iter().sum::<f64>() / n as f64;
            for (i, r) in column.into_iter().enumerate() {
                w[i * m + mu] = scale * (r - mean);
            }
        }

        // J = -WWᵀ/n off the diagonal. Emitting (i, j) in lexicographic order
        // lets `Graph::from_edges` append rather than insert into its sorted
        // adjacency.
        let discrete_scale = (n * n * (n - 1)) as f64;
        let mut couplings = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let dot: f64 = (0..m).map(|mu| w[i * m + mu] * w[j * m + mu]).sum();
                let j_ij = -dot / n as f64;
                let j_ij = match couplers {
                    WishartCouplers::Gaussian => j_ij,
                    WishartCouplers::Discrete => (j_ij * discrete_scale).round(),
                };
                couplings.push((i, j, j_ij));
            }
        }
        Self::from_ising(n, couplings, rng)
    }

    /// Checks that the recorded optimum is the value a solver will actually
    /// compute for the planted cut.
    ///
    /// The instance carries `f32` weights, so a construction whose couplers are
    /// too large would hand out an optimum that no run can match or even
    /// reproduce. This recomputes the planted cut in `f64` and compares it
    /// against [`MaxCut::calculate_cut_size`].
    ///
    /// What counts as agreement depends on the weights, and the difference
    /// matters when reading results. **Integer weights must agree exactly** —
    /// `f32` sums integers exactly until a partial sum leaves its representable
    /// range, so a discrepancy means the couplers have outgrown the objective
    /// and "did this run reach the optimum" has stopped being a decidable
    /// question. **Real-valued weights never agree exactly**, so they are only
    /// held to the standard worst-case bound for sequential summation; a run on
    /// such an instance can only be scored against the optimum up to that
    /// bound.
    ///
    /// It also rejects a zero weight, which is indistinguishable from an absent
    /// edge and would make the instance sparser than its structure claims.
    /// Construction already drops uncoupled pairs, so this is a backstop.
    ///
    /// # Errors
    ///
    /// Returns [`OptError::InvalidState`] describing the first check that fails.
    pub fn verify(&self) -> Result<(), OptError> {
        let all_integral = self.has_exact_optimum();
        let mut exact = 0.0f64;
        let mut magnitude = 0.0f64;
        let mut terms = 0usize;
        for (i, j, w) in self.problem.graph.edges() {
            if w == 0.0 {
                return Err(OptError::InvalidState(format!(
                    "planted instance has a zero-weight edge ({i}, {j}), which is \
                     indistinguishable from an absent one"
                )));
            }
            magnitude += (w as f64).abs();
            terms += 1;
            if self.planted[i] != self.planted[j] {
                exact += w as f64;
            }
        }

        let computed = self.problem.calculate_cut_size(&self.planted);
        if computed != self.optimum {
            return Err(OptError::InvalidState(format!(
                "recorded optimum {} disagrees with the instance's own cut value {computed}",
                self.optimum
            )));
        }

        let error = (computed as f64 - exact).abs();
        // With integer weights the f32 sum is exact until a partial sum leaves
        // the exactly representable range, so any discrepancy at all means the
        // couplers have outgrown the objective — reject it. Real-valued weights
        // can never be exact, so the standard worst-case bound for sequential
        // summation, `n · eps · Σ|x|`, is the strongest honest claim; the
        // instance sums each edge twice and halves.
        let tolerance = if all_integral {
            0.0
        } else {
            2.0 * terms as f64 * f32::EPSILON as f64 * magnitude
        };
        if error > tolerance {
            return Err(OptError::InvalidState(format!(
                "planted cut value does not survive f32: exact {exact}, stored {computed} \
                 (error {error} exceeds tolerance {tolerance}). The couplers are too \
                 large for the f32 objective — shrink the instance, or switch \
                 WishartCouplers::Discrete to Gaussian"
            )));
        }
        Ok(())
    }

    /// Whether reaching [`optimum`](Self::optimum) is a decidable question.
    ///
    /// True when every weight is an integer, which makes the `f32` objective
    /// exact and lets a run be compared against the optimum by equality. On a
    /// real-weighted instance — `WishartCouplers::Gaussian` — a run that
    /// genuinely found the planted state may still compute a value a few bits
    /// away, so it can only be scored within a tolerance.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::common::seeded_rng;
    /// use optopus::problem::{PlantedMaxCut, TileProbs2d, WishartCouplers};
    ///
    /// let mut rng = seeded_rng(1);
    /// let tiles = PlantedMaxCut::tile_planting_2d(6, TileProbs2d::new(0.5, 0.0, 0.5), &mut rng);
    /// assert!(tiles.has_exact_optimum());
    ///
    /// let dense = PlantedMaxCut::wishart(16, 0.5, WishartCouplers::Gaussian, &mut rng);
    /// assert!(!dense.has_exact_optimum());
    /// ```
    pub fn has_exact_optimum(&self) -> bool {
        self.problem.graph.edges().all(|(_, _, w)| w.fract() == 0.0)
    }

    /// Applies the gauge transformation and converts the Ising model into a
    /// MaxCut instance.
    ///
    /// `couplings` holds each edge once, as `(i, j, J_ij)`, for the model whose
    /// ground state is the all-aligned configuration.
    fn from_ising(n: usize, couplings: Vec<(usize, usize, f64)>, rng: &mut impl Rng) -> Self {
        // The gauge `s` becomes the planted cut: it is the ground state of the
        // transformed model, and switching preserves frustration exactly.
        let planted: Vec<bool> = (0..n).map(|_| rng.random_bool(0.5)).collect();
        let sign = |i: usize| if planted[i] { 1.0 } else { -1.0 };

        // A zero coupling means the two spins are not coupled, so it must not
        // become an edge: a zero-weight edge is invisible to the objective and
        // indistinguishable from an absent one, and it would inflate the edge
        // count the instance reports. Rounding the discrete Wishart couplers
        // produces these regularly — the tile families never do.
        let edges = couplings.into_iter().filter_map(|(i, j, j_ij)| {
            let w = (-j_ij * sign(i) * sign(j)) as f32;
            (w != 0.0).then_some((i, j, w))
        });

        let problem = MaxCut::new(Graph::from_edges(edges));
        // `len` can fall below `n` only if the highest-numbered vertices ended
        // up with no nonzero coupling at all, which drops them from the
        // instance. Their side is arbitrary either way, so the optimum is
        // unchanged; `planted` keeps its full length and stays index-aligned.
        debug_assert!(
            problem.graph.len() <= n,
            "planted instance grew past its own vertex count"
        );
        // Store what the instance itself reports, so `optimum` is always a
        // value a run can actually reach; `verify` is what confirms it also
        // equals the exact one.
        let optimum = problem.calculate_cut_size(&planted);
        Self {
            problem,
            planted,
            optimum,
        }
    }
}

/// Draws the four coupler values of one plaquette, in the edge order
/// `(0,1), (0,2), (1,3), (2,3)`.
///
/// The class index `c` is the number of ground states modulo spin inversion.
/// All four edges start strong and ferromagnetic; `c` of them are weakened to
/// `+1`, and one of those weak ones is flipped antiferromagnetic. The single
/// negative coupler makes the 4-cycle frustrated, so some edge must be
/// unsatisfied, and the cheapest choice always costs `2` — which the aligned
/// state achieves by leaving the `-1` edge unsatisfied. That is why the aligned
/// state is a ground state for every class.
fn sample_plaquette(p: TileProbs2d, rng: &mut impl Rng) -> [f64; 4] {
    let r: f64 = rng.random();
    let class = if r < p.p1 {
        1
    } else if r < p.p1 + p.p2 {
        2
    } else if r < p.p1 + p.p2 + p.p3 {
        3
    } else {
        4
    };

    let mut order = [0usize, 1, 2, 3];
    order.shuffle(rng);
    let mut j = [2.0f64; 4];
    for &e in &order[..class] {
        j[e] = 1.0;
    }
    j[order[0]] = -1.0;
    j
}

/// Draws the twelve coupler values of one voxel as `(a, b, J)` over local
/// indices `0..8`, where index `dm + 2 dn + 4 dk` names a cube corner.
///
/// Every edge of the cube starts ferromagnetic; the class flips two or three of
/// them. The reference implementation fixes the sub-class probabilities
/// `pC21 = pC41 = 0`, so only C2,2, C4,2 and C6,1 ever occur, and that is
/// mirrored here. The voxel is then relabelled by a random element of the
/// octahedral group, which is what spreads the flipped edges over all
/// orientations.
fn sample_voxel(p: TileProbs3d, rng: &mut impl Rng) -> Vec<(usize, usize, f64)> {
    /// The twelve edges of a unit cube in the `dm + 2 dn + 4 dk` labelling.
    const CUBE_EDGES: [(usize, usize); 12] = [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ];

    let r: f64 = rng.random();
    let flipped: &[(usize, usize)] = if r < p.p_2fp {
        // C2,2 — opposite ferromagnetic bonds on the same face. Two bonds
        // broken in the ground state, 4 ground states modulo inversion.
        &[(0, 1), (2, 3)]
    } else if r < p.p_2fp + p.p_4fp {
        // C4,2 — diagonally opposite bonds. Two broken, 2 ground states.
        &[(0, 1), (6, 7)]
    } else {
        // C6,1 — the only 6-frustrated-facet element. Three broken, 8 ground
        // states. The reference implementation reports this as the hardest.
        &[(0, 1), (2, 6), (5, 7)]
    };

    let relabel = random_octahedral(rng);
    CUBE_EDGES
        .iter()
        .map(|&(a, b)| {
            let j = if flipped.contains(&(a, b)) { -1.0 } else { 1.0 };
            (relabel[a], relabel[b], j)
        })
        .collect()
}

/// Draws a uniformly random element of the octahedral group `Oh` as a
/// relabelling of the eight cube corners.
///
/// The 48 elements are exactly the 6 permutations of the coordinate axes times
/// the 8 independent axis flips, which is the same group and the same uniform
/// measure as the reference implementation's "24 rotations then optional
/// inversion" — only the way a draw is spelled differs.
fn random_octahedral(rng: &mut impl Rng) -> [usize; 8] {
    let mut axes = [0usize, 1, 2];
    axes.shuffle(rng);
    let flip: [bool; 3] = [rng.random(), rng.random(), rng.random()];

    let mut relabel = [0usize; 8];
    for (corner, slot) in relabel.iter_mut().enumerate() {
        *slot = (0..3)
            .map(|a| {
                let bit = (corner >> axes[a]) & 1 == 1;
                usize::from(bit != flip[a]) << a
            })
            .sum();
    }
    relabel
}

/// One standard normal variate by the Box–Muller transform.
///
/// `rand_distr` is not a dependency, and the second variate of the pair is not
/// worth the state needed to keep it.
fn standard_normal(rng: &mut impl Rng) -> f64 {
    // Drawing from (0, 1] rather than [0, 1) keeps the logarithm finite.
    let u1: f64 = 1.0 - rng.random::<f64>();
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::seeded_rng;
    use std::collections::HashSet;

    /// Best cut over every assignment. Only usable for tiny `n`.
    fn brute_force_max_cut(problem: &MaxCut, n: usize) -> f32 {
        (0..(1u64 << n))
            .map(|mask| {
                let cut: Vec<bool> = (0..n).map(|i| mask >> i & 1 == 1).collect();
                problem.calculate_cut_size(&cut)
            })
            .fold(f32::NEG_INFINITY, f32::max)
    }

    #[test]
    fn tile_planting_2d_plants_the_true_optimum() {
        // n = 16 is the smallest legal lattice and the largest we can enumerate
        // comfortably. Sweep the class mixture so every plaquette class occurs.
        let mixtures = [
            TileProbs2d::new(1.0, 0.0, 0.0),
            TileProbs2d::new(0.0, 1.0, 0.0),
            TileProbs2d::new(0.0, 0.0, 1.0),
            TileProbs2d::new(0.0, 0.0, 0.0),
            TileProbs2d::new(0.25, 0.25, 0.25),
        ];
        for (t, p) in mixtures.into_iter().enumerate() {
            for seed in 0..4 {
                let mut rng = seeded_rng(1000 + seed);
                let planted = PlantedMaxCut::tile_planting_2d(4, p, &mut rng);
                planted.verify().unwrap();
                assert_eq!(
                    planted.optimum,
                    brute_force_max_cut(&planted.problem, 16),
                    "mixture {t}, seed {seed}: planted cut is not the maximum"
                );
            }
        }
    }

    #[test]
    fn wishart_plants_the_true_optimum() {
        for couplers in [WishartCouplers::Gaussian, WishartCouplers::Discrete] {
            for seed in 0..4 {
                let mut rng = seeded_rng(2000 + seed);
                let planted = PlantedMaxCut::wishart(14, 0.75, couplers, &mut rng);
                planted.verify().unwrap();
                let best = brute_force_max_cut(&planted.problem, 14);
                assert!(
                    (planted.optimum - best).abs() <= 1e-3 * best.abs().max(1.0),
                    "{couplers:?}, seed {seed}: planted {} but best is {best}",
                    planted.optimum
                );
            }
        }
    }

    #[test]
    fn tile_planting_3d_voxel_classes_are_minimized_by_the_aligned_state() {
        // A 3D instance is too large to enumerate, but the tiles are
        // edge-disjoint, so it is enough that the aligned state minimizes every
        // voxel in isolation. Exhaustive over 2^8 states per voxel.
        let mixtures = [
            TileProbs3d::new(1.0, 0.0),
            TileProbs3d::new(0.0, 1.0),
            TileProbs3d::new(0.0, 0.0),
        ];
        for (t, p) in mixtures.into_iter().enumerate() {
            let mut rng = seeded_rng(3000 + t as u64);
            for draw in 0..64 {
                let voxel = sample_voxel(p, &mut rng);
                let energy = |s: &dyn Fn(usize) -> f64| {
                    -voxel.iter().map(|&(a, b, j)| j * s(a) * s(b)).sum::<f64>()
                };
                let aligned = energy(&|_| 1.0);
                for mask in 0..256u32 {
                    let e = energy(&|i| if mask >> i & 1 == 1 { 1.0 } else { -1.0 });
                    assert!(
                        aligned <= e + 1e-9,
                        "mixture {t}, draw {draw}, state {mask:08b}: energy {e} \
                         below the aligned {aligned}"
                    );
                }
            }
        }
    }

    #[test]
    fn tiles_partition_the_lattice_edges() {
        // The optimality argument needs the tiles to share vertices but never
        // edges. If two tiles overlapped, the later would silently overwrite
        // the earlier's couplers and the per-tile reasoning would not compose.
        let mut rng = seeded_rng(11);
        let planted = PlantedMaxCut::tile_planting_2d(8, TileProbs2d::new(0.2, 0.5, 0.1), &mut rng);
        assert_structure_matches(&planted, &Graph::grid_torus_2d(8));

        let planted = PlantedMaxCut::tile_planting_3d(6, TileProbs3d::new(0.1, 0.5), &mut rng);
        assert_structure_matches(&planted, &Graph::grid_torus_3d(6));
    }

    /// Asserts the planted instance covers exactly the lattice's edge set.
    ///
    /// This also catches two tiles *sharing* an edge, which is the failure that
    /// would break the optimality argument while leaving the graph looking
    /// fine. The tiles emit exactly as many couplers as the lattice has edges,
    /// and `Graph::from_edges` overwrites duplicates, so an overlap would
    /// silently collapse two couplers into one and leave some lattice edge
    /// uncovered — which shows up here as a missing edge.
    fn assert_structure_matches(planted: &PlantedMaxCut, lattice: &Graph) {
        let actual: HashSet<(usize, usize)> = planted
            .problem
            .graph
            .edges()
            .map(|(i, j, _)| (i, j))
            .collect();
        let expected: HashSet<(usize, usize)> = lattice.edges().map(|(i, j, _)| (i, j)).collect();
        assert_eq!(actual.len(), expected.len(), "edge count differs");
        assert_eq!(actual, expected, "edge set differs from the lattice");
    }

    #[test]
    fn the_gauge_hides_the_planted_state() {
        // Without the gauge transformation the answer is "everything on one
        // side", which every heuristic finds immediately.
        let mut rng = seeded_rng(5);
        let planted =
            PlantedMaxCut::tile_planting_2d(20, TileProbs2d::new(0.2, 0.5, 0.1), &mut rng);
        let ones = planted.planted.iter().filter(|&&b| b).count();
        assert!(
            ones > 100 && ones < 300,
            "gauge should split the 400 vertices roughly evenly, got {ones}"
        );
    }

    #[test]
    fn the_octahedral_group_is_covered_uniformly() {
        // 48 elements, each a distinct relabelling of the cube corners.
        let mut rng = seeded_rng(13);
        let seen: HashSet<[usize; 8]> = (0..20_000).map(|_| random_octahedral(&mut rng)).collect();
        assert_eq!(seen.len(), 48, "octahedral group not fully covered");
        // Every element is a permutation of the corners.
        for element in &seen {
            assert_eq!(element.iter().copied().collect::<HashSet<_>>().len(), 8);
        }
    }

    #[test]
    fn same_seed_reproduces_the_same_instance() {
        let build = || {
            let mut rng = seeded_rng(42);
            let a = PlantedMaxCut::tile_planting_2d(6, TileProbs2d::new(0.2, 0.5, 0.1), &mut rng);
            let b = PlantedMaxCut::tile_planting_3d(4, TileProbs3d::new(0.1, 0.5), &mut rng);
            let c = PlantedMaxCut::wishart(12, 0.75, WishartCouplers::Gaussian, &mut rng);
            (
                a.optimum, b.optimum, c.optimum, a.planted, b.planted, c.planted,
            )
        };
        assert_eq!(build(), build());
    }

    #[test]
    fn uncoupled_pairs_do_not_become_zero_weight_edges() {
        // Rounding the discrete Wishart couplers lands on exactly zero often
        // enough that the first sweep hit it: at n = 48 the complete graph is
        // never actually complete. A zero-weight edge would be invisible to the
        // objective while still being counted, so those pairs must be dropped.
        let complete = 48 * 47 / 2;
        let mut any_dropped = false;
        for seed in 10_812..10_818 {
            let mut rng = seeded_rng(seed);
            let planted = PlantedMaxCut::wishart(48, 0.65, WishartCouplers::Discrete, &mut rng);
            planted.verify().unwrap();
            assert!(
                planted.problem.graph.edges().all(|(_, _, w)| w != 0.0),
                "seed {seed} kept a zero-weight edge"
            );
            any_dropped |= planted.problem.graph.num_edges() < complete;
        }
        assert!(
            any_dropped,
            "test no longer exercises the zero-coupler path — pick seeds that do"
        );
    }

    #[test]
    fn verify_rejects_couplers_too_large_for_f32() {
        // The discrete Wishart scaling grows as N³. Measured at alpha = 0.75,
        // n = 96 still round-trips and n = 128 does not — the point of the test
        // is that the failure is *reported* rather than silently handing out an
        // optimum no run can reach.
        let mut rng = seeded_rng(17);
        let planted = PlantedMaxCut::wishart(128, 0.75, WishartCouplers::Discrete, &mut rng);
        assert!(
            planted.verify().is_err(),
            "discrete couplers at n = 128 should not survive f32"
        );

        let mut rng = seeded_rng(17);
        let small = PlantedMaxCut::wishart(64, 0.75, WishartCouplers::Discrete, &mut rng);
        small.verify().expect("n = 64 should still round-trip");
    }

    #[test]
    #[should_panic(expected = "even l >= 4")]
    fn panics_on_odd_lattice_size() {
        PlantedMaxCut::tile_planting_2d(5, TileProbs2d::new(0.2, 0.5, 0.1), &mut seeded_rng(0));
    }

    #[test]
    #[should_panic(expected = "alpha in (0, 1)")]
    fn panics_on_alpha_at_the_spectral_attack() {
        PlantedMaxCut::wishart(16, 1.0, WishartCouplers::Gaussian, &mut seeded_rng(0));
    }
}
