//! Portable, seed-reproducible random graph generation.
//!
//! Each structural family (Erdős–Rényi, Barabási–Albert scale-free,
//! Watts–Strogatz small-world) is a [`Graph`] constructor. In every model,
//! vertices are `0..n` (0-based, matching [`Graph`]'s internal indexing) and the
//! generated graph is simple: no self-loops and no multi-edges. Structure and
//! weights are separate concerns: a constructor returns an unweighted graph
//! (every edge at `1.0`), and [`Graph::with_random_weights`] chains onto it to
//! draw integer weights from an inclusive `(min, max)` range.
//!
//! All randomness flows through the passed [`Rng`]. [`seeded_rng`] names
//! [`ChaCha12Rng`] explicitly rather than [`StdRng`]: `rand` guarantees `StdRng`
//! is portable only *for a fixed `rand` version* and may swap the underlying
//! algorithm in a future release, which would silently change every generated
//! instance. The `golden_output_is_stable` test fails loudly if that ever moves.
//!
//! [`StdRng`]: rand::rngs::StdRng
//!
//! # Examples
//!
//! ```
//! use optopus::common::{Graph, seeded_rng};
//!
//! let mut rng = seeded_rng(42);
//! let g = Graph::erdos_renyi(50, 0.1, &mut rng).with_random_weights((1, 10), &mut rng);
//! // Same seed -> identical graph.
//! let mut rng = seeded_rng(42);
//! let g2 = Graph::erdos_renyi(50, 0.1, &mut rng).with_random_weights((1, 10), &mut rng);
//! assert_eq!(g.num_edges(), g2.num_edges());
//! ```

use std::collections::HashSet;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use super::Graph;

/// Returns a reproducible RNG for graph generation, seeded with `seed`.
///
/// [`ChaCha12Rng`] is a named algorithm that is portable across platforms,
/// architectures and `rand` versions, so the same seed produces the same graph
/// everywhere and stays valid when dependencies are upgraded.
///
/// # Examples
///
/// ```
/// use optopus::common::{Graph, seeded_rng};
///
/// let mut rng = seeded_rng(7);
/// let a = Graph::barabasi_albert(20, 2, &mut rng);
/// // Drawing from the same stream gives an independent graph.
/// let b = Graph::barabasi_albert(20, 2, &mut rng);
/// assert_eq!(a.num_edges(), b.num_edges());
/// ```
pub fn seeded_rng(seed: u64) -> ChaCha12Rng {
    ChaCha12Rng::seed_from_u64(seed)
}

impl Graph {
    /// Generates an Erdős–Rényi `G(n, p)` graph: each of the `n(n-1)/2` vertex
    /// pairs is connected independently with probability `p`, for `p · n(n-1)/2`
    /// edges on average.
    ///
    /// Uses geometric-skip sampling (Batagelj–Brandes): instead of tossing a coin
    /// for each pair, it draws the number of pairs to *skip* from the geometric
    /// distribution induced by `p`, so the cost is O(number of edges) rather than
    /// O(n²). This keeps large sparse instances feasible: `n = 10^6, p = 10^-5`
    /// yields ~5M edges in ~5M draws.
    ///
    /// Every edge gets weight `1.0`; chain
    /// [`with_random_weights`](Self::with_random_weights) for random weights.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0` or if `p` is outside `[0, 1]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::common::{Graph, seeded_rng};
    ///
    /// let g = Graph::erdos_renyi(100, 0.05, &mut seeded_rng(1));
    /// assert!(g.num_edges() > 0);
    /// ```
    pub fn erdos_renyi(n: usize, p: f64, rng: &mut impl Rng) -> Self {
        assert!(n > 0, "ErdosRenyi requires n > 0");
        assert!(
            (0.0..=1.0).contains(&p),
            "ErdosRenyi requires p in [0, 1], got {p}"
        );

        let mut pairs = Vec::new();
        if n >= 2 && p > 0.0 {
            // `(w, v)` walks the pairs in lexicographic order with w < v, skipping
            // ahead by a geometric number of pairs each step. At p == 1 this log is
            // -inf, so every skip is 1 and the walk visits every pair (complete graph).
            let log_q = (1.0 - p).ln();
            let mut v: usize = 1;
            let mut w: i64 = -1;
            while v < n {
                let r: f64 = rng.random::<f64>();
                // (1 - r) is in (0, 1], so the log is finite and the skip non-negative.
                let skip = 1.0 + ((1.0 - r).ln() / log_q).floor();
                // Guard against a skip so large it would overflow the pair index.
                if !skip.is_finite() || skip > i64::MAX as f64 {
                    break;
                }
                w += skip as i64;
                // Carry the overflow past the end of row v into the following rows.
                while v < n && w >= v as i64 {
                    w -= v as i64;
                    v += 1;
                }
                if v < n {
                    pairs.push((w as usize, v));
                }
            }
        }
        unweighted_graph(n, pairs)
    }

    /// Generates a Barabási–Albert scale-free graph: each new vertex attaches to
    /// `m` existing vertices with probability proportional to their degree,
    /// producing a power-law (hub-heavy) degree distribution.
    ///
    /// The seed is a clique on the first `m` vertices. Preferential attachment is
    /// implemented via a `targets` multiset where each vertex appears once per
    /// incident edge; sampling uniformly from it selects a vertex with probability
    /// proportional to its degree. The result has `C(m, 2) + m(n - m)` edges.
    ///
    /// Every edge gets weight `1.0`; chain
    /// [`with_random_weights`](Self::with_random_weights) for random weights.
    ///
    /// # Panics
    ///
    /// Panics if `m == 0` or if `m >= n`.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::common::{Graph, seeded_rng};
    ///
    /// let g = Graph::barabasi_albert(100, 3, &mut seeded_rng(1));
    /// assert_eq!(g.num_edges(), 3 + 3 * (100 - 3));
    /// ```
    pub fn barabasi_albert(n: usize, m: usize, rng: &mut impl Rng) -> Self {
        assert!(m > 0, "BarabasiAlbert requires m > 0");
        assert!(m < n, "BarabasiAlbert requires m < n (m={m}, n={n})");

        // Seed clique on vertices 0..m so every early vertex has a nonzero degree.
        let mut pairs: Vec<(usize, usize)> = (0..m)
            .flat_map(|i| ((i + 1)..m).map(move |j| (i, j)))
            .collect();
        let mut targets: Vec<usize> = pairs.iter().flat_map(|&(i, j)| [i, j]).collect();

        // Each new vertex v attaches to m distinct existing vertices.
        let mut chosen: Vec<usize> = Vec::with_capacity(m);
        for v in m..n {
            chosen.clear();
            while chosen.len() < m {
                // `targets` is only empty when m == 1 (no seed clique); then every
                // existing vertex is equally good.
                let candidate = match targets.len() {
                    0 => rng.random_range(0..v),
                    len => targets[rng.random_range(0..len)],
                };
                if !chosen.contains(&candidate) {
                    chosen.push(candidate);
                }
            }
            pairs.extend(chosen.iter().map(|&u| (u, v)));
            targets.extend(chosen.iter().flat_map(|&u| [u, v]));
        }
        unweighted_graph(n, pairs)
    }

    /// Generates a Watts–Strogatz small-world graph: a ring lattice joining every
    /// vertex to its `k` nearest neighbors, with each lattice edge rewired to a
    /// random target with probability `beta`. Yields short average path lengths
    /// with high clustering, and always `n · k / 2` edges.
    ///
    /// Every edge gets weight `1.0`; chain
    /// [`with_random_weights`](Self::with_random_weights) for random weights.
    ///
    /// # Panics
    ///
    /// Panics if `k` is odd or zero, if `k >= n`, or if `beta` is outside `[0, 1]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::common::{Graph, seeded_rng};
    ///
    /// let g = Graph::watts_strogatz(100, 6, 0.2, &mut seeded_rng(1));
    /// assert_eq!(g.num_edges(), 100 * 6 / 2);
    /// ```
    pub fn watts_strogatz(n: usize, k: usize, beta: f64, rng: &mut impl Rng) -> Self {
        assert!(k > 0, "WattsStrogatz requires k > 0");
        assert!(
            k.is_multiple_of(2),
            "WattsStrogatz requires even k, got {k}"
        );
        assert!(k < n, "WattsStrogatz requires k < n (k={k}, n={n})");
        assert!(
            (0.0..=1.0).contains(&beta),
            "WattsStrogatz requires beta in [0, 1], got {beta}"
        );

        // Ring lattice: connect each vertex to its `k / 2` clockwise neighbors. Two
        // such pairs coincide only if `d + d' == n`, which `k < n` rules out, so no
        // dedup is needed here. Edges keep their *lattice* orientation
        // `(source, clockwise neighbor)`: the canonical model rewires the far
        // endpoint and keeps the source, which for the wraparound edges at the ring
        // seam is not the lower-indexed vertex.
        let half = k / 2;
        let mut pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| (1..=half).map(move |d| (i, (i + d) % n)))
            .collect();
        // Adjacency as a set of ordered pairs, to reject multi-edges cheaply.
        let mut present: HashSet<(usize, usize)> = pairs.iter().map(|&e| ordered(e)).collect();

        // Rewire each edge with probability beta, keeping the lattice source.
        for edge in pairs.iter_mut() {
            if !rng.random_bool(beta) {
                continue;
            }
            let (a, b) = *edge;
            // Attempt to move the edge (a, b) to (a, w) for a fresh random w.
            for _ in 0..(2 * n) {
                let w = rng.random_range(0..n);
                if w == a || present.contains(&ordered((a, w))) {
                    continue;
                }
                present.remove(&ordered((a, b)));
                present.insert(ordered((a, w)));
                *edge = (a, w);
                break;
            }
        }
        unweighted_graph(n, pairs)
    }

    /// Replaces every edge weight with a nonzero integer drawn uniformly from
    /// the inclusive range `weight_range = (min, max)`, leaving the structure
    /// untouched.
    ///
    /// Chains onto any graph, generated or not -- including one loaded with
    /// [`load_from_file`](Self::load_from_file). Passing `min == max` puts a
    /// fixed weight on every edge.
    ///
    /// **Zero is never drawn.** A zero-weight edge is invisible to every
    /// objective and indistinguishable from a missing edge through
    /// [`get_weight`](Self::get_weight) / [`Index`](std::ops::Index), so it
    /// would silently make the graph sparser than its structure claims. When the
    /// range spans zero, weights are drawn uniformly from `[min, max] \ {0}`.
    ///
    /// # Panics
    ///
    /// Panics if `min > max`, if the range holds no nonzero value (`(0, 0)`), or
    /// if either endpoint exceeds `±2^24` in magnitude (the largest integer
    /// exactly representable as `f32`, the weight type -- beyond it a sampled
    /// weight could round outside the requested range).
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::common::{Graph, seeded_rng};
    ///
    /// let mut rng = seeded_rng(1);
    /// let g = Graph::watts_strogatz(50, 4, 0.2, &mut rng).with_random_weights((-5, 5), &mut rng);
    /// assert!(g.edges().all(|(_, _, w)| (-5.0..=5.0).contains(&w) && w != 0.0));
    /// ```
    pub fn with_random_weights(mut self, weight_range: (i64, i64), rng: &mut impl Rng) -> Self {
        let (min, max) = weight_range;
        assert!(
            min <= max,
            "weight_range min ({min}) must be <= max ({max})"
        );
        assert!(
            (min, max) != (0, 0),
            "weight_range (0, 0) contains no nonzero weight"
        );
        assert!(
            min.unsigned_abs() <= MAX_EXACT_F32_INT && max.unsigned_abs() <= MAX_EXACT_F32_INT,
            "weight_range ({min}, {max}) exceeds the exactly f32-representable range \
             of +/-{MAX_EXACT_F32_INT}"
        );
        // Skipping zero shortens the range by one value; drawing from the
        // shortened range and shifting the non-negative half up by one keeps
        // every remaining weight equally likely.
        let spans_zero = min <= 0 && 0 <= max;

        for i in 0..self.adj.len() {
            for idx in 0..self.adj[i].len() {
                let (j, _) = self.adj[i][idx];
                // Each undirected edge is weighted once, from its lower endpoint.
                if j < i {
                    continue;
                }
                let sampled = if spans_zero {
                    let v = rng.random_range(min..=max - 1);
                    if v >= 0 { v + 1 } else { v }
                } else {
                    rng.random_range(min..=max)
                };
                // The asserts above make the cast lossless, so the weight stays
                // inside the requested range.
                let w = sampled as f32;
                self.adj[i][idx].1 = w;
                if j != i {
                    self.set_directed(j, i, w);
                }
            }
        }
        self
    }
}

/// Largest integer magnitude that `f32` represents exactly (`2^24`).
const MAX_EXACT_F32_INT: u64 = 1 << 24;

/// Builds the `n`-vertex graph carrying `pairs` as unit-weight edges.
///
/// The graph is sized to `n` even when the last vertices ended up with no edges,
/// so it still reports the model's vertex count via [`Graph::len`].
fn unweighted_graph(n: usize, pairs: Vec<(usize, usize)>) -> Graph {
    let mut g = Graph::from_edges(pairs.into_iter().map(|(i, j)| (i, j, 1.0)));
    g.ensure_capacity(n);
    g
}

/// Returns the pair sorted so that the smaller vertex comes first.
fn ordered((i, j): (usize, usize)) -> (usize, usize) {
    if i <= j { (i, j) } else { (j, i) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One small graph per model, all generated from the same `seed`.
    fn small_graphs(seed: u64) -> [(&'static str, Graph); 3] {
        [
            (
                "erdos_renyi",
                weighted(seed, |r| Graph::erdos_renyi(12, 0.25, r)),
            ),
            (
                "barabasi_albert",
                weighted(seed, |r| Graph::barabasi_albert(12, 2, r)),
            ),
            (
                "watts_strogatz",
                weighted(seed, |r| Graph::watts_strogatz(12, 4, 0.25, r)),
            ),
        ]
    }

    /// Builds a graph from `seed`, then weights it from the same RNG stream.
    fn weighted(seed: u64, build: impl FnOnce(&mut ChaCha12Rng) -> Graph) -> Graph {
        let mut rng = seeded_rng(seed);
        build(&mut rng).with_random_weights((1, 10), &mut rng)
    }

    /// Renders a graph's edges as `"i-j:w"` entries, sorted and space-separated.
    fn digest(g: &Graph) -> String {
        let mut edges: Vec<(usize, usize, f32)> = g.edges().collect();
        edges.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        edges
            .iter()
            .map(|(i, j, w)| format!("{i}-{j}:{w}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn same_seed_reproduces_the_same_graph() {
        for ((name, a), (_, b)) in small_graphs(42).into_iter().zip(small_graphs(42)) {
            assert_eq!(digest(&a), digest(&b), "not reproducible for {name}");
        }
    }

    #[test]
    fn different_seeds_differ() {
        let a = weighted(1, |r| Graph::erdos_renyi(60, 0.3, r));
        let b = weighted(2, |r| Graph::erdos_renyi(60, 0.3, r));
        assert_ne!(digest(&a), digest(&b));
    }

    #[test]
    fn erdos_renyi_edge_count_in_expected_range() {
        let n = 200;
        let p = 0.1;
        let g = Graph::erdos_renyi(n, p, &mut seeded_rng(2024));
        let expected = p * (n * (n - 1) / 2) as f64;
        let actual = g.num_edges() as f64;
        // Generous band (±40%) to keep the statistical test robust.
        assert!(
            actual > expected * 0.6 && actual < expected * 1.4,
            "edge count {actual} far from expected {expected}"
        );
    }

    #[test]
    fn barabasi_albert_edge_count_is_exact() {
        let n = 100;
        let m = 3;
        let g = Graph::barabasi_albert(n, m, &mut seeded_rng(11));
        // Seed clique C(m, 2) plus m edges per new vertex (n - m of them).
        let seed_edges = m * (m - 1) / 2;
        assert_eq!(g.num_edges(), seed_edges + m * (n - m));
    }

    #[test]
    fn watts_strogatz_edge_count_is_conserved() {
        let n = 60;
        let k = 6;
        let g = Graph::watts_strogatz(n, k, 0.5, &mut seeded_rng(99));
        // Rewiring preserves the edge count of the ring lattice: n * k / 2.
        assert_eq!(g.num_edges(), n * k / 2);
    }

    #[test]
    fn generated_graphs_keep_isolated_vertices_in_len() {
        // p is small enough that some vertex almost surely ends up edge-free.
        let n = 100;
        let g = Graph::erdos_renyi(n, 0.01, &mut seeded_rng(4));
        assert!(g.num_vertices() < n, "test needs an isolated vertex");
        assert_eq!(g.len(), n);
    }

    #[test]
    fn generated_graphs_are_unweighted_by_default() {
        let g = Graph::barabasi_albert(40, 2, &mut seeded_rng(3));
        assert!(g.edges().all(|(_, _, w)| w == 1.0));
    }

    #[test]
    fn with_random_weights_keeps_structure_and_stays_symmetric() {
        let mut rng = seeded_rng(5);
        let plain = Graph::erdos_renyi(100, 0.3, &mut rng);
        let structure: HashSet<(usize, usize)> = plain.edges().map(|(i, j, _)| (i, j)).collect();
        assert!(!structure.is_empty());

        let weighted = plain.clone().with_random_weights((3, 7), &mut rng);
        assert_eq!(
            weighted
                .edges()
                .map(|(i, j, _)| (i, j))
                .collect::<HashSet<_>>(),
            structure
        );
        assert_eq!(weighted.len(), plain.len());
        for (i, j, w) in weighted.edges() {
            assert!((3.0..=7.0).contains(&w), "weight {w} out of range");
            assert_eq!(weighted[(j, i)], w, "asymmetric weight on ({i}, {j})");
        }
    }

    #[test]
    fn zero_is_never_drawn_and_both_signs_appear() {
        let mut rng = seeded_rng(8);
        // The only nonzero weights in (-1, 1) are -1 and 1.
        let g = Graph::erdos_renyi(100, 0.3, &mut rng).with_random_weights((-1, 1), &mut rng);
        assert!(g.num_edges() > 100);
        let mut seen = HashSet::new();
        for (_, _, w) in g.edges() {
            assert!(w == -1.0 || w == 1.0, "unexpected weight {w}");
            seen.insert(w.to_bits());
        }
        assert_eq!(seen.len(), 2, "both -1 and 1 should occur");
    }

    #[test]
    fn zero_is_skipped_at_a_range_endpoint() {
        let mut rng = seeded_rng(9);
        let g = Graph::erdos_renyi(60, 0.3, &mut rng).with_random_weights((0, 3), &mut rng);
        assert!(g.edges().all(|(_, _, w)| (1.0..=3.0).contains(&w)));
    }

    #[test]
    fn fixed_weight_when_min_equals_max() {
        let mut rng = seeded_rng(3);
        let g = Graph::barabasi_albert(40, 2, &mut rng).with_random_weights((5, 5), &mut rng);
        for (_, _, w) in g.edges() {
            assert_eq!(w, 5.0);
        }
    }

    #[test]
    fn generated_graphs_are_simple() {
        for (name, g) in small_graphs(17) {
            let mut seen = HashSet::new();
            for (a, b, _) in g.edges() {
                assert!(a < b, "{name}: edge not ordered i<j: ({a}, {b})");
                assert!(seen.insert((a, b)), "{name}: duplicate edge ({a}, {b})");
            }
        }
    }

    #[test]
    fn golden_output_is_stable() {
        // Pins the exact output of each model for a fixed seed. A change here
        // means the RNG algorithm or a generator's sampling order moved, which
        // would silently invalidate every previously generated instance.
        let expected = [
            "0-8:6 1-2:1 1-5:4 2-3:5 2-5:8 2-9:3 2-10:5 2-11:7 3-4:5 3-10:9 4-5:10 4-6:9 \
             4-10:7 5-7:3 5-9:6 6-7:4 7-9:1 7-10:5 8-9:10 8-10:7 9-11:5",
            "0-1:10 0-2:6 0-6:4 0-11:2 1-2:6 1-3:3 1-4:1 1-10:6 2-3:1 2-4:7 2-5:9 2-9:5 \
             3-7:6 3-8:1 4-5:4 4-7:5 5-6:8 5-11:3 6-8:5 6-10:7 7-9:5",
            "0-1:2 0-2:9 0-3:2 0-10:10 0-11:5 1-2:7 1-3:9 2-4:4 2-7:4 3-10:6 4-5:1 4-6:1 \
             4-10:8 5-6:9 5-7:3 5-9:9 6-7:3 6-8:1 6-11:9 7-9:5 8-9:10 8-10:10 9-10:8 9-11:10",
        ];
        for ((name, g), expected) in small_graphs(42).into_iter().zip(expected) {
            assert_eq!(digest(&g), expected, "output moved for {name}");
        }
    }

    #[test]
    fn erdos_renyi_handles_large_sparse_instances() {
        // Infeasible with pair-by-pair sampling (2 * 10^10 coin tosses); the
        // geometric-skip sampler only draws once per edge.
        let n = 200_000;
        let p = 1e-5;
        let g = Graph::erdos_renyi(n, p, &mut seeded_rng(7));
        let expected = p * (n as f64) * ((n - 1) as f64) / 2.0;
        let actual = g.num_edges() as f64;
        assert!(
            actual > expected * 0.9 && actual < expected * 1.1,
            "edge count {actual} far from expected {expected}"
        );
        for (i, j, _) in g.edges() {
            assert!(i < j && j < n, "edge ({i}, {j}) out of range for n = {n}");
        }
    }

    #[test]
    fn erdos_renyi_extreme_probabilities() {
        let empty = Graph::erdos_renyi(30, 0.0, &mut seeded_rng(1));
        assert_eq!(empty.num_edges(), 0);

        // p == 1 degenerates to a skip of 1 per step, i.e. the complete graph.
        let n = 30;
        let complete = Graph::erdos_renyi(n, 1.0, &mut seeded_rng(1));
        assert_eq!(complete.num_edges(), n * (n - 1) / 2);
    }

    #[test]
    fn watts_strogatz_rewiring_keeps_the_lattice_source() {
        // With k = 2 every vertex is the source of exactly one lattice edge, and
        // rewiring preserves the source -- so no vertex can be left isolated,
        // even at the ring seam where the source is the *higher* index.
        let n = 200;
        let g = Graph::watts_strogatz(n, 2, 1.0, &mut seeded_rng(5));
        for v in 0..n {
            assert!(g.degree(v) > 0, "vertex {v} left isolated by rewiring");
        }
    }

    #[test]
    #[should_panic(expected = "f32-representable")]
    fn panics_on_weight_range_beyond_f32_exactness() {
        let mut rng = seeded_rng(0);
        Graph::erdos_renyi(10, 0.1, &mut rng).with_random_weights((1, (1 << 24) + 1), &mut rng);
    }

    #[test]
    #[should_panic(expected = "weight_range")]
    fn panics_on_inverted_weight_range() {
        let mut rng = seeded_rng(0);
        Graph::erdos_renyi(10, 0.1, &mut rng).with_random_weights((5, 1), &mut rng);
    }

    #[test]
    #[should_panic(expected = "no nonzero weight")]
    fn panics_when_the_range_is_only_zero() {
        let mut rng = seeded_rng(0);
        Graph::erdos_renyi(10, 0.1, &mut rng).with_random_weights((0, 0), &mut rng);
    }

    #[test]
    #[should_panic(expected = "p in")]
    fn panics_on_invalid_probability() {
        Graph::erdos_renyi(10, 1.5, &mut seeded_rng(0));
    }

    #[test]
    #[should_panic(expected = "m < n")]
    fn panics_on_ba_m_too_large() {
        Graph::barabasi_albert(5, 5, &mut seeded_rng(0));
    }

    #[test]
    #[should_panic(expected = "even k")]
    fn panics_on_ws_odd_k() {
        Graph::watts_strogatz(10, 3, 0.1, &mut seeded_rng(0));
    }
}
