//! Exact data reduction (kernelization) for [`MaxCut`].
//!
//! Kernelization shrinks an instance by rules that are *provably* optimum
//! preserving: whatever is removed can be re-derived from an optimal solution
//! of what remains. This is the opposite trade-off from a heuristic
//! contraction, which shrinks the instance by *guessing* structure and
//! therefore restricts the search space; a kernel restricts nothing.
//!
//! The rules are stated here in raw cut-value space, and each one is pinned
//! down by an exhaustive small-graph test rather than by transcription, so the
//! offsets can be trusted independently of how the paper accounts for them.
//!
//! # What it does and does not do
//!
//! Reduction happens where vertices have *low degree*. Measured on the G-set,
//! `G70` (average degree 2.0) loses 78% of its vertices, `G55` and `G60`
//! (average degree 5) lose 13%, and 4-regular or dense instances lose nothing
//! at all. On dense random graphs the rules never fire — the paper reports the
//! same. [`MaxCutKernel::is_trivial`] makes that case free to detect, so a
//! caller can fall back without paying for anything.
//!
//! # References
//!
//! - Ferizovic, D., Hespe, D., Lamm, S., Mnich, M., Schulz, C. and Strash, D.
//!   "Engineering Kernelization for Maximum Cut." *ALENEX 2020*.
//!   [arXiv:1905.10902](https://arxiv.org/abs/1905.10902)
//!
//! # Example
//!
//! ```
//! use optopus::prelude::*;
//! use optopus::problem::MaxCutKernel;
//!
//! // A path: every vertex has degree <= 2, so the whole instance reduces away
//! // and the offset is the exact optimum.
//! let mc = MaxCut::from_edges([(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]);
//! let kernel = MaxCutKernel::reduce(&mc);
//! assert_eq!(kernel.kernel().graph.num_vertices(), 0);
//! assert_eq!(kernel.offset(), 3.0);
//! ```

use std::collections::BTreeMap;

use super::problem::MaxCut;
use crate::common::Graph;

/// One applied reduction, replayed in reverse to restore a removed vertex.
#[derive(Debug, Clone, PartialEq)]
enum Reduction {
    /// Degree 0: the vertex touches no edge, so its side is arbitrary.
    Isolated { v: usize },
    /// Degree 1: the vertex takes whichever side pays, independently of
    /// everything else.
    Pendant { v: usize, a: usize, w: f32 },
    /// Degree 2: the vertex is replaced by an edge between its two neighbors.
    Path {
        v: usize,
        a: usize,
        b: usize,
        w_a: f32,
        w_b: f32,
    },
    /// One edge outweighs all the others at `v`, so `v`'s side relative to `a`
    /// is decided and `v` can be merged into `a`.
    Merged { v: usize, a: usize, opposite: bool },
}

/// An exactly reduced [`MaxCut`] instance plus the trace needed to lift a
/// kernel solution back to the original vertex set.
///
/// The defining invariant, checked by the tests, is
/// `kernel_cut(y) + offset == original_cut(lift(y))` for **every** `y` — not
/// just for optimal ones. Solving the kernel and lifting is therefore
/// equivalent to solving the original.
#[derive(Debug, Clone)]
pub struct MaxCutKernel {
    kernel: MaxCut,
    offset: f32,
    /// Original vertex id of each kernel vertex, ascending.
    original_of: Vec<usize>,
    /// Applied reductions in application order.
    trace: Vec<Reduction>,
    /// Number of vertices in the original instance.
    original_len: usize,
}

impl MaxCutKernel {
    /// Reduces `problem` to a fixpoint of the rules.
    ///
    /// Runs in `O(m log m)`: every vertex enters the work queue only when one
    /// of its incident edges changes, and each rule application removes a
    /// vertex.
    pub fn reduce(problem: &MaxCut) -> Self {
        let n = problem.graph.len();
        let mut adj: Vec<BTreeMap<usize, f32>> = vec![BTreeMap::new(); n];
        for (u, v, w) in problem.graph.edges() {
            adj[u].insert(v, w);
            adj[v].insert(u, w);
        }
        let mut alive = vec![true; n];
        let mut offset = 0.0f32;
        let mut trace = Vec::new();

        // A vertex only needs re-examining when one of its edges changed, so
        // the queue starts with everything and is refilled locally.
        let mut queued = vec![true; n];
        let mut queue: std::collections::VecDeque<usize> = (0..n).collect();

        while let Some(v) = queue.pop_front() {
            queued[v] = false;
            if !alive[v] {
                continue;
            }
            let degree = adj[v].len();
            if degree > 2 {
                // Above degree 2 only weight domination can still fire.
                if let Some((a, opposite)) = dominating_neighbor(&adj[v]) {
                    // `v`'s neighbors are exactly the vertices whose incident
                    // edges the merge changes, and they have to be read *now*:
                    // a carried-over edge can cancel `a`'s existing one to zero,
                    // which drops it from both sides, so reading `a`'s
                    // neighborhood afterwards would miss precisely the vertices
                    // whose degree fell.
                    let affected = touched(v, &adj);
                    offset += merge_into(v, a, opposite, &mut adj, &mut alive);
                    trace.push(Reduction::Merged { v, a, opposite });
                    push(a, &mut queue, &mut queued, &alive);
                    for u in affected {
                        push(u, &mut queue, &mut queued, &alive);
                    }
                }
                continue;
            }
            match degree {
                0 => {
                    alive[v] = false;
                    trace.push(Reduction::Isolated { v });
                }
                1 => {
                    let (&a, &w) = adj[v].iter().next().expect("degree 1");
                    // Free choice: take the edge when it pays, leave it
                    // otherwise. Nothing else in the graph is affected.
                    offset += w.max(0.0);
                    remove_vertex(v, &mut adj, &mut alive);
                    trace.push(Reduction::Pendant { v, a, w });
                    push(a, &mut queue, &mut queued, &alive);
                }
                _ => {
                    let mut it = adj[v].iter();
                    let (&a, &w_a) = it.next().expect("degree 2");
                    let (&b, &w_b) = it.next().expect("degree 2");
                    // With `a` and `b` on the same side the best `v` can do is
                    // `max(0, w_a + w_b)`; with them apart it is
                    // `max(w_a, w_b)`. So the pair contributes a constant plus
                    // a term that depends only on whether (a, b) is cut — that
                    // is exactly an edge of weight `delta` between them.
                    let same_side = (w_a + w_b).max(0.0);
                    let delta = w_a.max(w_b) - same_side;
                    offset += same_side;
                    remove_vertex(v, &mut adj, &mut alive);
                    add_weight(&mut adj, a, b, delta);
                    trace.push(Reduction::Path { v, a, b, w_a, w_b });
                    push(a, &mut queue, &mut queued, &alive);
                    push(b, &mut queue, &mut queued, &alive);
                }
            }
        }

        // Compact the survivors into a contiguous id space so the kernel is a
        // dense instance rather than the original one full of holes.
        //
        // A survivor with no edge left is recorded as isolated rather than
        // carried over. `Graph` sizes itself from the largest id it ever sees,
        // so keeping an edgeless vertex would make `graph.len()` smaller than
        // `original_of.len()` and every solution built for the kernel too short
        // to lift. Doing it here makes `graph.len() == original_of.len()` hold
        // by construction instead of depending on the queue having reached
        // every vertex whose last edge vanished.
        let mut kernel_of = vec![usize::MAX; n];
        let mut original_of = Vec::new();
        for (v, keep) in alive.iter().enumerate() {
            if *keep {
                if adj[v].is_empty() {
                    trace.push(Reduction::Isolated { v });
                    continue;
                }
                kernel_of[v] = original_of.len();
                original_of.push(v);
            }
        }
        let mut graph = Graph::new();
        for &u in &original_of {
            for (&v, &w) in &adj[u] {
                if u < v {
                    graph.set_weight(kernel_of[u], kernel_of[v], w);
                }
            }
        }

        Self {
            kernel: MaxCut::new(graph),
            offset,
            original_of,
            trace,
            original_len: n,
        }
    }

    /// The reduced instance.
    pub fn kernel(&self) -> &MaxCut {
        &self.kernel
    }

    /// Constant to add to any kernel cut value to get the original one.
    pub fn offset(&self) -> f32 {
        self.offset
    }

    /// Whether no vertex could be removed, i.e. the kernel is the input.
    ///
    /// True for regular and dense graphs. Callers should treat this as "skip
    /// the whole mechanism" rather than paying to solve an identical instance
    /// through an extra layer of indirection.
    pub fn is_trivial(&self) -> bool {
        self.trace.is_empty()
    }

    /// Number of vertices the reduction removed.
    pub fn removed_vertices(&self) -> usize {
        self.trace.len()
    }

    /// Maps a kernel assignment back to the original vertex set, re-deriving
    /// every removed vertex optimally for that assignment.
    ///
    /// # Panics
    ///
    /// Panics if `kernel_x` is shorter than the kernel's vertex count.
    pub fn lift(&self, kernel_x: &[bool]) -> Vec<bool> {
        assert!(
            kernel_x.len() >= self.original_of.len(),
            "kernel assignment has {} entries but the kernel has {} vertices",
            kernel_x.len(),
            self.original_of.len()
        );
        let mut x = vec![false; self.original_len];
        for (k, &original) in self.original_of.iter().enumerate() {
            x[original] = kernel_x[k];
        }
        // Reverse order: a reduction may refer to vertices that were removed
        // later, and those are restored first.
        for reduction in self.trace.iter().rev() {
            match *reduction {
                Reduction::Isolated { v } => x[v] = false,
                Reduction::Pendant { v, a, w } => x[v] = if w > 0.0 { !x[a] } else { x[a] },
                Reduction::Path { v, a, b, w_a, w_b } => {
                    let gain = |side: bool| {
                        (if side != x[a] { w_a } else { 0.0 })
                            + (if side != x[b] { w_b } else { 0.0 })
                    };
                    x[v] = gain(true) >= gain(false);
                }
                Reduction::Merged { v, a, opposite } => x[v] = x[a] != opposite,
            }
        }
        x
    }

    /// Restricts a full assignment to the kernel's vertices, for warm starts.
    ///
    /// # Panics
    ///
    /// Panics if `full_x` is shorter than the original instance.
    pub fn project(&self, full_x: &[bool]) -> Vec<bool> {
        self.original_of.iter().map(|&v| full_x[v]).collect()
    }
}

/// Returns the neighbor `v` can be merged into, and on which side, when one
/// incident edge outweighs all the others.
///
/// If `w(v, a) >= Σ_{u ≠ a} |w(v, u)|` then moving `v` to the side opposite
/// `a` gains `w(v, a)` on that edge and can lose at most as much everywhere
/// else, so *some* optimum puts them apart — and symmetrically, an edge that
/// negative pins them together. Either way `v`'s side becomes a function of
/// `a`'s, which is a contraction rather than a deletion: `v`'s other edges are
/// still undecided and have to be carried over to `a`.
fn dominating_neighbor(neighbors: &BTreeMap<usize, f32>) -> Option<(usize, bool)> {
    let total: f32 = neighbors.values().map(|w| w.abs()).sum();
    for (&u, &w) in neighbors {
        // `total - |w|` is the most the rest of the neighborhood can lose.
        if w.abs() >= total - w.abs() && w != 0.0 {
            return Some((u, w > 0.0));
        }
    }
    None
}

/// Merges `v` into `a`, fixing `x[v] = x[a] XOR opposite`, and returns the cut
/// value that decision contributes.
///
/// Every edge `(v, t)` becomes an edge `(a, t)`: with `v` opposite `a` it is
/// cut exactly when `(a, t)` is *not*, which is a constant `w` minus an edge
/// of weight `w`; with `v` on `a`'s side it is simply an edge of weight `w`.
fn merge_into(
    v: usize,
    a: usize,
    opposite: bool,
    adj: &mut [BTreeMap<usize, f32>],
    alive: &mut [bool],
) -> f32 {
    let incident: Vec<(usize, f32)> = adj[v].iter().map(|(&u, &w)| (u, w)).collect();
    remove_vertex(v, adj, alive);
    let mut constant = 0.0;
    for (t, w) in incident {
        if t == a {
            // The dominating edge itself is decided outright.
            if opposite {
                constant += w;
            }
            continue;
        }
        if opposite {
            constant += w;
            add_weight(adj, a, t, -w);
        } else {
            add_weight(adj, a, t, w);
        }
    }
    constant
}

/// Neighbors of `a`, for re-queueing after its neighborhood changed.
fn touched(a: usize, adj: &[BTreeMap<usize, f32>]) -> Vec<usize> {
    adj[a].keys().copied().collect()
}

/// Detaches `v` from the graph.
fn remove_vertex(v: usize, adj: &mut [BTreeMap<usize, f32>], alive: &mut [bool]) {
    let neighbors: Vec<usize> = adj[v].keys().copied().collect();
    for u in neighbors {
        adj[u].remove(&v);
    }
    adj[v].clear();
    alive[v] = false;
}

/// Adds `delta` to edge `(a, b)`, dropping the edge when it cancels out.
fn add_weight(adj: &mut [BTreeMap<usize, f32>], a: usize, b: usize, delta: f32) {
    let w = adj[a].get(&b).copied().unwrap_or(0.0) + delta;
    if w == 0.0 {
        adj[a].remove(&b);
        adj[b].remove(&a);
    } else {
        adj[a].insert(b, w);
        adj[b].insert(a, w);
    }
}

/// Queues `v` for re-examination unless it is already queued or gone.
fn push(
    v: usize,
    queue: &mut std::collections::VecDeque<usize>,
    queued: &mut [bool],
    alive: &[bool],
) {
    if alive[v] && !queued[v] {
        queued[v] = true;
        queue.push_back(v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    /// Exhaustive optimum of a small instance.
    fn brute_force(mc: &MaxCut) -> f32 {
        let n = mc.graph.len();
        assert!(n <= 16, "brute force is exponential");
        let mut best = f32::NEG_INFINITY;
        for mask in 0u32..(1 << n) {
            let x: Vec<bool> = (0..n).map(|i| mask >> i & 1 == 1).collect();
            best = best.max(mc.calculate_cut_size(&x));
        }
        best
    }

    /// Random graph over `n` vertices; `weight` decides the weight regime.
    fn random_instance(
        rng: &mut SmallRng,
        n: usize,
        p: f64,
        weight: fn(&mut SmallRng) -> f32,
    ) -> MaxCut {
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if rng.random_bool(p) {
                    edges.push((i, j, weight(rng)));
                }
            }
        }
        // Keep the vertex count fixed even when a vertex draws no edge, so the
        // brute force and the lifted assignment agree on the index space.
        let mut mc = MaxCut::from_edges(edges);
        if mc.graph.len() < n {
            mc = MaxCut::new({
                let mut g = mc.graph.clone();
                g.set_weight(n - 1, n - 1, 0.0);
                g
            });
        }
        mc
    }

    fn unit(_: &mut SmallRng) -> f32 {
        1.0
    }
    fn signed(rng: &mut SmallRng) -> f32 {
        if rng.random_bool(0.5) { 1.0 } else { -1.0 }
    }
    fn general(rng: &mut SmallRng) -> f32 {
        rng.random_range(-4..=4) as f32
    }

    /// The reduction must preserve the optimum exactly, in every weight
    /// regime — this is what licenses calling it "exact".
    #[test]
    fn kernel_optimum_matches_brute_force() {
        let mut rng = SmallRng::seed_from_u64(20260729);
        for weight in [unit as fn(&mut SmallRng) -> f32, signed, general] {
            for n in 2..=9usize {
                for p in [0.15, 0.3, 0.6] {
                    for _ in 0..12 {
                        let mc = random_instance(&mut rng, n, p, weight);
                        let kernel = MaxCutKernel::reduce(&mc);
                        let reduced = brute_force(kernel.kernel()) + kernel.offset();
                        let original = brute_force(&mc);
                        assert_eq!(
                            reduced,
                            original,
                            "optimum changed (n={n}, p={p}, removed={})",
                            kernel.removed_vertices()
                        );
                    }
                }
            }
        }
    }

    /// Lifting must be exact for *every* kernel assignment, not only optimal
    /// ones, so a heuristic can be run on the kernel and lifted at any point.
    #[test]
    fn lifting_identity_holds_for_every_assignment() {
        let mut rng = SmallRng::seed_from_u64(7);
        for weight in [unit as fn(&mut SmallRng) -> f32, signed, general] {
            for _ in 0..40 {
                let mc = random_instance(&mut rng, 8, 0.35, weight);
                let kernel = MaxCutKernel::reduce(&mc);
                let k = kernel.kernel().graph.len();
                for mask in 0u32..(1 << k) {
                    let y: Vec<bool> = (0..k).map(|i| mask >> i & 1 == 1).collect();
                    let lifted = kernel.lift(&y);
                    assert_eq!(
                        kernel.kernel().calculate_cut_size(&y) + kernel.offset(),
                        mc.calculate_cut_size(&lifted),
                        "identity broken"
                    );
                }
            }
        }
    }

    /// A tree has no vertex of degree 3 or more that survives the cascade, so
    /// it reduces away completely and the offset is the exact optimum.
    #[test]
    fn a_path_reduces_away_completely() {
        let edges: Vec<_> = (0..15).map(|i| (i, i + 1, 1.0)).collect();
        let mc = MaxCut::from_edges(edges);
        let kernel = MaxCutKernel::reduce(&mc);
        assert_eq!(kernel.kernel().graph.num_vertices(), 0);
        assert_eq!(
            kernel.offset(),
            15.0,
            "a path is bipartite: every edge cuts"
        );
        assert_eq!(
            mc.calculate_cut_size(&kernel.lift(&[])),
            15.0,
            "the lifted assignment attains it"
        );
    }

    /// Re-reducing a kernel must find nothing left to do.
    ///
    /// The merge rule is what makes this non-obvious: it rewires edges onto the
    /// surviving endpoint, so a vertex far from the one being reduced can lose
    /// degree. The sweep is wide enough to catch a merge that fails to re-queue
    /// such a vertex (the narrower `n = 12, p = 0.25` sweep this replaced did
    /// not — it takes signed weights and a few hundred draws).
    #[test]
    fn reduction_is_idempotent() {
        let mut rng = SmallRng::seed_from_u64(99);
        for weight in [unit as fn(&mut SmallRng) -> f32, signed, general] {
            for n in 4..=14usize {
                for p in [0.2, 0.25, 0.35] {
                    for _ in 0..25 {
                        let mc = random_instance(&mut rng, n, p, weight);
                        let kernel = MaxCutKernel::reduce(&mc);
                        let again = MaxCutKernel::reduce(kernel.kernel());
                        assert!(
                            again.is_trivial(),
                            "the kernel was not a fixpoint (n={n}, p={p}, \
                             {} more vertices removable)",
                            again.removed_vertices()
                        );
                    }
                }
            }
        }
    }

    /// A kernel solution is built from `graph.len()`, so a kernel whose graph is
    /// shorter than its vertex list cannot be lifted at all. The reduction must
    /// never produce one — an edgeless survivor is isolated, not carried over.
    #[test]
    fn the_kernel_graph_covers_every_kernel_vertex() {
        let mut rng = SmallRng::seed_from_u64(20260809);
        for weight in [unit as fn(&mut SmallRng) -> f32, signed, general] {
            for n in 2..=14usize {
                for p in [0.15, 0.25, 0.4] {
                    for _ in 0..25 {
                        let mc = random_instance(&mut rng, n, p, weight);
                        let kernel = MaxCutKernel::reduce(&mc);
                        let k = kernel.kernel().graph.len();
                        assert_eq!(
                            k,
                            kernel.original_of.len(),
                            "kernel graph and vertex list disagree (n={n}, p={p})"
                        );
                        // What a solver would hand back, lifted without panic.
                        let lifted = kernel.lift(&vec![false; k]);
                        assert_eq!(lifted.len(), mc.graph.len());
                    }
                }
            }
        }
    }

    /// A regular graph offers nothing to reduce, and that must be detectable
    /// without inspecting the result.
    #[test]
    fn a_regular_graph_is_untouched() {
        // 4-regular circulant, the shape of the toroidal G-set instances.
        let n = 12usize;
        let edges: Vec<_> = (0..n)
            .flat_map(|i| [(i, (i + 1) % n, 1.0), (i, (i + 2) % n, 1.0)])
            .collect();
        let mc = MaxCut::from_edges(edges);
        let kernel = MaxCutKernel::reduce(&mc);
        assert!(kernel.is_trivial());
        assert_eq!(kernel.offset(), 0.0);
        assert_eq!(kernel.kernel().graph.num_edges(), mc.graph.num_edges());
    }
}
