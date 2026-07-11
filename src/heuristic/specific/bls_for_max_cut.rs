use super::super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::problem::max_cut::MaxCutFlipNeighbor;
use crate::problem::{MaxCut, MaxCutSwapNeighbor};
use crate::search_state::SearchState;
use crate::trait_defs::MoveToNeighbor;
use rand::Rng;
use rand::rngs::SmallRng;

// The positive-gain index attached to `MaxCutSolution` lets the local-search
// phase skip the O(n) neighborhood scan: any improving flip must be a vertex
// with strictly positive gain, so we only need to iterate `positive_gain`.

/// Perturbation type used inside Breakout Local Search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PerturbationType {
    /// Strong perturbation: apply `l` random flip moves.
    Strong,
    /// Weak flip perturbation: run tabu search for `l` iterations.
    WeakFlip,
    /// Weak swap perturbation: apply `l` swap moves guided by the tabu map.
    WeakSwap,
    /// Plateau cluster perturbation: flip connected clusters of zero-gain
    /// vertices (objective-preserving plateau traversal).
    PlateauCluster,
    /// Plateau independent-set perturbation: flip an independent set of
    /// zero-gain vertices (objective-preserving plateau jump).
    PlateauIndependent,
}

/// The BLS machinery shared by [`BreakoutLocalSearch`] and the RL-driven
/// variant: a flat `Vec`-based tabu map plus the fast greedy descent and the
/// three perturbation operators. The tabu state written during descent is the
/// same state consumed by the weak perturbations, which is why these operators
/// live together rather than as independent heuristics.
pub(super) struct BlsOps {
    /// Tabu tenure range `(min, max)` in iterations.
    tabu_tenure: (u64, u64),
    /// Vec-based tabu map indexed by vertex ID. Value = expiry iteration (0 = not tabu).
    tabu_vec: Vec<u64>,
    /// Epoch-stamped vertex marker for the plateau operators: `mark_vec[v] ==
    /// mark_epoch` ⇔ marked. Bumping the epoch clears all marks in O(1).
    mark_vec: Vec<u32>,
    mark_epoch: u32,
    /// Scratch vertex list for the plateau operators (BFS queue / selection).
    queue: Vec<usize>,
}

impl BlsOps {
    pub(super) fn new(tabu_tenure: (u64, u64)) -> Self {
        Self {
            tabu_tenure,
            tabu_vec: Vec::new(),
            mark_vec: Vec::new(),
            mark_epoch: 0,
            queue: Vec::new(),
        }
    }

    /// Resets the tabu state (for a new episode).
    pub(super) fn clear(&mut self) {
        self.tabu_vec.fill(0);
    }

    /// Ensures `tabu_vec` is large enough for the given problem instance.
    pub(super) fn ensure_capacity(&mut self, n: usize) {
        if self.tabu_vec.len() < n {
            self.tabu_vec.resize(n, 0);
        }
    }

    /// Starts a fresh mark generation for the plateau operators, growing
    /// `mark_vec` to `n` if needed. O(1) except on epoch wrap-around.
    fn next_mark_epoch(&mut self, n: usize) {
        if self.mark_vec.len() < n {
            self.mark_vec.resize(n, 0);
        }
        self.mark_epoch = self.mark_epoch.wrapping_add(1);
        if self.mark_epoch == 0 {
            self.mark_vec.fill(0);
            self.mark_epoch = 1;
        }
    }

    /// Checks if a vertex move is enabled (not tabu).
    #[inline]
    fn is_vertex_enabled(&self, vertex: usize, iteration: u64) -> bool {
        self.tabu_vec
            .get(vertex)
            .is_none_or(|&exp| iteration >= exp)
    }

    /// Adds a vertex to the tabu vec with a tenure sampled from `rng`.
    #[inline]
    fn add_vertex_to_tabu(&mut self, vertex: usize, iteration: u64, rng: &mut SmallRng) {
        let tabu_duration = rng.random_range(self.tabu_tenure.0..=self.tabu_tenure.1);
        if vertex < self.tabu_vec.len() {
            self.tabu_vec[vertex] = iteration + tabu_duration;
        }
    }

    /// Runs greedy local search until no improving flip move exists,
    /// recording each applied move in the tabu vec.
    ///
    /// Instead of scanning all `n` flip neighbors, this iterates only over
    /// vertices currently in `solution.positive_gain` — every improving flip
    /// must have strictly positive gain, so this set is a superset of the
    /// improving moves. On G-set instances the set shrinks rapidly as the
    /// search approaches a local optimum, turning the inner loop from O(n)
    /// into effectively O(improving_moves).
    pub(super) fn descent(&mut self, state: &mut SearchState<'_, MaxCut>) -> Result<(), OptError> {
        state.solution.enable_positive_gain_index();
        loop {
            let mut best_move_option: Option<MaxCutFlipNeighbor> = None;
            for &v in state.solution.positive_gain.as_slice() {
                let g = state.solution.gain[v];
                if let Some(best) = best_move_option
                    && best.gain >= g
                {
                    continue;
                }
                best_move_option = Some(MaxCutFlipNeighbor { i: v, gain: g });
            }

            if let Some(best_move) = best_move_option {
                self.add_vertex_to_tabu(best_move.i, state.iteration, &mut state.rng);
                state.apply(&best_move)?;
            } else {
                return Ok(());
            }
        }
    }

    /// Applies the strong perturbation: `l` random flip moves.
    /// Skips `update_best` per move; caller updates best after the phase.
    ///
    /// On a graph with no edged vertices (e.g. an empty sub-MaxCut extracted by
    /// [`SubProblemBasedCrossover`](crate::heuristic::SubProblemBasedCrossover)
    /// when the parents disagree only on an independent set) there is nothing to
    /// flip, so this just advances the iteration counter — mirroring how
    /// `weak_flip_perturbation` progresses when it finds no move — so the outer
    /// stop condition still terminates instead of the sampler panicking on an
    /// empty range.
    pub(super) fn strong_perturbation(
        &mut self,
        l: u64,
        state: &mut SearchState<'_, MaxCut>,
    ) -> Result<(), OptError> {
        if state.instance.graph.vertices.is_empty() {
            for _ in 0..l {
                state.progress_iteration();
            }
            return Ok(());
        }
        for _ in 0..l {
            let neighbor = MaxCutFlipNeighbor::random_neighbor(
                state.instance,
                &state.solution,
                &mut state.rng,
            );

            self.add_vertex_to_tabu(neighbor.i, state.iteration, &mut state.rng);
            state.apply_move_only(&neighbor)?;
        }
        Ok(())
    }

    /// Applies the weak flip perturbation: inline tabu search for `l` iterations.
    ///
    /// Uses the BLS tabu map directly and scalar best tracking to avoid the
    /// overhead of constructing a `TabuSearch` object and its per-iteration
    /// `filter_best` Vec allocation.
    pub(super) fn weak_flip_perturbation(
        &mut self,
        l: u64,
        state: &mut SearchState<'_, MaxCut>,
    ) -> Result<(), OptError> {
        let end_iter = state.iteration + l;
        while state.iteration < end_iter {
            let mut best: Option<MaxCutFlipNeighbor> = None;
            for neighbor in MaxCutFlipNeighbor::iter(state.instance, &state.solution) {
                let enabled = self.is_vertex_enabled(neighbor.i, state.iteration);
                // Aspiration: accept a tabu move if it improves the global best.
                if !enabled
                    && neighbor.gain + state.solution.objective <= state.best_solution.objective
                {
                    continue;
                }
                if let Some(ref b) = best
                    && b.gain >= neighbor.gain
                {
                    continue;
                }
                best = Some(neighbor);
            }
            if let Some(best_move) = best {
                self.add_vertex_to_tabu(best_move.i, state.iteration, &mut state.rng);
                state.apply_move_only(&best_move)?;
            } else {
                state.progress_iteration();
            }
        }
        Ok(())
    }

    /// Applies the weak swap perturbation: `l` swap moves guided by the tabu map.
    ///
    /// Uses scalar best tracking per partition side instead of collecting
    /// tied-best lists into Vecs. Also tracks the oldest-tabu vertex per side
    /// during the same scan for the fallback path.
    pub(super) fn weak_swap_perturbation(
        &mut self,
        l: u64,
        state: &mut SearchState<'_, MaxCut>,
    ) -> Result<(), OptError> {
        for _ in 0..l {
            // Best vertex per partition side (by flip gain).
            let mut best_v0: Option<MaxCutFlipNeighbor> = None;
            let mut best_v1: Option<MaxCutFlipNeighbor> = None;
            // Oldest-tabu vertex per side for the fallback path (vertex, tabu_expiry).
            let mut oldest_tabu_v0: Option<(usize, u64)> = None;
            let mut oldest_tabu_v1: Option<(usize, u64)> = None;

            for neighbor in MaxCutFlipNeighbor::iter(state.instance, &state.solution) {
                let on_side0 = state.solution.x[neighbor.i];

                // Track best vertex per side (regardless of tabu status).
                let best_ref = if on_side0 { &mut best_v0 } else { &mut best_v1 };
                if best_ref.as_ref().is_none_or(|b| neighbor.gain > b.gain) {
                    *best_ref = Some(neighbor);
                }

                // Track oldest-tabu (smallest expiry) vertex per side for fallback.
                if !self.is_vertex_enabled(neighbor.i, state.iteration) {
                    let expiry = self.tabu_vec[neighbor.i];
                    let oldest_ref = if on_side0 {
                        &mut oldest_tabu_v0
                    } else {
                        &mut oldest_tabu_v1
                    };
                    if oldest_ref.as_ref().is_none_or(|&(_, e)| expiry < e) {
                        *oldest_ref = Some((neighbor.i, expiry));
                    }
                }
            }

            // Build the swap from the best vertex on each side.
            let (v0, v1) = match (best_v0, best_v1) {
                (Some(b0), Some(b1)) => (b0, b1),
                _ => {
                    // One side is empty — skip this swap iteration.
                    state.progress_iteration();
                    state.progress_iteration();
                    continue;
                }
            };

            let swap_gain = state.solution.gain[v0.i]
                + state.solution.gain[v1.i]
                + 2.0 * state.instance.graph.get_weight(v0.i, v1.i);

            // Aspiration: accept the best swap if it improves the global best.
            if swap_gain + state.solution.objective > state.best_solution.objective {
                let swap = MaxCutSwapNeighbor {
                    i: v0.i,
                    j: v1.i,
                    gain: swap_gain,
                };
                self.add_vertex_to_tabu(v0.i, state.iteration, &mut state.rng);
                self.add_vertex_to_tabu(v1.i, state.iteration, &mut state.rng);
                state.apply_move_only(&swap)?;
            } else {
                // Fallback: swap the oldest-tabu vertex on each side to force a
                // diversifying (breakout) move. If a side has no tabu vertex yet
                // (common early in the search, when directed swaps run at
                // `omega == 0`), fall back to its best vertex so the perturbation
                // still makes progress instead of failing.
                let i = oldest_tabu_v0.map_or(v0.i, |(v, _)| v);
                let j = oldest_tabu_v1.map_or(v1.i, |(v, _)| v);
                let fallback_gain = state.solution.gain[i]
                    + state.solution.gain[j]
                    + 2.0 * state.instance.graph.get_weight(i, j);
                let swap = MaxCutSwapNeighbor {
                    i,
                    j,
                    gain: fallback_gain,
                };
                self.add_vertex_to_tabu(i, state.iteration, &mut state.rng);
                self.add_vertex_to_tabu(j, state.iteration, &mut state.rng);
                state.apply_move_only(&swap)?;
            }
        }
        Ok(())
    }

    /// Applies the plateau-cluster perturbation: grows connected clusters of
    /// zero-gain vertices via BFS and flips them one by one, so the objective
    /// value is unchanged ("iso-site" cluster moves in the Ising-machine
    /// literature — the mechanism behind the recent G-set best-known updates).
    ///
    /// Each vertex is re-checked to still have exactly zero gain at flip time,
    /// because earlier flips in the same cluster change neighbouring gains.
    /// Flipped vertices are added to the tabu map so the following descent and
    /// weak perturbations do not immediately undo the traversal. When fewer
    /// than `l` zero-gain flips are available, the remaining budget falls back
    /// to [`strong_perturbation`](Self::strong_perturbation) so the requested
    /// perturbation strength stays meaningful.
    pub(super) fn plateau_cluster_perturbation(
        &mut self,
        l: u64,
        state: &mut SearchState<'_, MaxCut>,
    ) -> Result<(), OptError> {
        state.solution.enable_zero_gain_index();
        self.next_mark_epoch(state.instance.graph.len());

        let mut flips: u64 = 0;
        // Bound the reseed attempts so the loop terminates even when every
        // remaining zero-gain vertex is already marked.
        let mut attempts = 4 * l;
        while flips < l && attempts > 0 {
            attempts -= 1;
            let members = state.solution.zero_gain.as_slice();
            if members.is_empty() {
                break;
            }
            let seed = members[state.rng.random_range(0..members.len())];
            if self.mark_vec[seed] == self.mark_epoch {
                continue;
            }
            self.queue.clear();
            self.queue.push(seed);
            self.mark_vec[seed] = self.mark_epoch;
            let mut head = 0;
            while head < self.queue.len() && flips < l {
                let v = self.queue[head];
                head += 1;
                // Earlier flips in this cluster may have moved v off the
                // plateau; only flip while its gain is still exactly zero.
                if state.solution.gain[v] == 0.0 {
                    self.add_vertex_to_tabu(v, state.iteration, &mut state.rng);
                    state.apply_move_only(&MaxCutFlipNeighbor { i: v, gain: 0.0 })?;
                    flips += 1;
                }
                for &(j, _) in state.instance.graph.iter_on_adjacency(v) {
                    if self.mark_vec[j] != self.mark_epoch && state.solution.zero_gain.contains(j) {
                        self.mark_vec[j] = self.mark_epoch;
                        self.queue.push(j);
                    }
                }
            }
        }

        if flips < l {
            self.strong_perturbation(l - flips, state)?;
        }
        Ok(())
    }

    /// Applies the plateau independent-set perturbation: samples zero-gain
    /// vertices that are pairwise non-adjacent and flips them all. Independence
    /// guarantees each selected vertex's gain is untouched by the other
    /// selected flips, so every flip has exactly zero gain and the objective
    /// value is unchanged.
    ///
    /// Compared to [`plateau_cluster_perturbation`](Self::plateau_cluster_perturbation)
    /// this scatters the plateau move across the graph instead of walking one
    /// region. Falls back to [`strong_perturbation`](Self::strong_perturbation)
    /// for any unused budget.
    pub(super) fn plateau_independent_perturbation(
        &mut self,
        l: u64,
        state: &mut SearchState<'_, MaxCut>,
    ) -> Result<(), OptError> {
        state.solution.enable_zero_gain_index();
        self.next_mark_epoch(state.instance.graph.len());

        // Select an independent set of zero-gain vertices. Marks cover the
        // selected vertices and their neighbourhoods, so a marked candidate is
        // ineligible.
        self.queue.clear();
        let mut attempts = 4 * l;
        while (self.queue.len() as u64) < l && attempts > 0 {
            attempts -= 1;
            let members = state.solution.zero_gain.as_slice();
            if members.is_empty() {
                break;
            }
            let v = members[state.rng.random_range(0..members.len())];
            if self.mark_vec[v] == self.mark_epoch {
                continue;
            }
            self.mark_vec[v] = self.mark_epoch;
            for &(j, _) in state.instance.graph.iter_on_adjacency(v) {
                self.mark_vec[j] = self.mark_epoch;
            }
            self.queue.push(v);
        }

        #[cfg(debug_assertions)]
        let objective_before = state.solution.objective;
        let selected = self.queue.len() as u64;
        for idx in 0..self.queue.len() {
            let v = self.queue[idx];
            debug_assert_eq!(
                state.solution.gain[v], 0.0,
                "independence must keep gains zero"
            );
            self.add_vertex_to_tabu(v, state.iteration, &mut state.rng);
            state.apply_move_only(&MaxCutFlipNeighbor { i: v, gain: 0.0 })?;
        }
        #[cfg(debug_assertions)]
        debug_assert_eq!(objective_before, state.solution.objective);

        if selected < l {
            self.strong_perturbation(l - selected, state)?;
        }
        Ok(())
    }

    /// Applies the given perturbation type with strength `l`.
    pub(super) fn perturb(
        &mut self,
        perturbation_type: PerturbationType,
        l: u64,
        state: &mut SearchState<'_, MaxCut>,
    ) -> Result<(), OptError> {
        match perturbation_type {
            PerturbationType::Strong => self.strong_perturbation(l, state),
            PerturbationType::WeakFlip => self.weak_flip_perturbation(l, state),
            PerturbationType::WeakSwap => self.weak_swap_perturbation(l, state),
            PerturbationType::PlateauCluster => self.plateau_cluster_perturbation(l, state),
            PerturbationType::PlateauIndependent => self.plateau_independent_perturbation(l, state),
        }
    }
}

/// Breakout Local Search (BLS) for the MaxCut problem.
///
/// BLS alternates between a greedy local search phase (with tabu updates) and a
/// perturbation phase. The perturbation type is chosen probabilistically based on
/// the `omega` counter (number of consecutive non-improving iterations). With
/// `p = max(exp(−omega / t), p0)` the probability of a **weak** (directed)
/// perturbation:
///
/// - `omega == 0` (after an improvement, so `p = 1`): always a **weak**
///   perturbation — `flip` with probability `q`, `swap` with probability `1 − q` —
///   to gently exploit the freshly found region.
/// - `0 < omega <= t` (stuck): **weak** perturbation with probability `p * q`
///   (flip) or `p * (1 − q)` (swap), and **strong** (random) otherwise; `p`
///   decays toward `p0` as `omega` grows, so strong perturbations become more
///   likely.
/// - `omega > t`: the strongest **random** perturbation is forced and `omega`
///   is reset to 0.
///
/// The perturbation length `l` increases by 1 each time the solution does not change,
/// resetting to `l0` whenever the solution changes.
///
/// # References
///
/// - Benlic, U. and Hao, J.-K. "Breakout Local Search for the Max-Cut problem." *Engineering
///   Applications of Artificial Intelligence*, 26(3), 1162-1173, 2013.
///   [DOI](https://doi.org/10.1016/j.engappai.2012.09.001)
///
/// # Parameters
///
/// - `tabu_tenure` — tabu tenure range `(min, max)` in iterations
/// - `t` — period of the `omega` counter before it resets
/// - `l0` — initial perturbation length
/// - `p0` — minimum perturbation probability
/// - `q` — fraction of weak perturbations that use the flip strategy (vs. swap)
/// - `plateau_prob` — probability that a weak perturbation flips a connected
///   cluster of zero-gain vertices instead (objective-preserving plateau
///   traversal; useful on large sparse instances with wide plateaus). `0.0`
///   reproduces the original Benlic & Hao scheme exactly.
pub struct BreakoutLocalSearch {
    ops: BlsOps,
    stop_condition: StopCondition,
    t: u64,
    l0: u64,
    p0: f64,
    q: f64,
    plateau_prob: f64,
    omega: u64,
    l: u64,
    prev_best_objective: Option<f32>,
    /// Objective value of the previous solution, used for cheap change detection
    /// instead of cloning the entire solution.
    prev_solution_objective: Option<f32>,
}

impl BreakoutLocalSearch {
    /// # Panics
    ///
    /// Panics if `plateau_prob` is not within `[0.0, 1.0]`.
    pub fn new(
        stop_condition: StopCondition,
        tabu_tenure: (u64, u64),
        t: u64,
        l0: u64,
        p0: f64,
        q: f64,
        plateau_prob: f64,
    ) -> Self {
        assert!(
            (0.0..=1.0).contains(&plateau_prob),
            "plateau_prob must be within [0.0, 1.0], got {plateau_prob}"
        );
        Self {
            ops: BlsOps::new(tabu_tenure),
            stop_condition,
            t,
            l0,
            p0,
            q,
            plateau_prob,
            omega: 0,
            l: l0,
            prev_best_objective: None,
            prev_solution_objective: None,
        }
    }

    /// Updates the `omega` counter based on whether the best objective improved.
    fn update_omega(&mut self, state: &SearchState<'_, MaxCut>) {
        if let Some(prev_best_objective) = self.prev_best_objective
            && prev_best_objective >= state.solution.objective
        {
            self.omega += 1;
        } else {
            self.omega = 0;
        }

        self.prev_best_objective = Some(state.best_solution.objective);
    }

    /// Updates the perturbation length `l` based on whether the solution changed.
    /// Uses objective comparison instead of full solution clone for O(1) check.
    fn update_l(&mut self, state: &SearchState<'_, MaxCut>) {
        if let Some(prev_obj) = self.prev_solution_objective
            && prev_obj == state.solution.objective
        {
            self.l += 1;
        } else {
            self.l = self.l0;
        }

        self.prev_solution_objective = Some(state.solution.objective);
    }

    /// Determines the perturbation type for the current iteration.
    ///
    /// Follows the adaptive scheme of Benlic & Hao: the probability of a
    /// directed (weak) perturbation is `p = max(exp(−omega / t), p0)`, so when
    /// the search just improved (`omega == 0`, hence `p = 1`) it always applies
    /// a directed perturbation to gently exploit the freshly found region, and
    /// as `omega` grows the random (strong) perturbation becomes more likely.
    /// Only when `omega` exceeds the threshold `t` is the strongest random
    /// perturbation forced (and `omega` reset).
    fn get_perturbation_type(&mut self, rng: &mut SmallRng) -> PerturbationType {
        if self.omega > self.t {
            self.omega = 0;
            return PerturbationType::Strong;
        }

        let p = (-(self.omega as f64 / self.t as f64)).exp().max(self.p0);

        let prob: f64 = rng.random_range(0.0..=1.0);
        if prob <= p {
            // Weak (directed) branch: optionally traverse the plateau instead.
            // The extra draw only happens when plateau_prob > 0, so runs with
            // plateau_prob == 0.0 consume the same RNG stream as before.
            if self.plateau_prob > 0.0 && rng.random_range(0.0..=1.0) <= self.plateau_prob {
                return PerturbationType::PlateauCluster;
            }
            if prob <= p * self.q {
                PerturbationType::WeakFlip
            } else {
                PerturbationType::WeakSwap
            }
        } else {
            PerturbationType::Strong
        }
    }
}

impl Heuristic<MaxCut> for BreakoutLocalSearch {
    fn clear(&mut self) {
        self.omega = 0;
        self.l = self.l0;
        self.prev_best_objective = None;
        self.prev_solution_objective = None;
        self.ops.clear();
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, MaxCut>) -> Result<(), OptError> {
        self.ops.ensure_capacity(state.instance.graph.len());

        tracing::debug!(
            iteration = state.iteration,
            omega = self.omega,
            l = self.l,
            "BLS: local search phase start"
        );

        self.ops.descent(state)?;

        self.update_omega(state);
        self.update_l(state);

        let perturbation_type = self.get_perturbation_type(&mut state.rng);
        tracing::debug!(
            iteration = state.iteration,
            omega = self.omega,
            l = self.l,
            perturbation = ?perturbation_type,
            "BLS: perturbation selected"
        );

        self.ops.perturb(perturbation_type, self.l, state)?;

        // Update best once after the perturbation phase completes.
        state.update_best();

        Ok(())
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::Heuristic;
    use crate::problem::MaxCut;
    use crate::search_state::SearchState;

    /// Builds a small toroidal-like graph (degree 4, unit weights) that has both
    /// partition sides populated throughout the search.
    fn small_instance() -> MaxCut {
        let n = 30usize;
        let mut edges = Vec::new();
        for i in 0..n {
            edges.push((i, (i + 1) % n, 1.0));
            edges.push((i, (i + 2) % n, 1.0));
        }
        MaxCut::from_edges(edges)
    }

    /// Regression test: BLS must run to completion without erroring.
    ///
    /// The weak-swap perturbation previously returned `Err("No tabu v1")` when a
    /// partition side had no tabu vertex yet — a path that is hit frequently now
    /// that directed (weak) perturbations run at `omega == 0`. Running the full
    /// loop many times exercises all three perturbation types and the swap
    /// fallback; it must never error and must find a non-trivial cut.
    #[test]
    fn bls_runs_without_error_and_improves() {
        let mc = small_instance();
        for _ in 0..10 {
            let mut state = SearchState::new(&mc);
            let mut bls = BreakoutLocalSearch::new(
                StopCondition::iterations(5_000),
                (3, 15),
                1_000,
                5,
                0.8,
                0.5,
                0.0,
            );
            bls.run(&mut state).expect("BLS must not error");
            assert!(
                state.best_solution.objective > 0.0,
                "BLS should find a positive cut, got {}",
                state.best_solution.objective
            );
        }
    }

    /// BLS with plateau perturbations enabled must also run cleanly and find a
    /// non-trivial cut (exercises PlateauCluster through the full loop).
    #[test]
    fn bls_with_plateau_runs_without_error_and_improves() {
        let mc = small_instance();
        for seed in 0..10 {
            let mut state = SearchState::new_with_seed(&mc, seed);
            let mut bls = BreakoutLocalSearch::new(
                StopCondition::iterations(5_000),
                (3, 15),
                1_000,
                5,
                0.8,
                0.5,
                0.5,
            );
            bls.run(&mut state).expect("BLS+plateau must not error");
            assert!(state.best_solution.objective > 0.0);
        }
    }

    #[test]
    #[should_panic(expected = "plateau_prob must be within [0.0, 1.0]")]
    fn bls_rejects_invalid_plateau_prob() {
        let _ = BreakoutLocalSearch::new(
            StopCondition::iterations(1),
            (3, 15),
            1_000,
            5,
            0.8,
            0.5,
            1.5,
        );
    }

    /// Descends to a local optimum first so that the plateau (zero-gain set)
    /// is non-trivial, mirroring where the operators run in the real loop.
    fn descended_state<'a>(mc: &'a MaxCut, seed: u64, ops: &mut BlsOps) -> SearchState<'a, MaxCut> {
        let mut state = SearchState::new_with_seed(mc, seed);
        ops.ensure_capacity(mc.graph.len());
        ops.descent(&mut state).unwrap();
        state
    }

    /// The plateau-cluster perturbation must preserve the objective value
    /// bit-for-bit whenever the zero-gain set is non-empty (every applied flip
    /// has exactly zero gain), and it must actually move the solution.
    #[test]
    fn plateau_cluster_preserves_objective() {
        let mc = small_instance();
        let mut moved = false;
        for seed in 0..20 {
            let mut ops = BlsOps::new((3, 15));
            let mut state = descended_state(&mc, seed, &mut ops);
            state.solution.enable_zero_gain_index();
            if state.solution.zero_gain_count() == 0 {
                continue;
            }
            let objective_before = state.solution.objective;
            let x_before = state.solution.x.clone();
            // l small enough that the zero-gain set can absorb the full budget,
            // so the strong fallback (which would change the objective) stays off.
            ops.plateau_cluster_perturbation(1, &mut state).unwrap();
            assert_eq!(
                state.solution.objective, objective_before,
                "plateau cluster flips must not change the objective (seed {seed})"
            );
            moved |= state.solution.x != x_before;
        }
        assert!(moved, "the operator must move the solution at least once");
    }

    /// Same objective-invariance property for the independent-set variant.
    #[test]
    fn plateau_independent_preserves_objective() {
        let mc = small_instance();
        let mut moved = false;
        for seed in 0..20 {
            let mut ops = BlsOps::new((3, 15));
            let mut state = descended_state(&mc, seed, &mut ops);
            state.solution.enable_zero_gain_index();
            if state.solution.zero_gain_count() == 0 {
                continue;
            }
            let objective_before = state.solution.objective;
            let x_before = state.solution.x.clone();
            ops.plateau_independent_perturbation(1, &mut state).unwrap();
            assert_eq!(
                state.solution.objective, objective_before,
                "independent-set plateau flips must not change the objective (seed {seed})"
            );
            moved |= state.solution.x != x_before;
        }
        assert!(moved, "the operator must move the solution at least once");
    }

    /// Property test: after hundreds of mixed perturbations of all five types,
    /// the incrementally maintained gain vector and both gain indexes must
    /// agree with a from-scratch recomputation.
    #[test]
    fn mixed_perturbations_keep_gains_and_indexes_consistent() {
        use PerturbationType::*;
        let mc = small_instance();
        let mut ops = BlsOps::new((3, 15));
        let mut state = SearchState::new_with_seed(&mc, 7);
        ops.ensure_capacity(mc.graph.len());
        state.solution.enable_positive_gain_index();
        state.solution.enable_zero_gain_index();

        let schedule = [
            Strong,
            WeakFlip,
            PlateauCluster,
            WeakSwap,
            PlateauIndependent,
        ];
        for round in 0..60 {
            for &ptype in &schedule {
                ops.perturb(ptype, 3, &mut state).unwrap();
            }
            ops.descent(&mut state).unwrap();

            for v in 0..state.solution.x.len() {
                let expected = mc.calculate_gain(&state.solution.x, v);
                assert_eq!(
                    state.solution.gain[v], expected,
                    "gain[{v}] diverged after round {round}"
                );
                assert_eq!(
                    state.solution.positive_gain.contains(v),
                    expected > 0.0,
                    "positive_gain membership of {v} wrong after round {round}"
                );
                assert_eq!(
                    state.solution.zero_gain.contains(v),
                    expected == 0.0,
                    "zero_gain membership of {v} wrong after round {round}"
                );
            }
            let expected_objective = mc.calculate_cut_size(&state.solution.x);
            assert_eq!(
                state.solution.objective, expected_objective,
                "objective diverged after round {round}"
            );
        }
    }

    /// On an instance where no zero gain can ever arise (all-distinct powers
    /// of two as weights), the plateau operators must consume the full budget
    /// via the strong fallback instead of looping forever or under-perturbing.
    #[test]
    fn plateau_falls_back_to_strong_when_no_zero_gain() {
        let mc = MaxCut::from_edges([(0, 1, 1.0), (1, 2, 2.0), (0, 2, 4.0)]);
        for seed in 0..5 {
            let mut ops = BlsOps::new((3, 15));
            let mut state = SearchState::new_with_seed(&mc, seed);
            ops.ensure_capacity(mc.graph.len());
            let iter_before = state.iteration;
            ops.plateau_cluster_perturbation(4, &mut state).unwrap();
            assert_eq!(
                state.iteration - iter_before,
                4,
                "all 4 moves must be applied via the strong fallback"
            );

            let iter_before = state.iteration;
            ops.plateau_independent_perturbation(4, &mut state).unwrap();
            assert_eq!(state.iteration - iter_before, 4);
        }
    }

    /// On a graph with no edged vertices — as produced by
    /// `SubProblemBasedCrossover` when the two parents disagree only on an
    /// independent set — the perturbations must advance iterations without
    /// panicking (`random_neighbor` samples an empty range), and a full BLS run
    /// must terminate cleanly via its stop condition.
    #[test]
    fn bls_terminates_on_edgeless_graph() {
        let mc = MaxCut::new(crate::common::Graph::new());
        // Direct operator check: strong perturbation must progress, not panic.
        let mut ops = BlsOps::new((3, 15));
        let mut state = SearchState::new_with_seed(&mc, 0);
        ops.ensure_capacity(mc.graph.len());
        let before = state.iteration;
        ops.strong_perturbation(5, &mut state).unwrap();
        assert_eq!(state.iteration - before, 5);
        ops.plateau_cluster_perturbation(5, &mut state).unwrap();
        ops.plateau_independent_perturbation(5, &mut state).unwrap();

        // Full run: the failed-updates stop condition must fire and return.
        let mut state = SearchState::new_with_seed(&mc, 0);
        let mut bls = BreakoutLocalSearch::new(
            StopCondition::iterations(10_000).with_failed_updates(500),
            (3, 15),
            1_000,
            5,
            0.8,
            0.5,
            0.5,
        );
        bls.run(&mut state)
            .expect("BLS must terminate on an edgeless graph");
    }

    /// Seeded regression guard for the selection-rule refactor: with
    /// `plateau_prob = 0.0` no extra RNG draw happens, so a seeded run must be
    /// deterministic and identical across repetitions.
    #[test]
    fn bls_plateau_prob_zero_is_deterministic() {
        let mc = small_instance();
        let run = || {
            let mut state = SearchState::new_with_seed(&mc, 42);
            let mut bls = BreakoutLocalSearch::new(
                StopCondition::iterations(3_000),
                (3, 15),
                1_000,
                5,
                0.8,
                0.5,
                0.0,
            );
            bls.run(&mut state).unwrap();
            (
                state.best_solution.objective,
                state.best_iteration,
                state.best_solution.x.clone(),
            )
        };
        assert_eq!(run(), run());
    }
}
