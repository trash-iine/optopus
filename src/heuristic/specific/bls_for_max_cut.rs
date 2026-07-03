use super::super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::problem::max_cut::MaxCutFlipNeighbor;
use crate::problem::{MaxCut, MaxCutSwapNeighbor};
use crate::search_state::{MoveToNeighbor, SearchState};

// The positive-gain index attached to `MaxCutSolution` lets the local-search
// phase skip the O(n) neighborhood scan: any improving flip must be a vertex
// with strictly positive gain, so we only need to iterate `positive_gain`.

/// Perturbation type used inside Breakout Local Search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PerturbationType {
    /// Strong perturbation: apply `l` random flip moves.
    Strong,
    /// Weak flip perturbation: run tabu search for `l` iterations.
    WeakFlip,
    /// Weak swap perturbation: apply `l` swap moves guided by the tabu map.
    WeakSwap,
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
pub struct BreakoutLocalSearch {
    tabu_tenure: (u64, u64),
    stop_condition: StopCondition,
    t: u64,
    l0: u64,
    p0: f64,
    q: f64,
    omega: u64,
    l: u64,
    prev_best_objective: Option<f32>,
    /// Objective value of the previous solution, used for cheap change detection
    /// instead of cloning the entire solution.
    prev_solution_objective: Option<f32>,
    /// Vec-based tabu map indexed by vertex ID. Value = expiry iteration (0 = not tabu).
    tabu_vec: Vec<u64>,
}

impl BreakoutLocalSearch {
    pub fn new(
        tabu_tenure: (u64, u64),
        stop_condition: StopCondition,
        t: u64,
        l0: u64,
        p0: f64,
        q: f64,
    ) -> Self {
        Self {
            tabu_tenure,
            stop_condition,
            t,
            l0,
            p0,
            q,
            omega: 0,
            l: l0,
            prev_best_objective: None,
            prev_solution_objective: None,
            tabu_vec: Vec::new(),
        }
    }

    /// Ensures `tabu_vec` is large enough for the given problem instance.
    fn ensure_tabu_vec(&mut self, n: usize) {
        if self.tabu_vec.len() < n {
            self.tabu_vec.resize(n, 0);
        }
    }

    /// Checks if a vertex move is enabled (not tabu).
    ///
    /// `>=` so that the default entry `0` never marks a vertex tabu at iteration 0.
    #[inline]
    fn is_vertex_enabled(&self, vertex: usize, iteration: u64) -> bool {
        self.tabu_vec
            .get(vertex)
            .is_none_or(|&exp| iteration >= exp)
    }

    /// Adds a vertex to the tabu vec with a random tenure.
    #[inline]
    fn add_vertex_to_tabu(&mut self, vertex: usize, iteration: u64) {
        let tabu_duration = rand::random_range(self.tabu_tenure.0..=self.tabu_tenure.1);
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
    fn local_search_with_updating_tabu(
        &mut self,
        state: &mut SearchState<'_, MaxCut>,
    ) -> Result<(), OptError> {
        state.solution.enable_positive_gain_index();
        loop {
            let mut best_move_option: Option<MaxCutFlipNeighbor> = None;
            for &v in &state.solution.positive_gain {
                let g = state.solution.gain[v];
                if let Some(best) = best_move_option
                    && best.gain >= g
                {
                    continue;
                }
                best_move_option = Some(MaxCutFlipNeighbor { i: v, gain: g });
            }

            if let Some(best_move) = best_move_option {
                self.add_vertex_to_tabu(best_move.i, state.iteration);
                state.apply(&best_move)?;
            } else {
                return Ok(());
            }
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
    fn get_perturbation_type(&mut self) -> PerturbationType {
        if self.omega > self.t {
            self.omega = 0;
            return PerturbationType::Strong;
        }

        let p = (-(self.omega as f64 / self.t as f64)).exp().max(self.p0);

        let prob: f64 = rand::random_range(0.0..=1.0);
        if prob <= p * self.q {
            PerturbationType::WeakFlip
        } else if prob <= p {
            PerturbationType::WeakSwap
        } else {
            PerturbationType::Strong
        }
    }

    /// Applies the strong perturbation: `l` random flip moves.
    /// Skips `update_best` per move; caller updates best after the phase.
    fn apply_strong_perturbation(
        &mut self,
        state: &mut SearchState<'_, MaxCut>,
    ) -> Result<(), OptError> {
        for _ in 0..self.l {
            let neighbor = MaxCutFlipNeighbor::random_neighbor(state.instance, &state.solution);

            self.add_vertex_to_tabu(neighbor.i, state.iteration);
            state.apply_move_only(&neighbor)?;
        }
        Ok(())
    }

    /// Applies the weak flip perturbation: inline tabu search for `l` iterations.
    ///
    /// Uses the BLS tabu map directly and scalar best tracking to avoid the
    /// overhead of constructing a `TabuSearch` object and its per-iteration
    /// `filter_best` Vec allocation.
    fn apply_weak_flip_perturbation(
        &mut self,
        state: &mut SearchState<'_, MaxCut>,
    ) -> Result<(), OptError> {
        let end_iter = state.iteration + self.l;
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
                self.add_vertex_to_tabu(best_move.i, state.iteration);
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
    fn apply_weak_swap_perturbation(
        &mut self,
        state: &mut SearchState<'_, MaxCut>,
    ) -> Result<(), OptError> {
        for _ in 0..self.l {
            // Best vertex per partition side (by flip gain).
            let mut best_v0: Option<MaxCutFlipNeighbor> = None;
            let mut best_v1: Option<MaxCutFlipNeighbor> = None;
            // Oldest-tabu vertex per side for the fallback path (vertex, tabu_expiry).
            let mut oldest_tabu_v0: Option<(usize, u64)> = None;
            let mut oldest_tabu_v1: Option<(usize, u64)> = None;

            for neighbor in MaxCutFlipNeighbor::iter(state.instance, &state.solution) {
                let on_side0 = state.solution.cut[neighbor.i];

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
                self.add_vertex_to_tabu(v0.i, state.iteration);
                self.add_vertex_to_tabu(v1.i, state.iteration);
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
                self.add_vertex_to_tabu(i, state.iteration);
                self.add_vertex_to_tabu(j, state.iteration);
                state.apply_move_only(&swap)?;
            }
        }
        Ok(())
    }
}

impl Heuristic<MaxCut> for BreakoutLocalSearch {
    fn clear(&mut self) {
        self.omega = 0;
        self.l = self.l0;
        self.prev_best_objective = None;
        self.prev_solution_objective = None;
        self.tabu_vec.fill(0);
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, MaxCut>) -> Result<(), OptError> {
        self.ensure_tabu_vec(state.instance.graph.len());

        tracing::debug!(
            iteration = state.iteration,
            omega = self.omega,
            l = self.l,
            "BLS: local search phase start"
        );

        self.local_search_with_updating_tabu(state)?;

        self.update_omega(state);
        self.update_l(state);

        let perturbation_type = self.get_perturbation_type();
        tracing::debug!(
            iteration = state.iteration,
            omega = self.omega,
            l = self.l,
            perturbation = ?perturbation_type,
            "BLS: perturbation selected"
        );

        match perturbation_type {
            PerturbationType::Strong => {
                self.apply_strong_perturbation(state)?;
            }
            PerturbationType::WeakFlip => {
                self.apply_weak_flip_perturbation(state)?;
            }
            PerturbationType::WeakSwap => {
                self.apply_weak_swap_perturbation(state)?;
            }
        }

        // Update best once after the perturbation phase completes.
        state.update_best();

        Ok(())
    }

    fn is_done<'a>(&self, state: &SearchState<'a, MaxCut>) -> bool {
        self.stop_condition.is_done(state)
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
                (3, 15),
                StopCondition::iterations(5_000),
                1_000,
                5,
                0.8,
                0.5,
            );
            bls.run(&mut state).expect("BLS must not error");
            assert!(
                state.best_solution.objective > 0.0,
                "BLS should find a positive cut, got {}",
                state.best_solution.objective
            );
        }
    }
}
