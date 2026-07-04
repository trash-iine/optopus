//! Search state management for combinatorial optimization.
//!
//! The core traits live in [`crate::trait_defs`] and are re-exported here for
//! backward compatibility, so `crate::search_state::ProblemTrait` and friends
//! keep resolving.

pub use crate::trait_defs::{
    Crossover, Distance, EnabledTabu, Evaluable, Evaluate, MoveToNeighbor, ProblemTrait, Rankable,
    SubProblemExtractable, filter_best,
};
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// Controls how [`SearchState`] is cloned when starting a sub-run.
#[derive(Clone, Debug)]
pub enum SearchStateCloneType {
    /// Clone the state as-is.
    ///
    /// - Starts from the current solution
    /// - Retains the original start time and iteration counter
    /// - Retains the current best solution
    Simple,

    /// Clone the state and reset all best-solution tracking.
    ///
    /// - Starts from the current solution
    /// - Resets start time and iteration counter to zero
    /// - Sets the best solution to the current solution
    ClearBest,

    /// Clone the state starting from the best solution found so far.
    ///
    /// - Starts from the best solution
    /// - Resets start time and iteration counter to zero
    /// - Retains the best solution
    StartBest,
}

/// Holds the full runtime state of a heuristic search.
///
/// Contains the problem instance (by reference), the current solution,
/// the best solution found so far, and iteration / timing metadata.
///
/// # Field visibility policy
///
/// The fields below are split deliberately:
///
/// - **`pub` fields** are the live search state that heuristics (both built-in
///   and user-implemented) legitimately read **and write** during a run:
///   `solution`, `best_solution`, `iteration`, `best_iteration`, `best_time`,
///   `initial_solution`, `n_accepted`, `n_rejected`, `n_best_updates`, `rng`,
///   and the problem reference `instance`. Direct field access is intentional
///   — wrapping every one of these in a setter would only add noise without
///   strengthening any invariant, since heuristics need to mutate them
///   anyway.
///
/// - **`pub(crate)` fields** (`start_iteration`, `start_time`,
///   `start_n_accepted`, `start_n_rejected`, `start_n_best_updates`) are the
///   sub-run anchors used **only** by [`Self::clone_for_new_run`] and
///   [`Self::update_state`] to compute deltas when merging a sub-run's
///   progress back into its parent. They must never be touched from outside
///   this crate; an external write would silently corrupt the delta-merge
///   accounting.
///
/// In short: pub = "live state heuristics drive", pub(crate) = "internal
/// merge bookkeeping — hands off".
#[derive(Clone)]
pub struct SearchState<'a, Problem>
where
    Problem: ProblemTrait,
{
    /// Iteration count at the start of the current sub-run.
    pub(crate) start_iteration: u64,
    /// Wall-clock time when the current sub-run started.
    pub(crate) start_time: std::time::Instant,
    /// `n_accepted` at the start of the current sub-run (anchor for diff-merge).
    pub(crate) start_n_accepted: u64,
    /// `n_rejected` at the start of the current sub-run (anchor for diff-merge).
    pub(crate) start_n_rejected: u64,
    /// `n_best_updates` at the start of the current sub-run (anchor for diff-merge).
    pub(crate) start_n_best_updates: u64,
    /// Reference to the problem instance.
    pub instance: &'a Problem,
    /// Current iteration count.
    pub iteration: u64,
    /// Current solution.
    pub solution: Problem::Solution,
    /// Wall-clock time when the best solution was last updated.
    pub best_time: std::time::Instant,
    /// Iteration at which the best solution was last updated.
    pub best_iteration: u64,
    /// Best solution found so far.
    pub best_solution: Problem::Solution,
    /// The initial solution this sub-run started from. Updated only at
    /// construction time and when `clone_for_new_run` resets it; never
    /// touched by `apply` / `update_best` / `update_state`.
    ///
    /// Semantics across [`SearchStateCloneType`]:
    /// - `Simple`     — inherits the parent's `initial_solution`.
    /// - `ClearBest`  — re-anchored to the solution at clone time.
    /// - `StartBest`  — re-anchored to the best solution at clone time
    ///   (which is also the sub-run's starting solution).
    pub initial_solution: Problem::Solution,
    /// Number of moves accepted (`apply` / `apply_move_only` calls) since this
    /// sub-run started. Always satisfies
    /// `iteration - start_iteration == (n_accepted - start_n_accepted) + (n_rejected - start_n_rejected)`
    /// when only the standard methods on this state are used.
    pub n_accepted: u64,
    /// Number of iterations that advanced without applying a move
    /// (`progress_iteration` calls) since this sub-run started.
    pub n_rejected: u64,
    /// Number of times `update_best` actually replaced `best_solution`
    /// since this sub-run started.
    pub n_best_updates: u64,
    /// Shared random source used by every heuristic that needs randomness.
    ///
    /// Replaces ad-hoc `rand::rng()` calls; threading the RNG through this
    /// field is what makes runs reproducible from a single seed
    /// (see [`SearchState::new_with_seed`]).
    ///
    /// On `clone_for_new_run` the parent's RNG is **forked**: the child gets a
    /// fully independent stream, and the parent's stream advances by one fork.
    /// Sub-run RNG state is discarded by `update_state`, so meta-heuristic
    /// composition (Sequential / Iterated / Restart / GA) does not leak its
    /// internal RNG consumption back to the parent.
    pub rng: SmallRng,
}

impl<'a, Problem> SearchState<'a, Problem>
where
    Problem: ProblemTrait,
{
    /// Creates a new [`SearchState`] with a randomly generated initial solution,
    /// seeded from system entropy.
    pub fn new(instance: &'a Problem) -> Self {
        Self::from_rng(instance, SmallRng::from_os_rng())
    }

    /// Creates a new [`SearchState`] with a randomly generated initial solution,
    /// using a deterministic seed for full reproducibility.
    ///
    /// Given the same `seed` and `instance`, two states produce bit-identical
    /// initial solutions and (when used with seedable heuristics) bit-identical
    /// full runs.
    pub fn new_with_seed(instance: &'a Problem, seed: u64) -> Self {
        Self::from_rng(instance, SmallRng::seed_from_u64(seed))
    }

    /// Internal: construct from a fully prepared RNG.
    fn from_rng(instance: &'a Problem, mut rng: SmallRng) -> Self {
        let solution = instance.new_solution(&mut rng);
        let now = std::time::Instant::now();
        let state = Self {
            start_iteration: 0,
            start_time: now,
            start_n_accepted: 0,
            start_n_rejected: 0,
            start_n_best_updates: 0,
            instance,
            iteration: 0,
            solution: solution.clone(),
            best_time: now,
            best_iteration: 0,
            best_solution: solution.clone(),
            initial_solution: solution,
            n_accepted: 0,
            n_rejected: 0,
            n_best_updates: 0,
            rng,
        };
        tracing::debug!("SearchState initialized");
        state
    }

    /// Creates a new [`SearchState`] starting from a specific solution.
    ///
    /// Use this for warm starts, deterministic testing, or when a known-good solution
    /// should be the starting point. The provided solution is also set as the initial best
    /// and as `initial_solution`. RNG is seeded from system entropy.
    pub fn with_solution(instance: &'a Problem, solution: Problem::Solution) -> Self {
        Self::with_solution_from_rng(instance, solution, SmallRng::from_os_rng())
    }

    /// Like [`with_solution`](Self::with_solution) but with a deterministic seed.
    pub fn with_solution_and_seed(
        instance: &'a Problem,
        solution: Problem::Solution,
        seed: u64,
    ) -> Self {
        Self::with_solution_from_rng(instance, solution, SmallRng::seed_from_u64(seed))
    }

    fn with_solution_from_rng(
        instance: &'a Problem,
        solution: Problem::Solution,
        rng: SmallRng,
    ) -> Self {
        let now = std::time::Instant::now();
        Self {
            start_iteration: 0,
            start_time: now,
            start_n_accepted: 0,
            start_n_rejected: 0,
            start_n_best_updates: 0,
            instance,
            iteration: 0,
            solution: solution.clone(),
            best_time: now,
            best_iteration: 0,
            best_solution: solution.clone(),
            initial_solution: solution,
            n_accepted: 0,
            n_rejected: 0,
            n_best_updates: 0,
            rng,
        }
    }

    /// Returns the elapsed time since the current sub-run started.
    pub fn duration(&self) -> std::time::Duration {
        std::time::Instant::now() - self.start_time
    }

    /// Updates the best solution if the current solution is better.
    ///
    /// Returns `true` if the best solution was updated. Increments
    /// [`n_best_updates`](Self::n_best_updates) on each actual update.
    pub fn update_best(&mut self) -> bool {
        let ret = self.solution.is_better_than(&self.best_solution);

        if ret {
            self.best_solution = self.solution.clone();
            self.best_time = std::time::Instant::now();
            self.best_iteration = self.iteration;
            self.n_best_updates += 1;
            tracing::debug!(
                iteration = self.best_iteration,
                elapsed_secs = self.duration().as_secs_f64(),
                "Best solution updated"
            );
        }

        ret
    }

    /// Creates a copy of this state prepared for a new sub-run.
    ///
    /// The behavior depends on `clone_type`; see [`SearchStateCloneType`] for details.
    ///
    /// **RNG semantics**: the parent's RNG is *forked* — the child gets a fully
    /// independent stream, and the parent's stream advances by one fork's worth
    /// of state. This is why `&mut self` is required.
    pub fn clone_for_new_run(&mut self, clone_type: SearchStateCloneType) -> Self {
        let now = std::time::Instant::now();
        let child_rng = SmallRng::from_rng(&mut self.rng);
        match clone_type {
            SearchStateCloneType::Simple => Self {
                start_iteration: self.iteration,
                start_time: self.start_time,
                start_n_accepted: self.n_accepted,
                start_n_rejected: self.n_rejected,
                start_n_best_updates: self.n_best_updates,
                instance: self.instance,
                iteration: self.iteration,
                solution: self.solution.clone(),
                best_time: self.best_time,
                best_iteration: self.best_iteration,
                best_solution: self.best_solution.clone(),
                initial_solution: self.initial_solution.clone(),
                n_accepted: self.n_accepted,
                n_rejected: self.n_rejected,
                n_best_updates: self.n_best_updates,
                rng: child_rng,
            },
            SearchStateCloneType::ClearBest => Self {
                start_iteration: 0,
                start_time: now,
                start_n_accepted: 0,
                start_n_rejected: 0,
                start_n_best_updates: 0,
                instance: self.instance,
                iteration: 0,
                solution: self.solution.clone(),
                best_time: now,
                best_iteration: 0,
                best_solution: self.solution.clone(),
                initial_solution: self.solution.clone(),
                n_accepted: 0,
                n_rejected: 0,
                n_best_updates: 0,
                rng: child_rng,
            },
            SearchStateCloneType::StartBest => Self {
                start_iteration: 0,
                start_time: now,
                start_n_accepted: 0,
                start_n_rejected: 0,
                start_n_best_updates: 0,
                instance: self.instance,
                iteration: 0,
                solution: self.best_solution.clone(),
                best_time: now,
                best_iteration: 0,
                best_solution: self.best_solution.clone(),
                initial_solution: self.best_solution.clone(),
                n_accepted: 0,
                n_rejected: 0,
                n_best_updates: 0,
                rng: child_rng,
            },
        }
    }

    /// Merges the results of a completed sub-run back into this state.
    ///
    /// - The current solution is replaced with `cloned_state.solution`.
    /// - The iteration counter and the accept/reject/best-update counters are
    ///   incremented by each one's delta over the sub-run.
    /// - `initial_solution` is **not** overwritten: the parent's anchor is preserved.
    /// - If the sub-run found a better solution, the best solution is updated.
    ///
    /// # Panics
    ///
    /// Panics if `cloned_state` references a different problem instance.
    pub fn update_state(&mut self, cloned_state: Self) {
        if !std::ptr::eq(self.instance, cloned_state.instance) {
            panic!("Cannot update state with different instance");
        }

        if self.start_time > cloned_state.start_time {
            tracing::warn!(
                "Start time of new state is later than current state. \
                This may cause incorrect behavior."
            );
        }

        // update the current state with the new state
        self.solution = cloned_state.solution;

        // add iteration into the current iteration
        let old_iteration = self.iteration;
        self.iteration += cloned_state.iteration - cloned_state.start_iteration;
        self.n_accepted += cloned_state.n_accepted - cloned_state.start_n_accepted;
        self.n_rejected += cloned_state.n_rejected - cloned_state.start_n_rejected;
        self.n_best_updates += cloned_state.n_best_updates - cloned_state.start_n_best_updates;

        // update the best solution if the one of the new state is better than the current
        if cloned_state
            .best_solution
            .is_better_than(&self.best_solution)
        {
            // With `SearchStateCloneType::Simple` the inherited `best_iteration` can
            // predate `start_iteration`; saturate so the offset is 0 in that case.
            let sub_run_best_offset = cloned_state
                .best_iteration
                .saturating_sub(cloned_state.start_iteration);
            self.best_solution = cloned_state.best_solution;
            self.best_time = cloned_state.best_time;
            self.best_iteration = old_iteration + sub_run_best_offset;
        }
        tracing::debug!(
            iteration = self.iteration,
            best_iteration = self.best_iteration,
            "Sub-run state merged"
        );
    }

    /// Applies a neighborhood move, updates the iteration counter, and refreshes the best solution.
    /// Increments [`n_accepted`](Self::n_accepted).
    pub fn apply<Move>(&mut self, neighbor: &Move) -> Result<(), crate::error::OptError>
    where
        Move: MoveToNeighbor<Problem>,
    {
        self.iteration = neighbor.apply_to_iteration(self.iteration);
        neighbor.apply_to_solution(self.instance, &mut self.solution)?;
        self.n_accepted += 1;
        self.update_best();
        Ok(())
    }

    /// Applies a neighborhood move and updates the iteration counter, but does
    /// **not** refresh the best solution. Increments [`n_accepted`](Self::n_accepted).
    ///
    /// Use this in perturbation phases where moves intentionally diversify and
    /// a best-solution update is deferred until the phase completes. Call
    /// [`update_best`](Self::update_best) once after the phase ends.
    pub fn apply_move_only<Move>(&mut self, neighbor: &Move) -> Result<(), crate::error::OptError>
    where
        Move: MoveToNeighbor<Problem>,
    {
        self.iteration = neighbor.apply_to_iteration(self.iteration);
        neighbor.apply_to_solution(self.instance, &mut self.solution)?;
        self.n_accepted += 1;
        Ok(())
    }

    /// Increments the iteration counter by one without applying any move.
    /// Increments [`n_rejected`](Self::n_rejected).
    pub fn progress_iteration(&mut self) {
        self.iteration += 1;
        self.n_rejected += 1;
    }

    /// Picks a uniformly random move from the neighborhood of the current solution.
    ///
    /// Returns [`OptError::InvalidState`](crate::error::OptError::InvalidState) when
    /// the neighborhood is empty; `context` (typically the heuristic name) prefixes
    /// the error message.
    pub fn random_neighbor<N>(&mut self, context: &str) -> Result<N, crate::error::OptError>
    where
        N: MoveToNeighbor<Problem>,
    {
        use rand::seq::IteratorRandom;
        N::iter(self.instance, &self.solution)
            .choose(&mut self.rng)
            .ok_or_else(|| {
                crate::error::OptError::InvalidState(format!(
                    "{context}: neighborhood is empty, no move can be selected"
                ))
            })
    }

    /// Runs `heuristic` on a sub-state cloned with `clone_type`, then merges the
    /// result back — the standard `clone_for_new_run` → `run` → `update_state`
    /// triad used by meta-heuristics.
    pub fn run_sub(
        &mut self,
        heuristic: &mut dyn crate::heuristic::Heuristic<Problem>,
        clone_type: SearchStateCloneType,
    ) -> Result<(), crate::error::OptError> {
        let mut sub = self.clone_for_new_run(clone_type);
        heuristic.run(&mut sub)?;
        self.update_state(sub);
        Ok(())
    }

    /// Returns `true` if applying `m` to the current solution yields a solution
    /// better than the current solution.
    pub fn is_neighbor_better_than_current<Move>(&self, m: &Move) -> bool
    where
        Move: MoveToNeighbor<Problem>,
    {
        m.move_to_be_better_than(self.instance, &self.solution, &self.solution)
    }

    /// Returns `true` if applying `m` to the current solution yields a solution
    /// better than the best solution found so far.
    pub fn is_neighbor_better_than_best<Move>(&self, m: &Move) -> bool
    where
        Move: MoveToNeighbor<Problem>,
    {
        m.move_to_be_better_than(self.instance, &self.solution, &self.best_solution)
    }
}

impl<'a, Problem> std::fmt::Debug for SearchState<'a, Problem>
where
    Problem: ProblemTrait,
    Problem::Solution: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchState")
            .field(
                "current",
                &(
                    self.start_time.elapsed(),
                    self.iteration,
                    self.solution.clone(),
                ),
            )
            .field(
                "best",
                &(
                    self.best_time - self.start_time,
                    self.best_iteration,
                    self.best_solution.clone(),
                ),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::max_cut::MaxCut;
    use crate::problem::{MaxCutFlipNeighbor, MaxCutSolution};

    fn triangle() -> MaxCut {
        MaxCut::from_edges([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)])
    }

    fn first_flip(mc: &MaxCut, sol: &MaxCutSolution) -> MaxCutFlipNeighbor {
        MaxCutFlipNeighbor::iter(mc, sol).next().unwrap()
    }

    #[test]
    fn new_records_initial_solution_and_zero_counters() {
        let mc = triangle();
        let state = SearchState::new(&mc);
        assert_eq!(state.iteration, 0);
        assert_eq!(state.n_accepted, 0);
        assert_eq!(state.n_rejected, 0);
        assert_eq!(state.n_best_updates, 0);
        // initial == current == best at construction
        assert_eq!(state.initial_solution.cut, state.solution.cut);
        assert_eq!(state.initial_solution.cut, state.best_solution.cut);
    }

    #[test]
    fn with_solution_anchors_initial_to_provided_solution() {
        let mc = triangle();
        let sol = MaxCutSolution::new_from_cut(&mc, vec![true, false, true]);
        let state = SearchState::with_solution(&mc, sol.clone());
        assert_eq!(state.initial_solution.cut, sol.cut);
        assert_eq!(state.solution.cut, sol.cut);
        assert_eq!(state.best_solution.cut, sol.cut);
    }

    #[test]
    fn apply_increments_n_accepted_only() {
        let mc = triangle();
        let mut state = SearchState::new(&mc);
        let m = first_flip(&mc, &state.solution);
        state.apply(&m).unwrap();
        assert_eq!(state.n_accepted, 1);
        assert_eq!(state.n_rejected, 0);
        assert_eq!(state.iteration, 1);
    }

    #[test]
    fn progress_iteration_increments_n_rejected_only() {
        let mc = triangle();
        let mut state = SearchState::new(&mc);
        state.progress_iteration();
        state.progress_iteration();
        assert_eq!(state.n_accepted, 0);
        assert_eq!(state.n_rejected, 2);
        assert_eq!(state.iteration, 2);
    }

    #[test]
    fn update_best_counts_real_updates_only() {
        let mc = triangle();
        // Start from a non-optimal solution so that flipping yields improvement
        let sol = MaxCutSolution::new_from_cut(&mc, vec![false, false, false]);
        let mut state = SearchState::with_solution(&mc, sol);
        let before = state.n_best_updates;
        let m = first_flip(&mc, &state.solution);
        state.apply(&m).unwrap();
        assert!(state.n_best_updates > before, "best should have improved");

        // No-op update_best (current unchanged) should not bump the counter
        let after_apply = state.n_best_updates;
        let updated = state.update_best();
        assert!(!updated);
        assert_eq!(state.n_best_updates, after_apply);
    }

    #[test]
    fn clone_for_new_run_simple_inherits_everything() {
        let mc = triangle();
        let mut state = SearchState::new(&mc);
        let m = first_flip(&mc, &state.solution);
        state.apply(&m).unwrap();
        state.progress_iteration();
        let parent_initial = state.initial_solution.cut.clone();
        let parent_n_accepted = state.n_accepted;
        let parent_n_rejected = state.n_rejected;
        let parent_n_best = state.n_best_updates;

        let child = state.clone_for_new_run(SearchStateCloneType::Simple);
        assert_eq!(child.initial_solution.cut, parent_initial);
        assert_eq!(child.n_accepted, parent_n_accepted);
        assert_eq!(child.n_rejected, parent_n_rejected);
        assert_eq!(child.n_best_updates, parent_n_best);
        assert_eq!(child.start_n_accepted, parent_n_accepted);
        assert_eq!(child.start_n_rejected, parent_n_rejected);
        assert_eq!(child.start_n_best_updates, parent_n_best);
    }

    #[test]
    fn update_state_simple_clone_without_improvement_does_not_underflow() {
        let mc = triangle();
        let sol = MaxCutSolution::new_from_cut(&mc, vec![false, false, false]);
        let mut state = SearchState::with_solution(&mc, sol);
        let m = first_flip(&mc, &state.solution);
        state.apply(&m).unwrap(); // best found at iteration 1
        state.progress_iteration();
        state.progress_iteration(); // iteration = 3 > best_iteration = 1

        // Simple clone inherits best_iteration (1) < start_iteration (3);
        // merging back a sub-run with no improvement must not underflow.
        let child = state.clone_for_new_run(SearchStateCloneType::Simple);
        let best_iteration_before = state.best_iteration;
        state.update_state(child);
        assert_eq!(state.best_iteration, best_iteration_before);
        assert_eq!(state.iteration, 3);
    }

    #[test]
    fn clone_for_new_run_clear_best_reanchors_to_current() {
        let mc = triangle();
        let sol = MaxCutSolution::new_from_cut(&mc, vec![true, false, false]);
        let mut state = SearchState::with_solution(&mc, sol);
        state.progress_iteration(); // bump n_rejected so we can tell it gets cleared

        let child = state.clone_for_new_run(SearchStateCloneType::ClearBest);
        assert_eq!(child.iteration, 0);
        assert_eq!(child.n_accepted, 0);
        assert_eq!(child.n_rejected, 0);
        assert_eq!(child.n_best_updates, 0);
        assert_eq!(child.initial_solution.cut, child.solution.cut);
        assert_eq!(child.initial_solution.cut, state.solution.cut);
    }

    #[test]
    fn clone_for_new_run_start_best_reanchors_to_best() {
        let mc = triangle();
        let sol = MaxCutSolution::new_from_cut(&mc, vec![false, false, false]);
        let mut state = SearchState::with_solution(&mc, sol);
        let m = first_flip(&mc, &state.solution);
        state.apply(&m).unwrap();
        // current solution should now differ from initial; best should equal current
        let best_cut = state.best_solution.cut.clone();

        let child = state.clone_for_new_run(SearchStateCloneType::StartBest);
        assert_eq!(child.iteration, 0);
        assert_eq!(child.n_accepted, 0);
        assert_eq!(child.n_rejected, 0);
        assert_eq!(child.n_best_updates, 0);
        assert_eq!(child.initial_solution.cut, best_cut);
        assert_eq!(child.solution.cut, best_cut);
        assert_eq!(child.best_solution.cut, best_cut);
    }

    #[test]
    fn new_with_seed_is_deterministic() {
        let mc = triangle();
        let a = SearchState::new_with_seed(&mc, 42);
        let b = SearchState::new_with_seed(&mc, 42);
        assert_eq!(a.initial_solution.cut, b.initial_solution.cut);
    }

    #[test]
    fn new_with_seed_different_seeds_can_differ() {
        let mc = MaxCut::from_edges((0..30).map(|i| (i, (i + 1) % 30, 1.0)));
        let a = SearchState::new_with_seed(&mc, 1);
        let b = SearchState::new_with_seed(&mc, 2);
        // Two unrelated seeds on a 30-bit space almost certainly disagree.
        assert_ne!(a.initial_solution.cut, b.initial_solution.cut);
    }

    #[test]
    fn fork_advances_parent_rng() {
        use rand::Rng;
        let mc = triangle();
        let mut a = SearchState::new_with_seed(&mc, 7);
        let mut b = SearchState::new_with_seed(&mc, 7);
        // Fork the child off of `a`; this consumes a chunk of a's stream.
        let _child = a.clone_for_new_run(SearchStateCloneType::ClearBest);
        let next_a: u64 = a.rng.random();
        let next_b: u64 = b.rng.random();
        // After forking, the parent's next draw must differ from the unforked baseline.
        assert_ne!(next_a, next_b);
    }

    #[test]
    fn sibling_subruns_have_independent_streams() {
        use rand::Rng;
        let mc = triangle();
        let mut parent = SearchState::new_with_seed(&mc, 7);
        let mut child1 = parent.clone_for_new_run(SearchStateCloneType::ClearBest);
        let mut child2 = parent.clone_for_new_run(SearchStateCloneType::ClearBest);
        let n1: u64 = child1.rng.random();
        let n2: u64 = child2.rng.random();
        assert_ne!(n1, n2, "two sibling forks must yield distinct streams");
    }

    #[test]
    fn update_state_merges_counter_deltas() {
        let mc = triangle();
        let sol = MaxCutSolution::new_from_cut(&mc, vec![false, false, false]);
        let mut parent = SearchState::with_solution(&mc, sol);
        // Pre-existing parent counters to verify additive merge
        parent.progress_iteration();
        parent.progress_iteration();
        let parent_initial_before = parent.initial_solution.cut.clone();

        // Spawn ClearBest sub-run (counters start at 0)
        let mut child = parent.clone_for_new_run(SearchStateCloneType::ClearBest);
        let m = first_flip(&mc, &child.solution);
        child.apply(&m).unwrap();
        child.progress_iteration();
        let child_accepted = child.n_accepted;
        let child_rejected = child.n_rejected;
        let child_best = child.n_best_updates;

        parent.update_state(child);
        assert_eq!(parent.n_accepted, child_accepted);
        assert_eq!(parent.n_rejected, 2 + child_rejected);
        assert_eq!(parent.n_best_updates, child_best);
        // initial_solution must NOT be overwritten by the child's
        assert_eq!(parent.initial_solution.cut, parent_initial_before);
    }
}
