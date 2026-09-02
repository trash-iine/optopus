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
///
/// All three variants keep the parent's iteration frame: the child's
/// `iteration` runs on from where the parent stands, and `start_iteration`
/// marks where the phase began. Everything budget-shaped is measured against
/// that anchor — [`SearchState::iterations_this_run`],
/// [`StopCondition`](crate::heuristic::StopCondition), and the deltas
/// [`SearchState::update_state`] merges — so a phase still starts at zero *of
/// its own budget*.
///
/// One shared frame is what makes an iteration number mean the same thing in a
/// parent and its sub-run — which is what lets tabu prohibitions, whose
/// boundaries are absolute iterations, cross between them.
///
/// The accept / reject / best-update counters measure a phase rather than
/// timestamp anything, so every variant starts them at zero, and
/// [`SearchState::update_state`] adds them straight into the parent.
#[derive(Clone, Debug)]
pub enum SearchStateCloneType {
    /// Clone the state as-is.
    ///
    /// - Starts from the current solution
    /// - Retains the original start time
    /// - Retains the current best solution
    Simple,

    /// Clone the state and reset all best-solution tracking.
    ///
    /// - Starts from the current solution
    /// - Resets the clocks, and re-anchors `start_iteration` to now
    /// - Sets the best solution to the current solution
    ClearBest,

    /// Clone the state starting from the best solution found so far.
    ///
    /// - Starts from the best solution
    /// - Resets the clocks, and re-anchors `start_iteration` to now
    /// - Retains the best solution
    StartBest,
}

/// A single best-solution improvement recorded during a run.
///
/// Points are recorded by [`SearchState::update_best`] whenever an objective
/// probe is installed via [`SearchState::set_objective_probe`], giving an
/// anytime trajectory of the search. The wall-clock `instant` is absolute, so
/// elapsed times stay consistent even across sub-run clones whose relative
/// timers reset (`ClearBest` / `StartBest`).
#[derive(Clone, Copy, Debug)]
pub struct TrajectoryPoint {
    /// Wall-clock instant at which the improvement happened.
    pub instant: std::time::Instant,
    /// Iteration at which the improvement happened (remapped into the parent's
    /// frame when a sub-run is merged).
    pub iteration: u64,
    /// Objective value of the new best solution, as reported by the probe.
    pub objective: f64,
}

/// Holds the full runtime state of a heuristic search.
///
/// Contains the problem instance (by reference), the current solution,
/// the best solution found so far, and iteration / timing metadata.
///
/// The `pub(crate)` sub-run anchors (`start_iteration`, `start_time`) are
/// internal merge bookkeeping — hands off. The private `tabu` field is the one
/// piece of live search state that is not `pub`: what it holds is meaningless
/// without a move to interpret it, so every access goes through one
/// ([`Self::tabu_allows`], [`Self::record_tabu`], [`Self::reserve_tabu_vars`]).
#[derive(Clone)]
pub struct SearchState<'a, Problem>
where
    Problem: ProblemTrait,
{
    /// Iteration count at the start of the current sub-run.
    pub(crate) start_iteration: u64,
    /// Wall-clock time when the current sub-run started.
    pub(crate) start_time: std::time::Instant,
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
    /// Number of moves accepted (`apply` / `apply_move_only` calls).
    /// Always satisfies `iterations_this_run() == n_accepted + n_rejected`
    /// when only the standard methods on this state are used.
    pub n_accepted: u64,
    /// Number of iterations that advanced without applying a move
    /// (`progress_iteration` calls).
    pub n_rejected: u64,
    /// Number of times `update_best` actually replaced `best_solution`.
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
    /// Anytime trajectory: one point per recorded best-solution improvement.
    ///
    /// Empty unless an objective probe is installed with
    /// [`Self::set_objective_probe`]. Points merged from `ClearBest` /
    /// `StartBest` sub-runs track the *sub-run's* best, which may be worse
    /// than an earlier parent best — consumers wanting a monotone incumbent
    /// curve must filter with the problem's optimization direction.
    pub trajectory: Vec<TrajectoryPoint>,
    /// Extracts an `f64` objective from a solution for trajectory recording.
    /// `None` (the default) disables recording entirely, keeping
    /// `update_best` allocation-free for library users who don't need it.
    pub(crate) objective_probe: Option<fn(&Problem::Solution) -> f64>,
    /// The tabu prohibitions this search remembers, and the tenure every write
    /// to them draws from.
    ///
    /// Access to this tabu map is only allowed by applying a move to neighbor.
    /// See [`set_tabu_tenure`](Self::set_tabu_tenure) and friends.
    tabu: crate::common::TabuMemory,
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
            trajectory: Vec::new(),
            objective_probe: None,
            tabu: crate::common::TabuMemory::default(),
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
            trajectory: Vec::new(),
            objective_probe: None,
            tabu: crate::common::TabuMemory::default(),
        }
    }

    /// Installs an objective probe, enabling anytime-trajectory recording.
    ///
    /// After this call every actual best-solution improvement appends a
    /// [`TrajectoryPoint`] to [`trajectory`](Self::trajectory). The probe is
    /// inherited by sub-run clones, so improvements found inside
    /// meta-heuristic phases are recorded too.
    pub fn set_objective_probe(&mut self, probe: fn(&Problem::Solution) -> f64) {
        self.objective_probe = Some(probe);
    }

    /// Iterations run since this state started, i.e. excluding the work a
    /// parent run had already done when [`clone_for_new_run`](Self::clone_for_new_run)
    /// forked it.
    ///
    /// The counterpart of [`duration`](Self::duration) for an iteration
    /// budget: a heuristic that normalizes anything by how far through its
    /// budget it is has to measure from its own start, not from zero, or a
    /// sub-run reads the parent's progress as its own.
    pub fn iterations_this_run(&self) -> u64 {
        self.iteration - self.start_iteration
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
            if let Some(probe) = self.objective_probe {
                self.trajectory.push(TrajectoryPoint {
                    instant: self.best_time,
                    iteration: self.iteration,
                    objective: probe(&self.best_solution),
                });
            }
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
    /// **Tabu semantics**: the child starts with an empty tabu memory, whichever
    /// clone type is used — a phase is its own tabu list. What it learns does
    /// come back: [`update_state`](Self::update_state) carries the sub-run's
    /// prohibitions into this state along with its solution. A phase that should
    /// *start* from the parent's asks with
    /// [`inherit_tabu_from`](Self::inherit_tabu_from):
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// let mc = MaxCut::from_edges([(0, 1, 1.0), (1, 2, 1.0)]);
    /// let mut state = SearchState::new_with_seed(&mc, 1);
    /// state.set_tabu_tenure((5, 10));
    ///
    /// let m = MaxCutFlipNeighbor::new(&mc, &state.solution, 1);
    /// state.apply(&m)?;                        // applying records vertex 1
    /// assert!(!state.tabu_allows(&m));
    ///
    /// let mut sub = state.clone_for_new_run(SearchStateCloneType::ClearBest);
    /// assert!(sub.tabu_allows(&m));            // a new phase starts free
    /// assert_eq!(sub.tabu_tenure(), (0, 0));
    ///
    /// sub.inherit_tabu_from(&state);           // ...unless it asks for the parent's
    /// assert!(!sub.tabu_allows(&m));
    /// assert_eq!(sub.tabu_tenure(), (5, 10));
    /// # Ok::<(), optopus::error::OptError>(())
    /// ```
    pub fn clone_for_new_run(&mut self, clone_type: SearchStateCloneType) -> Self {
        let now = std::time::Instant::now();
        let child_rng = SmallRng::from_rng(&mut self.rng);
        match clone_type {
            SearchStateCloneType::Simple => Self {
                start_iteration: self.iteration,
                start_time: self.start_time,
                instance: self.instance,
                iteration: self.iteration,
                solution: self.solution.clone(),
                best_time: self.best_time,
                best_iteration: self.best_iteration,
                best_solution: self.best_solution.clone(),
                initial_solution: self.initial_solution.clone(),
                n_accepted: 0,
                n_rejected: 0,
                n_best_updates: 0,
                rng: child_rng,
                trajectory: Vec::new(),
                objective_probe: self.objective_probe,
                tabu: crate::common::TabuMemory::default(),
            },
            SearchStateCloneType::ClearBest => Self {
                start_iteration: self.iteration,
                start_time: now,
                instance: self.instance,
                iteration: self.iteration,
                solution: self.solution.clone(),
                best_time: now,
                best_iteration: self.iteration,
                best_solution: self.solution.clone(),
                initial_solution: self.solution.clone(),
                n_accepted: 0,
                n_rejected: 0,
                n_best_updates: 0,
                rng: child_rng,
                trajectory: Vec::new(),
                objective_probe: self.objective_probe,
                tabu: crate::common::TabuMemory::default(),
            },
            SearchStateCloneType::StartBest => Self {
                start_iteration: self.iteration,
                start_time: now,
                instance: self.instance,
                iteration: self.iteration,
                solution: self.best_solution.clone(),
                best_time: now,
                best_iteration: self.iteration,
                best_solution: self.best_solution.clone(),
                initial_solution: self.best_solution.clone(),
                n_accepted: 0,
                n_rejected: 0,
                n_best_updates: 0,
                rng: child_rng,
                trajectory: Vec::new(),
                objective_probe: self.objective_probe,
                tabu: crate::common::TabuMemory::default(),
            },
        }
    }

    /// Merges the results of a completed sub-run back into this state.
    ///
    /// - The current solution is replaced with `cloned_state.solution`, and the
    ///   tabu memory with the sub-run's — both are where the search got to.
    /// - The iteration counter is advanced by the sub-run's own progress
    ///   (`iteration - start_iteration`), and the accept/reject/best-update
    ///   counters — which the sub-run counted from zero — are added on.
    /// - `initial_solution` is not overwritten.
    /// - If the sub-run found a better solution, the best solution is updated.
    /// - The sub-run's RNG is discarded, so its internal draws do not leak into
    ///   this state's stream.
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

        // The prohibitions come back with the solution, and for the same
        // reason: both are where the search got to. This is a replacement, not
        // a merge — whatever this state had recorded before the phase is gone,
        // exactly as its solution is. It is only sound because the sub-run ran
        // in *this* state's iteration frame, so the boundaries it recorded are
        // already absolute here; see `clone_for_new_run`.
        self.tabu = cloned_state.tabu;

        // add iteration into the current iteration
        let old_iteration = self.iteration;
        self.iteration += cloned_state.iteration - cloned_state.start_iteration;
        self.n_accepted += cloned_state.n_accepted;
        self.n_rejected += cloned_state.n_rejected;
        self.n_best_updates += cloned_state.n_best_updates;

        // Append the sub-run's trajectory, remapping iterations into this
        // state's frame (same saturating scheme as `best_iteration` below).
        // Instants are absolute, so the time axis needs no adjustment.
        self.trajectory
            .extend(cloned_state.trajectory.iter().map(|p| TrajectoryPoint {
                instant: p.instant,
                iteration: old_iteration + p.iteration.saturating_sub(cloned_state.start_iteration),
                objective: p.objective,
            }));

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
        // `best_time` adopted from a sub-run always postdates this state's
        // `start_time` (sub-runs are cloned after this state started); the
        // `start_time` inversion warning above covers the exotic cases.
        debug_assert!(
            self.best_time >= self.start_time,
            "best_time must not precede start_time after a sub-run merge"
        );
        tracing::debug!(
            iteration = self.iteration,
            best_iteration = self.best_iteration,
            "Sub-run state merged"
        );
    }

    /// Sets the tabu tenure range `(min, max)` every recorded move is forbidden
    /// for.
    ///
    /// A heuristic that wants tabu behavior sets this at the top of its
    /// `run_once`; it is a plain field write. The default is `(0, 0)`, which
    /// records each applied move but frees it again on the next iteration —
    /// remembering without forbidding, which is what a search that never asks
    /// about the tabu map wants.
    ///
    /// # Panics
    ///
    /// Panics if `tenure.0 > tenure.1` (an empty range).
    pub fn set_tabu_tenure(&mut self, tenure: (u64, u64)) {
        self.tabu.set_tenure(tenure);
    }

    /// The tenure range [`set_tabu_tenure`](Self::set_tabu_tenure) installed.
    pub fn tabu_tenure(&self) -> (u64, u64) {
        self.tabu.tenure()
    }

    /// Drops every tabu prohibition, keeping the tenure.
    pub fn reset_tabu(&mut self) {
        self.tabu.clear();
    }

    /// Returns whether `neighbor` is free to be applied at the current
    /// iteration, i.e. not forbidden by what this search has recorded.
    ///
    /// A key nothing has recorded yet is free, so a search that has not started
    /// recording allows everything.
    ///
    /// The question is asked of the move type rather than of its erased policy,
    /// so [`is_move_enabled`](EnabledTabu::is_move_enabled) inlines into a
    /// neighborhood scan. A move that does not implement [`EnabledTabu`]
    /// therefore cannot be asked at all — it fails to compile here, which is
    /// the same answer [`record_tabu`](Self::record_tabu) gives at runtime.
    #[inline]
    pub fn tabu_allows<N>(&self, neighbor: &N) -> bool
    where
        N: MoveToNeighbor<Problem> + EnabledTabu,
    {
        neighbor.is_move_enabled(&self.tabu, self.iteration)
    }

    /// Records `neighbor` as tabu for a tenure drawn from
    /// [`tabu_tenure`](Self::tabu_tenure), counted from the current iteration.
    ///
    /// [`apply`](Self::apply) already does this for every move it applies, so a
    /// heuristic needs this only to forbid a move it is *not* applying.
    ///
    /// # Errors
    ///
    /// Returns [`OptError::Unsupported`](crate::error::OptError::Unsupported)
    /// when the move has no tabu policy — it does not implement
    /// [`EnabledTabu`], or implements it without overriding
    /// [`MoveToNeighbor::tabu_policy`]. Applying such a move is fine (see
    /// [`apply`](Self::apply)); it is only reaching for the tabu memory that has
    /// no answer.
    pub fn record_tabu<Move>(&mut self, neighbor: &Move) -> Result<(), crate::error::OptError>
    where
        Move: MoveToNeighbor<Problem>,
    {
        let policy = tabu_policy::<Problem, Move>(neighbor)?;
        policy.add_to_tabu_map(&mut self.tabu, self.iteration, &mut self.rng);
        Ok(())
    }

    /// Checks that `neighbor`'s type hands a tabu policy over, without touching
    /// the memory.
    ///
    /// [`apply`](Self::apply) records silently when there is no policy, which is
    /// what keeps tabu optional for a problem that never wanted it — but a
    /// heuristic whose entire method *is* the tabu list would then run without
    /// one and never say so. Such a heuristic calls this once per iteration.
    ///
    /// # Errors
    ///
    /// As [`record_tabu`](Self::record_tabu).
    pub fn require_tabu_policy<Move>(&self, neighbor: &Move) -> Result<(), crate::error::OptError>
    where
        Move: MoveToNeighbor<Problem>,
    {
        tabu_policy::<Problem, Move>(neighbor).map(|_| ())
    }

    /// Carries `parent`'s prohibitions and tenure into this state, replacing
    /// whatever it held.
    ///
    /// [`clone_for_new_run`](Self::clone_for_new_run) starts a sub-run free,
    /// because a phase is usually its own tabu list. This is the opt-out, for a
    /// phase that should go on avoiding what the parent just did.
    ///
    /// The boundaries are translated into this state's iteration frame — see
    /// [`TabuMemory::inherit`](crate::common::TabuMemory::inherit). Between a
    /// state and its sub-run that translation is the identity, since they share
    /// the frame; it earns its keep when the two states were built separately,
    /// which is the case the method stays total for.
    pub fn inherit_tabu_from(&mut self, parent: &Self) {
        self.tabu
            .inherit(&parent.tabu, parent.iteration, self.iteration);
    }

    /// Grows the tabu memory's dense key space to `0..n` up front.
    ///
    /// Pure pre-allocation — recording grows it on demand, and does so without
    /// changing what it draws — for a heuristic that knows the instance size
    /// and is about to record `n` keys.
    pub fn reserve_tabu_vars(&mut self, n: usize) {
        self.tabu.reserve_vars(n);
    }

    /// Records `neighbor` if it has a tabu policy, and does nothing if it does
    /// not.
    ///
    /// This is the half of [`record_tabu`](Self::record_tabu) that `apply` uses.
    /// It is deliberately silent rather than fallible: a problem that never
    /// implements [`EnabledTabu`] must still be able to apply its moves, so
    /// "this move has no tabu policy" is an answer here and an error only when
    /// a caller asks about the map explicitly.
    ///
    /// For a move without a policy the `tabu_policy` call monomorphizes to
    /// `None` and the whole body compiles away.
    #[inline]
    fn record_applied<Move>(&mut self, neighbor: &Move)
    where
        Move: MoveToNeighbor<Problem>,
    {
        if let Some(policy) = neighbor.tabu_policy() {
            policy.add_to_tabu_map(&mut self.tabu, self.iteration, &mut self.rng);
        }
    }

    /// Applies a neighborhood move, updates the iteration counter, and refreshes the best solution.
    /// Increments [`n_accepted`](Self::n_accepted).
    ///
    /// The move is recorded in the tabu memory **before** the iteration
    /// advances, so a tenure of `d` forbids exactly the `d` iterations that
    /// follow the one the move was made on. A move whose
    /// [`tabu_policy`](MoveToNeighbor::tabu_policy) is `None` records nothing
    /// and is applied all the same.
    pub fn apply<Move>(&mut self, neighbor: &Move) -> Result<(), crate::error::OptError>
    where
        Move: MoveToNeighbor<Problem>,
    {
        self.record_applied(neighbor);
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
        self.record_applied(neighbor);
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
        N::random_neighbor(self.instance, &self.solution, &mut self.rng).ok_or_else(|| {
            crate::error::OptError::InvalidState(format!(
                "{context}: neighborhood is empty, no move can be selected"
            ))
        })
    }

    /// Opens a sub-state on a [`ProblemReduction`](crate::trait_defs::ProblemReduction)'s
    /// target, warm-started from the current solution.
    ///
    /// The reduction is a pure map — it knows how to project a solution and
    /// nothing about search state — so the *crossing* is this method: the seed
    /// is drawn from this state's RNG, which is what keeps a seeded run
    /// reproducible through a reduction, and the sub-state starts with zeroed
    /// counters so [`close_reduction`](Self::close_reduction) can merge it
    /// back.
    pub fn open_reduction<'r, R>(&mut self, reduction: &'r R) -> SearchState<'r, R::Target>
    where
        R: crate::trait_defs::ProblemReduction<Source = Problem> + ?Sized,
    {
        let seed = rand::RngCore::next_u64(&mut self.rng);
        let start = reduction.project(&self.solution);
        SearchState::with_solution_and_seed(reduction.target(), start, seed)
    }

    /// Folds a sub-run that ran on a [`ProblemReduction`](crate::trait_defs::ProblemReduction)'s
    /// target back into this state — the closing half of
    /// [`open_reduction`](Self::open_reduction).
    ///
    /// **The counters are merged before the solution moves, and that order is
    /// load-bearing.** The sub-run ran on a *different* instance, so it cannot
    /// go through [`update_state`](Self::update_state), which requires the same
    /// instance; but its work is comparable and must not be dropped, or a
    /// benchmark reports near-zero counters exactly on the instances where the
    /// reduction did something. Merging *after* installing the solution would
    /// record a `best_iteration` that omits the sub-run entirely — invisible in
    /// the objective, which is why this is one method rather than two a caller
    /// assembles. The sub-state is assumed to start from zeroed counters, which
    /// `open_reduction` guarantees.
    ///
    /// The lifted solution is then installed as-is. `lift` returns a complete
    /// `Solution`, caches included, so there is nothing left to reconstruct;
    /// the optional improving-move indexes (`positive_gain`, `zero_gain`) are
    /// opt-in and rebuilt by whoever asks for them next.
    ///
    /// It is `sub.best_solution` that crosses, not `sub.solution`: a tabu-style
    /// sub-run usually ends part-way out of a local optimum, so where it
    /// stopped is not what it found.
    pub fn close_reduction<R>(&mut self, reduction: &R, sub: &SearchState<'_, R::Target>)
    where
        R: crate::trait_defs::ProblemReduction<Source = Problem> + ?Sized,
    {
        self.iteration += sub.iteration;
        self.n_accepted += sub.n_accepted;
        self.n_rejected += sub.n_rejected;
        self.n_best_updates += sub.n_best_updates;

        self.solution = reduction.lift(self.instance, &self.solution, &sub.best_solution);

        self.update_best();
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

/// The tabu policy of `neighbor`, or the error the fallible tabu API reports
/// when it has none.
fn tabu_policy<Problem, Move>(neighbor: &Move) -> Result<&dyn EnabledTabu, crate::error::OptError>
where
    Problem: ProblemTrait,
    Move: MoveToNeighbor<Problem>,
{
    neighbor.tabu_policy().ok_or_else(|| {
        crate::error::OptError::Unsupported(format!(
            "{} has no tabu policy: implement EnabledTabu for it and override \
             MoveToNeighbor::tabu_policy with `Some(self)`",
            std::any::type_name::<Move>()
        ))
    })
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
        assert_eq!(state.initial_solution.x, state.solution.x);
        assert_eq!(state.initial_solution.x, state.best_solution.x);
    }

    #[test]
    fn with_solution_anchors_initial_to_provided_solution() {
        let mc = triangle();
        let sol = MaxCutSolution::new_from_assignment(&mc, vec![true, false, true]);
        let state = SearchState::with_solution(&mc, sol.clone());
        assert_eq!(state.initial_solution.x, sol.x);
        assert_eq!(state.solution.x, sol.x);
        assert_eq!(state.best_solution.x, sol.x);
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
        let sol = MaxCutSolution::new_from_assignment(&mc, vec![false, false, false]);
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
        let parent_initial = state.initial_solution.x.clone();

        let child = state.clone_for_new_run(SearchStateCloneType::Simple);
        assert_eq!(child.initial_solution.x, parent_initial);
        assert_eq!(child.iteration, state.iteration);
        assert_eq!(child.start_iteration, state.iteration);
        assert_eq!(
            child.best_iteration, state.best_iteration,
            "best is retained"
        );
        // The counters measure the phase, so even `Simple` starts them at zero.
        assert_eq!(child.n_accepted, 0);
        assert_eq!(child.n_rejected, 0);
        assert_eq!(child.n_best_updates, 0);
    }

    #[test]
    fn update_state_simple_clone_without_improvement_does_not_underflow() {
        let mc = triangle();
        let sol = MaxCutSolution::new_from_assignment(&mc, vec![false, false, false]);
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

    /// `ClearBest` re-anchors the phase without leaving the parent's iteration
    /// frame: the counters run on, and `start_iteration` — not zero — is what
    /// marks the beginning, so the phase's own budget still starts empty.
    #[test]
    fn clone_for_new_run_clear_best_reanchors_to_current() {
        let mc = triangle();
        let sol = MaxCutSolution::new_from_assignment(&mc, vec![true, false, false]);
        let mut state = SearchState::with_solution(&mc, sol);
        state.progress_iteration(); // bump n_rejected so we can tell it carries over

        let child = state.clone_for_new_run(SearchStateCloneType::ClearBest);
        assert_eq!(child.iteration, state.iteration);
        assert_eq!(child.start_iteration, state.iteration);
        assert_eq!(child.iterations_this_run(), 0, "the phase starts at zero");
        assert_eq!(
            child.best_iteration, state.iteration,
            "and so does its best"
        );
        assert_eq!(child.n_accepted, 0, "the counters measure the phase");
        assert_eq!(child.n_rejected, 0);
        assert_eq!(child.n_best_updates, 0);
        assert_eq!(child.initial_solution.x, child.solution.x);
        assert_eq!(child.initial_solution.x, state.solution.x);
    }

    #[test]
    fn clone_for_new_run_start_best_reanchors_to_best() {
        let mc = triangle();
        let sol = MaxCutSolution::new_from_assignment(&mc, vec![false, false, false]);
        let mut state = SearchState::with_solution(&mc, sol);
        let m = first_flip(&mc, &state.solution);
        state.apply(&m).unwrap();
        // current solution should now differ from initial; best should equal current
        let best_cut = state.best_solution.x.clone();

        let child = state.clone_for_new_run(SearchStateCloneType::StartBest);
        assert_eq!(child.iteration, state.iteration);
        assert_eq!(child.start_iteration, state.iteration);
        assert_eq!(child.iterations_this_run(), 0);
        assert_eq!(child.best_iteration, state.iteration);
        assert_eq!(child.n_accepted, 0);
        assert_eq!(child.n_rejected, 0);
        assert_eq!(child.n_best_updates, 0);
        assert_eq!(child.initial_solution.x, best_cut);
        assert_eq!(child.solution.x, best_cut);
        assert_eq!(child.best_solution.x, best_cut);
    }

    #[test]
    fn new_with_seed_is_deterministic() {
        let mc = triangle();
        let a = SearchState::new_with_seed(&mc, 42);
        let b = SearchState::new_with_seed(&mc, 42);
        assert_eq!(a.initial_solution.x, b.initial_solution.x);
    }

    #[test]
    fn new_with_seed_different_seeds_can_differ() {
        let mc = MaxCut::from_edges((0..30).map(|i| (i, (i + 1) % 30, 1.0)));
        let a = SearchState::new_with_seed(&mc, 1);
        let b = SearchState::new_with_seed(&mc, 2);
        // Two unrelated seeds on a 30-bit space almost certainly disagree.
        assert_ne!(a.initial_solution.x, b.initial_solution.x);
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
    fn trajectory_empty_without_probe() {
        let mc = triangle();
        let sol = MaxCutSolution::new_from_assignment(&mc, vec![false, false, false]);
        let mut state = SearchState::with_solution(&mc, sol);
        let m = first_flip(&mc, &state.solution);
        state.apply(&m).unwrap();
        assert!(state.n_best_updates > 0);
        assert!(state.trajectory.is_empty());
    }

    #[test]
    fn trajectory_records_real_best_updates_only() {
        let mc = triangle();
        let sol = MaxCutSolution::new_from_assignment(&mc, vec![false, false, false]);
        let mut state = SearchState::with_solution(&mc, sol);
        state.set_objective_probe(|s| s.objective as f64);
        let m = first_flip(&mc, &state.solution);
        state.apply(&m).unwrap(); // improves: cut 0 -> 2
        assert_eq!(state.trajectory.len(), 1);
        assert_eq!(state.trajectory[0].objective, 2.0);
        assert_eq!(state.trajectory[0].iteration, 1);

        // A no-op update_best must not append a point.
        state.update_best();
        assert_eq!(state.trajectory.len(), 1);
    }

    #[test]
    fn update_state_merges_and_remaps_trajectory() {
        let mc = triangle();
        let sol = MaxCutSolution::new_from_assignment(&mc, vec![false, false, false]);
        let mut parent = SearchState::with_solution(&mc, sol);
        parent.set_objective_probe(|s| s.objective as f64);
        parent.progress_iteration();
        parent.progress_iteration(); // parent iteration = 2

        let mut child = parent.clone_for_new_run(SearchStateCloneType::ClearBest);
        assert!(child.trajectory.is_empty(), "child starts empty");
        let m = first_flip(&mc, &child.solution);
        child.apply(&m).unwrap(); // child-frame iteration 1

        parent.update_state(child);
        assert_eq!(parent.trajectory.len(), 1);
        // Child-frame iteration 1 remapped into the parent frame: 2 + 1.
        assert_eq!(parent.trajectory[0].iteration, 3);
        assert_eq!(parent.trajectory[0].objective, 2.0);
        assert!(parent.best_time >= parent.start_time);
    }

    #[test]
    fn update_state_merges_counter_deltas() {
        let mc = triangle();
        let sol = MaxCutSolution::new_from_assignment(&mc, vec![false, false, false]);
        let mut parent = SearchState::with_solution(&mc, sol);
        // Pre-existing parent counters to verify additive merge
        parent.progress_iteration();
        parent.progress_iteration();
        let parent_initial_before = parent.initial_solution.x.clone();

        // A sub-run counts its own moves from zero, and the merge adds them on.
        let mut child = parent.clone_for_new_run(SearchStateCloneType::ClearBest);
        let m = first_flip(&mc, &child.solution);
        child.apply(&m).unwrap();
        child.progress_iteration();
        let (accepted_in_phase, rejected_in_phase) = (child.n_accepted, child.n_rejected);
        let best_in_phase = child.n_best_updates;
        assert_eq!((accepted_in_phase, rejected_in_phase), (1, 1));

        parent.update_state(child);
        assert_eq!(parent.n_accepted, accepted_in_phase);
        assert_eq!(parent.n_rejected, 2 + rejected_in_phase);
        assert_eq!(parent.n_best_updates, best_in_phase);
        // initial_solution must NOT be overwritten by the child's
        assert_eq!(parent.initial_solution.x, parent_initial_before);
    }

    /// The tabu memory the state carries: what `apply` records into it, what a
    /// sub-run inherits, and what a move without a tabu policy gets instead.
    mod tabu {
        use super::*;
        use crate::error::OptError;
        use crate::problem::MaxCutSwapNeighbor;

        /// A problem whose move deliberately does **not** implement
        /// [`EnabledTabu`], i.e. the case a user hits by implementing only the
        /// three traits a local search needs.
        struct Untabued;

        #[derive(Clone, Debug)]
        struct Counter(i32);

        impl Rankable for Counter {
            fn is_better_than(&self, other: &Self) -> bool {
                self.0 > other.0
            }
        }

        impl ProblemTrait for Untabued {
            type Solution = Counter;
            fn new_solution(&self, _rng: &mut impl rand::Rng) -> Counter {
                Counter(0)
            }
        }

        #[derive(Clone, Debug)]
        struct Increment;

        impl Rankable for Increment {
            fn is_better_than(&self, _other: &Self) -> bool {
                false
            }
        }

        impl MoveToNeighbor<Untabued> for Increment {
            fn apply_to_solution(
                &self,
                _prob: &Untabued,
                sol: &mut Counter,
            ) -> Result<(), crate::error::OptError> {
                sol.0 += 1;
                Ok(())
            }
            fn iter(_p: &Untabued, _s: &Counter) -> impl Iterator<Item = Self> + Send {
                std::iter::once(Increment)
            }
            fn move_to_be_better_than(
                &self,
                _p: &Untabued,
                src: &Counter,
                other: &Counter,
            ) -> bool {
                src.0 + 1 > other.0
            }
        }

        /// Not implementing [`EnabledTabu`] must cost nothing at the point a
        /// move is applied — that is what keeps tabu an opt-in trait rather
        /// than a tax on every problem.
        #[test]
        fn a_move_without_a_tabu_policy_applies_normally() {
            let prob = Untabued;
            let mut state = SearchState::new_with_seed(&prob, 1);
            state.apply(&Increment).unwrap();
            state.apply_move_only(&Increment).unwrap();
            assert_eq!(state.solution.0, 2);
            assert_eq!(state.n_accepted, 2);
        }

        /// ...and reaching for the tabu map through such a move must say so,
        /// rather than quietly answering "allowed" — a `TabuSearch` over it
        /// would otherwise degrade into a plain best-move descent.
        #[test]
        fn a_move_without_a_tabu_policy_cannot_be_asked_about_the_map() {
            let prob = Untabued;
            let mut state = SearchState::new_with_seed(&prob, 1);

            // A read is a *compile* error for such a move — it cannot name a
            // `TabuMap` — so what is left to check at runtime is the erased
            // pair, which any `MoveToNeighbor` can reach.
            let required = state.require_tabu_policy(&Increment);
            assert!(
                matches!(required, Err(OptError::Unsupported(ref m)) if m.contains("Increment")),
                "expected an Unsupported error naming the move, got {required:?}"
            );
            assert!(matches!(
                state.record_tabu(&Increment),
                Err(OptError::Unsupported(_))
            ));
        }

        /// `apply` is what records, so a move is forbidden for exactly the
        /// `tenure` iterations that follow the one it was made on — the same
        /// boundary an explicit `record` at the pre-move iteration produced.
        #[test]
        fn apply_records_the_move_at_the_iteration_it_was_made_on() {
            let mc = triangle();
            let mut state = SearchState::new_with_seed(&mc, 3);
            state.set_tabu_tenure((5, 5));

            let m = first_flip(&mc, &state.solution);
            state.apply(&m).unwrap(); // made at iteration 0, so blocked through 5
            assert_eq!(state.iteration, 1);

            let probe = MaxCutFlipNeighbor { i: m.i, gain: 0.0 };
            assert!(!state.tabu_allows(&probe));
            for _ in 0..4 {
                state.progress_iteration();
            }
            assert_eq!(state.iteration, 5);
            assert!(!state.tabu_allows(&probe), "still blocked at 5");
            state.progress_iteration();
            assert!(state.tabu_allows(&probe), "free again at 6");
        }

        /// The default tenure remembers without forbidding: a search that never
        /// sets one is never blocked by its own moves.
        #[test]
        fn the_default_tenure_forbids_nothing() {
            let mc = triangle();
            let mut state = SearchState::new_with_seed(&mc, 3);
            assert_eq!(state.tabu_tenure(), (0, 0));

            let m = first_flip(&mc, &state.solution);
            state.apply(&m).unwrap();
            assert!(state.tabu_allows(&MaxCutFlipNeighbor { i: m.i, gain: 0.0 }));
        }

        /// Flip and swap both key on `TabuKey::Var`, so a vertex a flip forbade is
        /// forbidden to the swap too. Breakout Local Search is built on this:
        /// its weak swap must not undo what the descent's flips wrote.
        #[test]
        fn flip_and_swap_share_one_map() {
            let mc = triangle();
            let sol = MaxCutSolution::new_from_assignment(&mc, vec![true, false, false]);
            let mut state = SearchState::with_solution_and_seed(&mc, sol, 3);
            state.set_tabu_tenure((9, 9));

            state
                .apply(&MaxCutFlipNeighbor::new(&mc, &state.solution, 0))
                .unwrap();
            let swap = MaxCutSwapNeighbor::new(&mc, &state.solution, 0, 1);
            assert!(
                !state.tabu_allows(&swap),
                "the swap moves vertex 0, which the flip just forbade"
            );
        }

        /// `reset_tabu` frees every move and leaves the tenure in place.
        #[test]
        fn reset_tabu_keeps_the_tenure() {
            let mc = triangle();
            let mut state = SearchState::new_with_seed(&mc, 3);
            state.set_tabu_tenure((9, 9));
            let m = first_flip(&mc, &state.solution);
            state.apply(&m).unwrap();

            state.reset_tabu();
            assert!(state.tabu_allows(&MaxCutFlipNeighbor { i: m.i, gain: 0.0 }));
            assert_eq!(state.tabu_tenure(), (9, 9));
        }

        /// A sub-run is a new phase and starts free, while a plain clone is a
        /// copy and carries the prohibitions along.
        #[test]
        fn a_sub_run_starts_free_but_a_plain_clone_does_not() {
            let mc = triangle();
            let mut state = SearchState::new_with_seed(&mc, 3);
            state.set_tabu_tenure((9, 9));
            let m = first_flip(&mc, &state.solution);
            state.apply(&m).unwrap();
            let probe = MaxCutFlipNeighbor { i: m.i, gain: 0.0 };

            let copy = state.clone();
            assert!(!copy.tabu_allows(&probe));
            assert_eq!(copy.tabu_tenure(), (9, 9));

            let sub = state.clone_for_new_run(SearchStateCloneType::Simple);
            assert!(sub.tabu_allows(&probe));
            assert_eq!(sub.tabu_tenure(), (0, 0));
        }

        /// A sub-run shares the parent's iteration frame, so inheriting is a
        /// plain carry-over: a key blocked until 104 arrives blocked until 104,
        /// and the child — starting at the parent's own iteration — sees
        /// exactly the tenure that was left.
        #[test]
        fn inheriting_into_a_sub_run_carries_the_boundary_as_it_stands() {
            let mc = triangle();
            let mut state = SearchState::new_with_seed(&mc, 3);
            state.set_tabu_tenure((3, 3));
            for _ in 0..100 {
                state.progress_iteration();
            }
            let m = first_flip(&mc, &state.solution);
            state.apply(&m).unwrap(); // recorded at iteration 100, free again at 104
            let probe = MaxCutFlipNeighbor { i: m.i, gain: 0.0 };

            let mut sub = state.clone_for_new_run(SearchStateCloneType::ClearBest);
            assert_eq!(sub.iteration, state.iteration, "one frame, not two");
            sub.inherit_tabu_from(&state);

            while sub.iteration < 104 {
                assert!(!sub.tabu_allows(&probe), "blocked at {}", sub.iteration);
                sub.progress_iteration();
            }
            assert!(sub.tabu_allows(&probe), "free from 104, as in the parent");
        }

        /// A sub-run's prohibitions come back with its solution — and with the
        /// tenure it recorded them under. The boundaries need no translation
        /// because the phase ran in the parent's own frame, which is the whole
        /// reason `clone_for_new_run` no longer restarts the counter.
        #[test]
        fn update_state_carries_the_sub_runs_prohibitions_back() {
            let mc = triangle();
            let mut state = SearchState::new_with_seed(&mc, 3);
            for _ in 0..10 {
                state.progress_iteration();
            }

            let mut sub = state.clone_for_new_run(SearchStateCloneType::ClearBest);
            sub.set_tabu_tenure((3, 3));
            for _ in 0..100 {
                sub.progress_iteration();
            }
            let m = first_flip(&mc, &sub.solution);
            sub.apply(&m).unwrap(); // recorded at 110, free again at 114
            let probe = MaxCutFlipNeighbor { i: m.i, gain: 0.0 };
            assert!(!sub.tabu_allows(&probe));

            state.update_state(sub);
            assert_eq!(state.iteration, 111, "the parent absorbed the phase");
            assert_eq!(state.tabu_tenure(), (3, 3), "the tenure comes back too");
            while state.iteration < 114 {
                assert!(
                    !state.tabu_allows(&probe),
                    "still blocked at {}",
                    state.iteration
                );
                state.progress_iteration();
            }
            assert!(
                state.tabu_allows(&probe),
                "free from 114, as in the sub-run"
            );
        }
    }

    /// Crossing a [`ProblemReduction`](crate::trait_defs::ProblemReduction) is
    /// a search-state operation, and these pin what "consistent" means for it.
    /// `MaxCutKernel` is the one implementation the core library has.
    mod reduction_crossing {
        use super::*;
        use crate::common::hamming_distance;
        use crate::problem::MaxCutKernel;
        use crate::trait_defs::ProblemReduction;
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};

        /// A sparse instance, i.e. one the kernel rules actually reduce.
        fn reducible_instance(seed: u64, n: usize) -> MaxCut {
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut edges = Vec::new();
            for i in 0..n {
                for j in (i + 1)..n {
                    if rng.random_bool(2.5 / n as f64) {
                        edges.push((i, j, 1.0));
                    }
                }
            }
            edges.push((n - 1, n - 2, 1.0));
            MaxCut::from_edges(edges)
        }

        /// Closing must land on exactly the solution a from-scratch rebuild of
        /// the lifted assignment would produce, caches included — and must
        /// charge nothing for landing there.
        ///
        /// `lift` returns a complete `Solution`, so installing it needs no
        /// reconstruction and is not a move: the sub-run's own counters are the
        /// whole cost of the crossing. (An earlier version walked to the lifted
        /// assignment one flip at a time and charged each one, which paid for a
        /// rebuild `lift` had already done and inflated `iteration` with moves
        /// no search made.)
        #[test]
        fn close_reduction_lands_on_the_lifted_solution_with_exact_caches() {
            let mc = reducible_instance(11, 300);
            let kernel = MaxCutKernel::new(&mc);
            assert!(!kernel.is_trivial(), "the instance must actually reduce");

            let mut state = SearchState::new_with_seed(&mc, 3);
            let sub = state.open_reduction(&kernel);
            let before_x = state.solution.x.clone();
            let before_iteration = state.iteration;

            state.close_reduction(&kernel, &sub);

            let target = kernel.lift(&sub.best_solution.x);
            let rebuilt = MaxCutSolution::new_from_assignment(&mc, target.clone());
            assert_eq!(state.solution.x, rebuilt.x, "assignments diverged");
            assert_eq!(state.solution.gain, rebuilt.gain, "gain caches diverged");
            assert_eq!(
                state.solution.objective, rebuilt.objective,
                "objectives diverged"
            );
            assert!(
                hamming_distance(&before_x, &target) > 0,
                "the lifted solution must actually differ, or this proves nothing"
            );
            assert_eq!(
                state.iteration, before_iteration,
                "installing the lifted solution is not a move and must cost nothing"
            );
        }

        /// A warm start opened on the target must reproduce the reduction's own
        /// projection, and must consume exactly one draw from the parent's RNG
        /// — the sub-run's whole trajectory hangs off that seed.
        #[test]
        fn open_reduction_projects_the_incumbent_and_draws_one_seed() {
            let mc = reducible_instance(2, 200);
            let kernel = MaxCutKernel::new(&mc);

            let mut state = SearchState::new_with_seed(&mc, 5);
            // A probe advanced by exactly one draw: `open_reduction` must leave
            // the state's RNG in the same place.
            let mut probe = state.rng.clone();
            rand::RngCore::next_u64(&mut probe);

            let sub = state.open_reduction(&kernel);
            assert_eq!(
                sub.solution.x,
                ProblemReduction::project(&kernel, &state.solution).x,
                "the warm start must be the reduction's projection"
            );
            assert_eq!(sub.iteration, 0, "a sub-state starts from zeroed counters");
            assert_eq!(
                rand::RngCore::next_u64(&mut state.rng),
                rand::RngCore::next_u64(&mut probe),
                "opening must consume exactly one draw"
            );
        }

        /// `close_reduction` must carry the sub-run's accounting across —
        /// exactly that, and nothing else. Dropping it is invisible in the
        /// objective and shows up as a benchmark reporting near-zero counters
        /// precisely on the instances where the reduction did something.
        #[test]
        fn close_reduction_charges_the_sub_run() {
            let mc = reducible_instance(4, 250);
            let kernel = MaxCutKernel::new(&mc);
            let mut state = SearchState::new_with_seed(&mc, 1);

            // A hand-rolled sub-run rather than a real heuristic: what is being
            // pinned is the accounting of the crossing, and this module has no
            // business knowing what a heuristic is.
            let mut sub = state.open_reduction(&kernel);
            for _ in 0..20 {
                let m = first_flip(kernel.kernel(), &sub.solution);
                sub.apply(&m).unwrap();
            }
            assert!(sub.iteration > 0, "the sub-run must have done something");

            let before_iteration = state.iteration;
            let before_x = state.solution.x.clone();
            state.close_reduction(&kernel, &sub);

            assert!(
                hamming_distance(&before_x, &state.solution.x) > 0,
                "the crossing must have moved the solution"
            );
            assert_eq!(state.iteration, before_iteration + sub.iteration);
            assert_eq!(
                state.best_solution.objective,
                mc.calculate_cut_size(&state.best_solution.x),
                "the installed solution's cached objective must be exact"
            );
        }
    }
}
