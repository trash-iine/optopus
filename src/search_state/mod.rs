//! Search state management and core traits for combinatorial optimization.

pub mod specific_trait;

use rayon::prelude::*;
pub use specific_trait::{
    Crossover, EnabledTabu, Evaluate, Evaluable, MoveToNeighbor, ProblemTrait, Rankable,
    SubProblemExtractable, filter_best,
};

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
}

impl<'a, Problem> SearchState<'a, Problem>
where
    Problem: ProblemTrait,
{
    /// Creates a new [`SearchState`] with a randomly generated initial solution.
    pub fn new(instance: &'a Problem) -> Self {
        let solution = instance.new_solution(&mut rand::rng());
        let best_solution = solution.clone();
        let state = Self {
            start_iteration: 0,
            start_time: std::time::Instant::now(),
            instance,
            iteration: 0,
            solution,
            best_time: std::time::Instant::now(),
            best_iteration: 0,
            best_solution,
        };
        tracing::debug!("SearchState initialized");
        state
    }

    /// Creates a new [`SearchState`] starting from a specific solution.
    ///
    /// Use this for warm starts, deterministic testing, or when a known-good solution
    /// should be the starting point. The provided solution is also set as the initial best.
    pub fn with_solution(instance: &'a Problem, solution: Problem::Solution) -> Self {
        let best_solution = solution.clone();
        Self {
            start_iteration: 0,
            start_time: std::time::Instant::now(),
            instance,
            iteration: 0,
            solution,
            best_time: std::time::Instant::now(),
            best_iteration: 0,
            best_solution,
        }
    }

    /// Returns the elapsed time since the current sub-run started.
    pub fn duration(&self) -> std::time::Duration {
        std::time::Instant::now() - self.start_time
    }

    /// Updates the best solution if the current solution is better.
    ///
    /// Returns `true` if the best solution was updated.
    pub fn update_best(&mut self) -> bool {
        let ret = self.solution.is_better_than(&self.best_solution);

        if ret {
            self.best_solution = self.solution.clone();
            self.best_time = std::time::Instant::now();
            self.best_iteration = self.iteration;
            tracing::debug!(
                iteration = self.best_iteration,
                elapsed_secs = self.duration().as_secs_f64(),
                "Best solution updated"
            );
        }

        return ret;
    }

    /// Creates a copy of this state prepared for a new sub-run.
    ///
    /// The behavior depends on `clone_type`; see [`SearchStateCloneType`] for details.
    pub fn clone_for_new_run(&self, clone_type: SearchStateCloneType) -> Self {
        let now = std::time::Instant::now();
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
            },
            SearchStateCloneType::ClearBest => Self {
                start_iteration: 0,
                start_time: now,
                instance: self.instance,
                iteration: 0,
                solution: self.solution.clone(),
                best_time: now,
                best_iteration: 0,
                best_solution: self.solution.clone(),
            },
            SearchStateCloneType::StartBest => Self {
                start_iteration: 0,
                start_time: now,
                instance: self.instance,
                iteration: 0,
                solution: self.best_solution.clone(),
                best_time: now,
                best_iteration: 0,
                best_solution: self.best_solution.clone(),
            },
        }
    }

    /// Merges the results of a completed sub-run back into this state.
    ///
    /// - The current solution is replaced with `cloned_state.solution`.
    /// - The iteration counter is incremented by the iterations performed in the sub-run.
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
        let sub_run_best_offset = cloned_state.best_iteration - cloned_state.start_iteration;
        let old_iteration = self.iteration;
        self.iteration += cloned_state.iteration - cloned_state.start_iteration;

        // update the best solution if the one of the new state is better than the current
        if cloned_state
            .best_solution
            .is_better_than(&self.best_solution)
        {
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
    pub fn apply<Move>(&mut self, neighbor: &Move) -> Result<(), crate::error::OptError>
    where
        Move: MoveToNeighbor<Problem>,
    {
        self.iteration = neighbor.apply_to_iteration(self.iteration);
        neighbor.apply_to_solution(&self.instance, &mut self.solution)?;
        self.update_best();
        Ok(())
    }

    /// Applies a neighborhood move and updates the iteration counter, but does
    /// **not** refresh the best solution.
    ///
    /// Use this in perturbation phases where moves intentionally diversify and
    /// a best-solution update is deferred until the phase completes. Call
    /// [`update_best`](Self::update_best) once after the phase ends.
    pub fn apply_move_only<Move>(&mut self, neighbor: &Move) -> Result<(), crate::error::OptError>
    where
        Move: MoveToNeighbor<Problem>,
    {
        self.iteration = neighbor.apply_to_iteration(self.iteration);
        neighbor.apply_to_solution(&self.instance, &mut self.solution)?;
        Ok(())
    }

    /// Increments the iteration counter by one without applying any move.
    pub fn progress_iteration(&mut self) {
        self.iteration += 1;
    }

    /// Returns `true` if applying `m` to the current solution yields a solution
    /// better than the current solution.
    pub fn is_neighbor_better_than_current<Move>(&self, m: &Move) -> bool
    where
        Move: MoveToNeighbor<Problem>,
    {
        m.move_to_be_better_than(&self.instance, &self.solution, &self.solution)
    }

    /// Returns `true` if applying `m` to the current solution yields a solution
    /// better than the best solution found so far.
    pub fn is_neighbor_better_than_best<Move>(&self, m: &Move) -> bool
    where
        Move: MoveToNeighbor<Problem>,
    {
        m.move_to_be_better_than(&self.instance, &self.solution, &self.best_solution)
    }

    /// Returns the best move from `move_list` using parallel chunk-based evaluation.
    pub fn get_best_move_par_chunks<M>(
        &self,
        move_list: impl Iterator<Item = M>,
        chunk_size: usize,
    ) -> Option<M>
    where
        M: Send + Sync + Clone + Rankable,
        Problem: Sync,
        Problem::Solution: Sync,
    {
        let move_vec: Vec<_> = move_list.collect();
        let opt = move_vec
            .par_chunks(chunk_size)
            .map(|chunk| {
                chunk
                    .into_iter()
                    .max_by(|first, second| {
                        if first.is_better_than(second) {
                            std::cmp::Ordering::Greater
                        } else if second.is_better_than(first) {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Equal
                        }
                    })
                    .unwrap()
            })
            .max_by(|first, second| {
                if first.is_better_than(second) {
                    std::cmp::Ordering::Greater
                } else if second.is_better_than(first) {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            });

        opt.cloned()
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
