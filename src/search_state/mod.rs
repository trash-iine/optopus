pub mod specific_trait;

use rayon::prelude::*;
pub use specific_trait::{
    filter_best, EnabledTabu, EnumerateMoveToNeighbor, Evaluable, MoveToNeigbor, ProblemTrait,
    Rankable,
};

/// The type of the search state clone
#[derive(Clone, Debug)]
pub enum SearchStateCloneType {
    /// Just clone the state.
    ///
    /// - Start from the current solution
    /// - Keep the start time
    /// - Keep the current iteration
    /// - Keep the best solution
    Simple,

    /// Clear the best solution.
    ///
    /// - Start from the current solution
    /// - Clear the start time
    /// - Clear the current iteration
    /// - Clear the best solution (set the current solution to the best)
    ClearBest,

    /// Start from the best solution.
    ///
    /// - Start from the best solution
    /// - Clear the start time
    /// - Clear the current iteration
    /// - Keep the best solution
    StartBest,
}

#[derive(Clone)]
pub struct SearchState<'a, Problem>
where
    Problem: ProblemTrait,
{
    pub start_iteration: u64,
    pub start_time: std::time::Instant,
    pub instance: &'a Problem,
    pub iteration: u64,
    pub solution: Problem::Solution,
    pub best_time: std::time::Instant,
    pub best_iteration: u64,
    pub best_solution: Problem::Solution,
}

impl<'a, Problem> SearchState<'a, Problem>
where
    Problem: ProblemTrait,
{
    pub fn new(instance: &'a Problem) -> Self {
        let solution = instance.new_solution(&mut rand::rng());
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

    pub fn duration(&self) -> std::time::Duration {
        std::time::Instant::now() - self.start_time
    }

    pub fn update_best(&mut self) -> bool {
        let ret = self.solution.is_better_than(&self.best_solution);

        if ret {
            self.best_solution = self.solution.clone();
            self.best_time = std::time::Instant::now();
            self.best_iteration = self.iteration;
        }

        return ret;
    }

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

    pub fn update(&mut self, cloned_state: Self) {
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
        self.iteration += cloned_state.iteration - cloned_state.start_iteration;

        // update the best solution if the one of the new state is better than the current
        if cloned_state
            .best_solution
            .is_better_than(&self.best_solution)
        {
            self.best_solution = cloned_state.best_solution;
            self.best_time = cloned_state.best_time;
            self.best_iteration =
                self.iteration + cloned_state.best_iteration - cloned_state.start_iteration;
        }
    }

    pub fn apply<Move>(&mut self, neighbor: &Move)
    where
        Move: MoveToNeigbor<Problem>,
    {
        self.iteration = neighbor.apply_to_iteration(self.iteration);
        neighbor.apply_to_solution(&self.instance, &mut self.solution);
        self.update_best();
    }

    pub fn progress_iteration(&mut self) {
        self.iteration += 1;
    }

    pub fn is_neighbor_better_than_current<Move>(&self, m: &Move) -> bool
    where
        Move: MoveToNeigbor<Problem>,
    {
        m.move_to_be_better_than(&self.instance, &self.solution, &self.solution)
    }

    pub fn is_neighbor_better_than_best<Move>(&self, m: &Move) -> bool
    where
        Move: MoveToNeigbor<Problem>,
    {
        m.move_to_be_better_than(&self.instance, &self.solution, &self.best_solution)
    }

    pub fn get_best_move_par_chunks<MoveToNeighbor>(
        &self,
        move_list: impl Iterator<Item = MoveToNeighbor>,
        chunk_size: usize,
    ) -> Option<MoveToNeighbor>
    where
        Self: EnumerateMoveToNeighbor<MoveToNeighbor>,
        MoveToNeighbor: Send + Sync + Clone,
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
                        if self.is_first_move_better_than_second(first, second) {
                            std::cmp::Ordering::Greater
                        } else if self.is_first_move_better_than_second(second, first) {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Equal
                        }
                    })
                    .unwrap()
            })
            .max_by(|first, second| {
                if self.is_first_move_better_than_second(first, second) {
                    std::cmp::Ordering::Greater
                } else if self.is_first_move_better_than_second(second, first) {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            });

        if let Some(v) = opt {
            Some(v.clone())
        } else {
            None
        }
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
