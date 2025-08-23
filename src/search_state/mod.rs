pub mod specific_trait;

pub use specific_trait::{EnabledTabu, EnumerateMoveToNeighbor, Evaluable, ProblemTrait};

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
    pub objective: Problem::Objective,
    pub best_time: std::time::Instant,
    pub best_iteration: u64,
    pub best_solution: Problem::Solution,
    pub best_objective: Problem::Objective,
}

impl<'a, Problem> SearchState<'a, Problem>
where
    Problem: ProblemTrait,
    Problem::Solution: Clone,
{
    pub fn new(instance: &'a Problem, mut rng: rand::rngs::ThreadRng) -> Self {
        let solution = instance.new_solution(&mut rng);
        let best_solution = solution.clone();
        let objective = instance.calculate_objective(&solution);
        let best_objective = objective.clone();
        Self {
            start_iteration: 0,
            start_time: std::time::Instant::now(),
            instance,
            iteration: 0,
            solution,
            objective,
            best_time: std::time::Instant::now(),
            best_iteration: 0,
            best_solution,
            best_objective,
        }
    }

    pub fn duration(&self) -> std::time::Duration {
        std::time::Instant::now() - self.start_time
    }

    pub fn update_best(&mut self) -> bool {
        let ret = self
            .instance
            .is_first_objective_better_than_second(&self.objective, &self.best_objective);

        if ret {
            self.best_solution = self.solution.clone();
            self.best_objective = self.objective.clone();
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
                objective: self.objective.clone(),
                best_time: self.best_time,
                best_iteration: self.best_iteration,
                best_solution: self.best_solution.clone(),
                best_objective: self.best_objective.clone(),
            },
            SearchStateCloneType::ClearBest => Self {
                start_iteration: 0,
                start_time: now,
                instance: self.instance,
                iteration: 0,
                solution: self.solution.clone(),
                objective: self.objective.clone(),
                best_time: now,
                best_iteration: 0,
                best_solution: self.solution.clone(),
                best_objective: self.objective.clone(),
            },
            SearchStateCloneType::StartBest => Self {
                start_iteration: 0,
                start_time: now,
                instance: self.instance,
                iteration: 0,
                solution: self.best_solution.clone(),
                objective: self.best_objective.clone(),
                best_time: now,
                best_iteration: 0,
                best_solution: self.best_solution.clone(),
                best_objective: self.best_objective.clone(),
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
        self.objective = cloned_state.objective;

        // add iteration into the current iteration
        self.iteration += cloned_state.iteration - cloned_state.start_iteration;

        // update the best solution if the one of the new state is better than the current
        if self.instance.is_first_objective_better_than_second(
            &cloned_state.best_objective,
            &self.best_objective,
        ) {
            self.best_solution = cloned_state.best_solution;
            self.best_objective = cloned_state.best_objective;
            self.best_time = cloned_state.best_time;
            self.best_iteration =
                self.iteration + cloned_state.best_iteration - cloned_state.start_iteration;
        }
    }

    pub fn apply<MoveToNeighbor>(&mut self, neighbor: &MoveToNeighbor)
    where
        Self: EnumerateMoveToNeighbor<MoveToNeighbor>,
    {
        self.apply_to_iteration(neighbor);
        self.apply_to_solution(neighbor);
        self.apply_to_objective(neighbor);
        self.update_best();
    }

    pub fn progress_iteration(&mut self) {
        self.iteration += 1;
    }
}

impl<'a, Problem> std::fmt::Debug for SearchState<'a, Problem>
where
    Problem: ProblemTrait,
    Problem::Objective: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchState")
            .field(
                "current",
                &(
                    self.start_time.elapsed(),
                    self.iteration,
                    self.objective.clone(),
                ),
            )
            .field(
                "best",
                &(
                    self.best_time - self.start_time,
                    self.best_iteration,
                    self.best_objective.clone(),
                ),
            )
            .finish()
    }
}
