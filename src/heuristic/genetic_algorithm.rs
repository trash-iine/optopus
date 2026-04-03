use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::{
    Crossover, ProblemTrait, Rankable, SearchState, SearchStateCloneType, SubProblemExtractable,
};

/// Genetic algorithm meta-heuristic.
///
/// Maintains a population of `population_size` candidate solutions.
/// Each call to `run_once`:
/// 1. **Selection**: picks two parents by tournament selection.
/// 2. **Crossover**: combines them with operator `C` to produce an offspring.
/// 3. **Mutation**: applies the inner `mutation` heuristic to the offspring
///    using the sub-run clone/merge pattern (same as [`crate::heuristic::Iterated`]).
/// 4. **Replacement**: inserts the (possibly improved) offspring into the population,
///    evicting the worst member when at capacity.
///
/// The global best solution is tracked in `SearchState::best_solution`.
///
/// # Type parameters
///
/// - `P` — the problem type; must implement [`ProblemTrait`].
/// - `C` — the crossover operator; must implement [`Crossover<P>`].
///
/// # Example
///
/// ```rust,ignore
/// use optopus::heuristic::{GeneticAlgorithm, LocalSearch, StopCondition, SubProblemBasedCrossover};
/// use optopus::problem::{MaxCut, MaxCutFlipNeighbor};
/// use optopus::search_state::SearchState;
///
/// let mut ga = GeneticAlgorithm::new(
///     StopCondition::iterations(10_000),
///     50,
///     SubProblemBasedCrossover {
///         sub_heuristic: Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
///             StopCondition::failed_updates(1),
///         )),
///     },
///     Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
///         StopCondition::failed_updates(1),
///     )),
/// );
/// ga.run(&mut state).unwrap();
/// ```
pub struct GeneticAlgorithm<P: ProblemTrait, C> {
    pub stop_condition: StopCondition,
    pub population_size: usize,
    /// Crossover operator stored as a value because `Crossover::crossover` takes `&mut self`.
    pub crossover: C,
    /// Mutation operator — any [`Heuristic<P>`] works (local search, SA, random walk, …).
    pub mutation: Box<dyn Heuristic<P>>,
    population: Vec<P::Solution>,
    /// Index of the best solution in `population`. Tracked incrementally to avoid O(n) scans.
    best_idx: Option<usize>,
}

impl<P: ProblemTrait, C> GeneticAlgorithm<P, C> {
    pub fn new(
        stop_condition: StopCondition,
        population_size: usize,
        crossover: C,
        mutation: Box<dyn Heuristic<P>>,
    ) -> Self {
        assert!(population_size >= 2, "population_size must be at least 2");
        Self {
            stop_condition,
            population_size,
            crossover,
            mutation,
            population: Vec::new(),
            best_idx: None,
        }
    }

    fn initialize_population(&mut self, instance: &P, rng: &mut impl rand::Rng) {
        while self.population.len() < self.population_size {
            self.population.push(instance.new_solution(rng));
        }

        self.best_idx = self
            .population
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                if a.is_better_than(b) {
                    std::cmp::Ordering::Greater
                } else if b.is_better_than(a) {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .map(|(i, _)| i);
    }

    /// Returns a reference to the two tournament winners (may be the same individual).
    fn tournament_select<'s>(&'s self, rng: &mut impl rand::Rng) -> (&'s P::Solution, &'s P::Solution) {
        let i = rng.random_range(0..self.population.len());
        let j = rng.random_range(0..self.population.len());
        let a = &self.population[i];
        let b = &self.population[j];
        let winner_a = if a.is_better_than(b) { a } else { b };

        let k = rng.random_range(0..self.population.len());
        let l = rng.random_range(0..self.population.len());
        let c = &self.population[k];
        let d = &self.population[l];
        let winner_b = if c.is_better_than(d) { c } else { d };

        (winner_a, winner_b)
    }

    /// Inserts `offspring` into the population.
    /// If the population is at capacity, replaces the worst member (if offspring is better).
    /// Maintains `best_idx` incrementally to avoid an O(n) best-member scan in `run_once`.
    fn insert_into_population(&mut self, offspring: P::Solution) {
        if self.population.len() < self.population_size {
            let new_idx = self.population.len();
            let is_new_best = match self.best_idx {
                None => true,
                Some(b) => offspring.is_better_than(&self.population[b]),
            };
            if is_new_best {
                self.best_idx = Some(new_idx);
            }
            self.population.push(offspring);
            return;
        }

        // Find the worst member (O(n) — unavoidable without a heap).
        let worst_idx = self
            .population
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                if a.is_better_than(b) {
                    std::cmp::Ordering::Greater
                } else if b.is_better_than(a) {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .map(|(i, _)| i)
            .unwrap();

        if offspring.is_better_than(&self.population[worst_idx]) {
            self.population[worst_idx] = offspring;

            // Update best_idx:
            // - If the replaced slot was the previous best, rescan (rare edge case).
            // - Otherwise compare offspring with current best.
            if self.best_idx == Some(worst_idx) {
                self.best_idx = self
                    .population
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        if a.is_better_than(b) {
                            std::cmp::Ordering::Greater
                        } else if b.is_better_than(a) {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Equal
                        }
                    })
                    .map(|(i, _)| i);
            } else if self.population[worst_idx]
                .is_better_than(&self.population[self.best_idx.unwrap()])
            {
                self.best_idx = Some(worst_idx);
            }
        }
    }
}

impl<P, C> Heuristic<P> for GeneticAlgorithm<P, C>
where
    P: ProblemTrait,
    C: Crossover<P>,
{
    /// Clears the population so the next `run` starts fresh.
    fn clear(&mut self) {
        self.population.clear();
        self.best_idx = None;
    }

    fn is_done<'a>(&self, state: &SearchState<'a, P>) -> bool {
        self.stop_condition.is_done(state)
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        let mut rng = rand::rng();

        // --- Initialise population on first call ---
        if self.population.len() < self.population_size {
            self.initialize_population(state.instance, &mut rng);
        }

        // --- Selection ---
        let (parent_a, parent_b) = self.tournament_select(&mut rng);
        let parent_a = parent_a.clone();
        let parent_b = parent_b.clone();

        // --- Crossover ---
        let offspring = self.crossover.crossover(state.instance, &parent_a, &parent_b);

        // --- Mutation (sub-run clone/merge pattern from Iterated) ---
        state.solution = offspring;
        let mut sub_state = state.clone_for_new_run(SearchStateCloneType::ClearBest);
        self.mutation.run(&mut sub_state)?;
        let mutated = sub_state.best_solution.clone();
        state.update_state(sub_state);

        // --- Population replacement ---
        self.insert_into_population(mutated);

        // Keep state.solution pointing at the current population best (O(1) via best_idx).
        state.solution = self.population[self.best_idx.unwrap()].clone();
        state.update_best();

        Ok(())
    }
}

// ─── SubProblemBasedCrossover ─────────────────────────────────────────────────

/// Generic crossover operator that works for any problem implementing [`SubProblemExtractable`].
///
/// For each crossover call it:
/// 1. Extracts a sub-problem containing only the variables that differ between the two parents.
/// 2. Solves the sub-problem with `sub_heuristic`.
/// 3. Lifts the sub-solution back to the full solution space.
///
/// # MaxCut example
///
/// - Vertices with the same side in both parents are fixed; their edges become bias terms.
/// - Vertices with different sides form the sub-MaxCut instance.
/// - `lift_solution` merges the fixed sides with the sub-problem result.
pub struct SubProblemBasedCrossover<P: SubProblemExtractable> {
    /// Heuristic used to solve the sub-problem (e.g. [`crate::heuristic::LocalSearch`]).
    pub sub_heuristic: Box<dyn Heuristic<P>>,
}

impl<P: SubProblemExtractable> Crossover<P> for SubProblemBasedCrossover<P> {
    fn crossover(
        &mut self,
        prob: &P,
        sol1: &P::Solution,
        sol2: &P::Solution,
    ) -> P::Solution {
        let sub_prob = prob.extract_sub_problem(sol1, sol2);
        let mut sub_state = SearchState::new(&sub_prob);
        self.sub_heuristic
            .run(&mut sub_state)
            .expect("sub_heuristic failed inside SubProblemBasedCrossover");
        prob.lift_solution(sol1, sol2, &sub_state.best_solution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::LocalSearch;
    use crate::problem::{MaxCut, MaxCutFlipNeighbor, MaxCutSolution};

    struct CloneFirstParent;

    impl Crossover<MaxCut> for CloneFirstParent {
        fn crossover(
            &mut self,
            _prob: &MaxCut,
            sol1: &MaxCutSolution,
            _sol2: &MaxCutSolution,
        ) -> MaxCutSolution {
            sol1.clone()
        }
    }

    #[test]
    fn genetic_algorithm_initializes_best_idx_before_first_replacement() {
        let mut mc = MaxCut::new();
        mc.add_weight(0, 1, 1.0);
        mc.add_weight(0, 2, 1.0);
        mc.add_weight(1, 2, 1.0);

        let mut state = SearchState::new(&mc);
        let mut ga = GeneticAlgorithm::new(
            StopCondition::iterations(1),
            4,
            CloneFirstParent,
            Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
                StopCondition::failed_updates(1),
            )),
        );

        ga.run(&mut state).unwrap();

        assert!(ga.best_idx.is_some());
        assert!(state.best_solution.objective >= 0.0);
    }
}
