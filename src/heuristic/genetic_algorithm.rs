use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::{Crossover, ProblemTrait, Rankable, SearchState, SearchStateCloneType};

/// Genetic algorithm meta-heuristic.
///
/// Maintains a population of `population_size` candidate solutions.
/// On the first `run_once` call the population is seeded with random solutions
/// (optionally refined by `init_improvement`). Each subsequent call:
/// 1. **Selection**: picks two parents by tournament selection.
/// 2. **Crossover**: combines them with operator `C` to produce an offspring.
/// 3. **Mutation**: applies the inner `mutation` heuristic to the offspring
///    using the sub-run clone/merge pattern (same as [`crate::heuristic::Iterated`]).
/// 4. **Replacement**: inserts the (possibly improved) offspring into the population,
///    evicting the worst member when at capacity.
///
/// When `init_improvement` is `Some`, each random initial individual is also passed
/// through that heuristic via the same sub-run pattern. This reproduces the
/// Galinier-Hao Hybrid Evolutionary Algorithm (HEA) for graph colouring when paired
/// with a [`crate::heuristic::TabuSearch`] mutation operator.
///
/// The global best solution is tracked in `SearchState::best_solution`.
///
/// # References
///
/// - Holland, J. H. *Adaptation in Natural and Artificial Systems*. University of Michigan Press, 1975.
/// - Goldberg, D. E. *Genetic Algorithms in Search, Optimization, and Machine Learning*.
///   Addison-Wesley, 1989.
/// - Galinier, P. and Hao, J.-K. "Hybrid Evolutionary Algorithms for Graph Coloring."
///   *Journal of Combinatorial Optimization*, 3(4), 379-397, 1999.
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
    /// Optional per-individual local-improvement applied to each random seed during
    /// population initialisation. `None` (default) means the initial population is
    /// pure random; `Some(op)` reproduces the HEA pattern.
    pub init_improvement: Option<Box<dyn Heuristic<P>>>,
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
        Self::new_with_init(stop_condition, population_size, crossover, mutation, None)
    }

    /// Like [`Self::new`] but also runs `init_improvement` on every member of the
    /// initial random population. Pass `Some(Box::new(TabuSearch::new(...)))` to
    /// recover the Galinier-Hao HEA configuration.
    pub fn new_with_init(
        stop_condition: StopCondition,
        population_size: usize,
        crossover: C,
        mutation: Box<dyn Heuristic<P>>,
        init_improvement: Option<Box<dyn Heuristic<P>>>,
    ) -> Self {
        assert!(population_size >= 2, "population_size must be at least 2");
        Self {
            stop_condition,
            population_size,
            crossover,
            mutation,
            init_improvement,
            population: Vec::new(),
            best_idx: None,
        }
    }

    /// Sub-run clone/merge pattern shared by population init and offspring mutation.
    /// Sets `state.solution = seed`, runs `op` on a `ClearBest` sub-state, then
    /// merges iteration counters back into `state` and returns the refined solution.
    fn improve_via_sub_run<'a>(
        state: &mut SearchState<'a, P>,
        seed: P::Solution,
        op: &mut dyn Heuristic<P>,
    ) -> Result<P::Solution, OptError> {
        state.solution = seed;
        let mut sub_state = state.clone_for_new_run(SearchStateCloneType::ClearBest);
        op.run(&mut sub_state)?;
        let result = sub_state.best_solution.clone();
        state.update_state(sub_state);
        Ok(result)
    }

    fn initialize_population<'a>(
        &mut self,
        state: &mut SearchState<'a, P>,
        rng: &mut impl rand::Rng,
    ) -> Result<(), OptError> {
        while self.population.len() < self.population_size {
            let seed = state.instance.new_solution(rng);
            let member = match self.init_improvement.as_mut() {
                Some(op) => Self::improve_via_sub_run(state, seed, op.as_mut())?,
                None => seed,
            };
            self.population.push(member);
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
        Ok(())
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
            self.initialize_population(state, &mut rng)?;
        }

        // --- Selection ---
        let (parent_a, parent_b) = self.tournament_select(&mut rng);
        let parent_a = parent_a.clone();
        let parent_b = parent_b.clone();

        // --- Crossover ---
        let offspring = self.crossover.crossover(state.instance, &parent_a, &parent_b);

        // --- Mutation (sub-run clone/merge pattern from Iterated) ---
        let mutated = Self::improve_via_sub_run(state, offspring, self.mutation.as_mut())?;

        // --- Population replacement ---
        self.insert_into_population(mutated);

        // Keep state.solution pointing at the current population best (O(1) via best_idx).
        state.solution = self.population[self.best_idx.unwrap()].clone();
        state.update_best();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Graph;
    use crate::heuristic::{LocalSearch, TabuSearch};
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
        let mc = MaxCut::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]);
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

    #[test]
    fn genetic_algorithm_with_init_improvement_refines_initial_population() {
        let mc = MaxCut::new(Graph::from_edges([
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 2, 1.0),
            (1, 3, 1.0),
            (2, 3, 1.0),
        ]));
        let mut state = SearchState::new(&mc);

        let mut hea = GeneticAlgorithm::new_with_init(
            StopCondition::iterations(4),
            4,
            CloneFirstParent,
            Box::new(TabuSearch::<MaxCutFlipNeighbor>::new(
                StopCondition::failed_updates(1),
                (1, 5),
                None,
            )),
            Some(Box::new(TabuSearch::<MaxCutFlipNeighbor>::new(
                StopCondition::failed_updates(1),
                (1, 5),
                None,
            ))),
        );

        hea.run(&mut state).unwrap();

        assert!(hea.best_idx.is_some());
        assert_eq!(hea.population.len(), 4);
        assert!(state.best_solution.objective >= 0.0);
    }

    #[test]
    #[should_panic(expected = "population_size must be at least 2")]
    fn genetic_algorithm_rejects_population_size_one() {
        let _ = GeneticAlgorithm::<MaxCut, CloneFirstParent>::new(
            StopCondition::iterations(1),
            1,
            CloneFirstParent,
            Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
                StopCondition::failed_updates(1),
            )),
        );
    }
}
