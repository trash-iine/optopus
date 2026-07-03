use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::{
    Crossover, Distance, ProblemTrait, Rankable, SearchState, SearchStateCloneType,
};

/// How [`GeneticAlgorithm`] picks the two parents for each crossover step.
#[derive(Clone, Copy, Debug, Default)]
pub enum ParentSelection {
    /// Two independent binary tournaments.
    ///
    /// For each parent: sample two indices uniformly with replacement and
    /// keep the better-ranked one. Standard textbook GA selection.
    #[default]
    Tournament,
    /// Diversity-aware selection.
    ///
    /// 1. Pick parent A uniformly at random from the population.
    /// 2. Compute the [`Distance`] from A to every other member.
    /// 3. Pick parent B uniformly at random from the `top_k` members with
    ///    the largest distance (clamped to `population_size - 1`).
    ///
    /// Promotes exploration by avoiding crossover between near-identical
    /// individuals. Requires `P::Solution: Distance`.
    DistantTopK { top_k: usize },
}

/// Genetic algorithm meta-heuristic.
///
/// Maintains a population of `population_size` candidate solutions.
/// On the first `run_once` call the population is seeded with random solutions
/// (optionally refined by `init_improvement`). Each subsequent call:
/// 1. Selection: picks two parents by tournament selection.
/// 2. Crossover: combines them with operator `C` to produce an offspring.
/// 3. Mutation: applies the inner `mutation` heuristic to the offspring
///    using the sub-run clone/merge pattern (same as [`crate::heuristic::Iterated`]).
/// 4. Replacement: inserts the (possibly improved) offspring into the population,
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
    /// Strategy for sampling the two parents each iteration.
    pub parent_selection: ParentSelection,
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
            parent_selection: ParentSelection::default(),
            population: Vec::new(),
            best_idx: None,
        }
    }

    /// Builder-style override of the parent-selection strategy.
    pub fn with_parent_selection(mut self, strategy: ParentSelection) -> Self {
        self.parent_selection = strategy;
        self
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
    ) -> Result<(), OptError> {
        while self.population.len() < self.population_size {
            let seed = state.instance.new_solution(&mut state.rng);
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
            .max_by(|(_, a), (_, b)| crate::trait_defs::rank_cmp(*a, *b))
            .map(|(i, _)| i);
        Ok(())
    }

    /// Returns indices of the two parents according to `self.parent_selection`.
    /// May return the same index twice (rare, but allowed — matches the original tournament).
    fn select_parent_indices(&self, rng: &mut impl rand::Rng) -> (usize, usize)
    where
        P::Solution: Distance,
    {
        match self.parent_selection {
            ParentSelection::Tournament => self.tournament_indices(rng),
            ParentSelection::DistantTopK { top_k } => self.distant_top_k_indices(rng, top_k),
        }
    }

    fn tournament_indices(&self, rng: &mut impl rand::Rng) -> (usize, usize) {
        (self.tournament_one(rng), self.tournament_one(rng))
    }

    fn tournament_one(&self, rng: &mut impl rand::Rng) -> usize {
        let i = rng.random_range(0..self.population.len());
        let j = rng.random_range(0..self.population.len());
        if self.population[i].is_better_than(&self.population[j]) {
            i
        } else {
            j
        }
    }

    fn distant_top_k_indices(&self, rng: &mut impl rand::Rng, top_k: usize) -> (usize, usize)
    where
        P::Solution: Distance,
    {
        let n = self.population.len();
        let a = rng.random_range(0..n);
        let parent_a = &self.population[a];

        // Score every other index by distance to parent A.
        let mut scored: Vec<(usize, usize)> = (0..n)
            .filter(|&j| j != a)
            .map(|j| (j, parent_a.distance(&self.population[j])))
            .collect();

        // Partial sort: place the `k` largest distances at the front.
        let k = top_k.clamp(1, scored.len());
        scored.select_nth_unstable_by(k - 1, |x, y| y.1.cmp(&x.1));
        let candidates: Vec<usize> = scored[..k].iter().map(|(j, _)| *j).collect();
        let b = candidates[rng.random_range(0..candidates.len())];

        (a, b)
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
            .min_by(|(_, a), (_, b)| crate::trait_defs::rank_cmp(*a, *b))
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
                    .max_by(|(_, a), (_, b)| crate::trait_defs::rank_cmp(*a, *b))
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
    P::Solution: Distance,
    C: Crossover<P>,
{
    /// Clears the population so the next `run` starts fresh.
    fn clear(&mut self) {
        self.population.clear();
        self.best_idx = None;
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        if self.population.len() < self.population_size {
            self.initialize_population(state)?;
        }

        let (i_a, i_b) = self.select_parent_indices(&mut state.rng);
        let parent_a = self.population[i_a].clone();
        let parent_b = self.population[i_b].clone();

        let offspring =
            self.crossover
                .crossover(state.instance, &parent_a, &parent_b, &mut state.rng)?;

        let mutated = Self::improve_via_sub_run(state, offspring, self.mutation.as_mut())?;

        self.insert_into_population(mutated);

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
            _rng: &mut rand::rngs::SmallRng,
        ) -> Result<MaxCutSolution, OptError> {
            Ok(sol1.clone())
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
    fn genetic_algorithm_distant_top_k_runs_and_keeps_population_invariant() {
        let mc = MaxCut::new(Graph::from_edges([
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 2, 1.0),
            (1, 3, 1.0),
            (2, 3, 1.0),
            (3, 4, 1.0),
            (4, 5, 1.0),
        ]));
        let mut state = SearchState::new(&mc);
        let mut ga = GeneticAlgorithm::new(
            StopCondition::iterations(50),
            4,
            CloneFirstParent,
            Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
                StopCondition::failed_updates(1),
            )),
        )
        .with_parent_selection(ParentSelection::DistantTopK { top_k: 2 });

        ga.run(&mut state).unwrap();

        assert_eq!(ga.population.len(), 4);
        assert!(ga.best_idx.is_some());
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
