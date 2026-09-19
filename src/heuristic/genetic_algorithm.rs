use super::{Heuristic, StopCondition};
use crate::common::{BiasedFitnessPopulation, binary_tournament};
use crate::error::OptError;
use crate::search_state::{
    Crossover, Distance, Evaluate, ProblemTrait, Rankable, SearchState, SearchStateCloneType,
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
    /// Vidal's *biased fitness*: rank members by a blend of cost rank and
    /// diversity rank, then run a binary tournament on that rank.
    ///
    /// The other two strategies rank on cost alone, which collapses the
    /// population onto one basin within a few hundred generations. Every
    /// survivor descends from the same early winner, and crossover between
    /// near-identical parents produces near-identical children. Blending in a
    /// diversity rank is what lets a genetic search keep going for millions of
    /// generations. It is the mechanism behind HGS, which uses the same
    /// [`BiasedFitnessPopulation`].
    ///
    /// This strategy also changes survivor selection, not just parent
    /// selection: members are trimmed by the same blended rank, clones first.
    /// The two halves cannot be separated. Ranking parents by diversity while
    /// evicting purely on cost would let the population converge anyway, one
    /// eviction at a time.
    ///
    /// `n_elite` and `n_closest` are Vidal's `nbElite` and `nbClose`, whose
    /// published values are `4` and `5`. Neither is defaulted. Note that a
    /// population of `n_elite` or fewer ranks on cost alone, the diversity half
    /// being weighted by `1 - n_elite / N`.
    BiasedFitness { n_elite: usize, n_closest: usize },
}

/// The population, in whichever form the selection strategy needs.
///
/// Kept as a choice rather than always ranking, because the ranking is not
/// free: it measures O(N) distances per insertion and re-ranks in O(N² log N).
/// The cost-only strategies read no fitness, so they would pay that for
/// nothing.
enum Members<S> {
    /// Insertion order, worst-replaced. What `Tournament` and `DistantTopK`
    /// use.
    Plain(Vec<S>),
    /// Ranked by biased fitness, pairwise distances cached.
    Ranked(BiasedFitnessPopulation<S>),
}

impl<S: Distance + Evaluate + 'static> Members<S> {
    /// Builds the representation the strategy needs, and with it the cost the
    /// ranked one is ordered by. Separate from the rest of the impl because
    /// only this needs to know what a member is worth.
    fn for_strategy(strategy: ParentSelection) -> Self {
        match strategy {
            ParentSelection::BiasedFitness { n_elite, n_closest } => {
                // The one place the two quantities the ranking reads are
                // stated. A solution's objective with its direction applied,
                // and the distance its own impl defines.
                Self::Ranked(BiasedFitnessPopulation::new(
                    n_elite,
                    n_closest,
                    Box::new(|s: &S| s.evaluate().minimized()),
                    Box::new(|a: &S, b: &S| a.distance(b) as f64),
                ))
            }
            _ => Self::Plain(Vec::new()),
        }
    }
}

impl<S> Members<S> {
    fn as_slice(&self) -> &[S] {
        match self {
            Self::Plain(v) => v,
            Self::Ranked(p) => p.members(),
        }
    }

    fn len(&self) -> usize {
        self.as_slice().len()
    }

    fn clear(&mut self) {
        match self {
            Self::Plain(v) => v.clear(),
            Self::Ranked(p) => p.clear(),
        }
    }

    fn iter(&self) -> std::slice::Iter<'_, S> {
        self.as_slice().iter()
    }
}

impl<S> std::ops::Index<usize> for Members<S> {
    type Output = S;

    fn index(&self, i: usize) -> &S {
        &self.as_slice()[i]
    }
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
/// With [`with_init_improvement`](Self::with_init_improvement), each random
/// initial individual is also passed through that heuristic via the same
/// sub-run pattern. This reproduces the Galinier-Hao Hybrid Evolutionary
/// Algorithm (HEA) for graph coloring when paired with a
/// [`crate::heuristic::TabuSearch`] mutation operator.
///
/// The global best solution is tracked in `SearchState::best_solution`.
///
/// # References
///
/// - Holland, J. H. *Adaptation in Natural and Artificial Systems*. University of Michigan Press, 1975.
/// - Goldberg, D. E. *Genetic Algorithms in Search, Optimization, and Machine Learning*.
///   Addison-Wesley, 1989.
/// - Galinier, P. and Hao, J.-K. "Hybrid Evolutionary Algorithms for Graph Coloring."
///   Journal of Combinatorial Optimization, 3(4), 379-397, 1999.
///
/// # Type parameters
///
/// - `P`, the problem type; must implement [`ProblemTrait`].
/// - `C`, the crossover operator; must implement [`Crossover<P>`].
///
/// # Example
///
/// ```rust,ignore
/// use optopus::heuristic::{
///     GeneticAlgorithm, LocalSearch, ParentSelection, StopCondition, SubProblemBasedCrossover,
/// };
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
///     ParentSelection::Tournament,
/// );
/// ga.run(&mut state).unwrap();
/// ```
pub struct GeneticAlgorithm<P: ProblemTrait, C> {
    pub stop_condition: StopCondition,
    pub population_size: usize,
    /// Crossover operator stored as a value because `Crossover::crossover` takes `&mut self`.
    pub crossover: C,
    /// Mutation operator, any [`Heuristic<P>`] works (local search, SA, random walk, …).
    pub mutation: Box<dyn Heuristic<P>>,
    /// Optional per-individual local improvement applied to each random seed
    /// during population initialization. See
    /// [`with_init_improvement`](Self::with_init_improvement).
    pub init_improvement: Option<Box<dyn Heuristic<P>>>,
    /// Strategy for sampling the two parents each iteration. Fixed at
    /// construction: it also decides how `population` is stored, so the two
    /// cannot be changed apart.
    parent_selection: ParentSelection,
    population: Members<P::Solution>,
}

impl<P, C> GeneticAlgorithm<P, C>
where
    P: ProblemTrait,
    // `'static` because the ranked representation owns a boxed cost closure
    // whose signature names the solution type. Every solution here is owned
    // data, so it costs nothing in practice.
    P::Solution: Distance + Evaluate + 'static,
{
    /// # Panics
    ///
    /// Panics if `population_size < 2`.
    pub fn new(
        stop_condition: StopCondition,
        population_size: usize,
        crossover: C,
        mutation: Box<dyn Heuristic<P>>,
        parent_selection: ParentSelection,
    ) -> Self {
        assert!(population_size >= 2, "population_size must be at least 2");
        Self {
            stop_condition,
            population_size,
            crossover,
            mutation,
            init_improvement: None,
            parent_selection,
            population: Members::for_strategy(parent_selection),
        }
    }

    /// Builder-style: runs `op` on every member of the initial random
    /// population, which is the Galinier-Hao HEA configuration.
    ///
    /// The population is not built until the first `run_once`, so this may be
    /// set at any point before then.
    #[must_use]
    pub fn with_init_improvement(mut self, op: Box<dyn Heuristic<P>>) -> Self {
        self.init_improvement = Some(op);
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
            self.admit(member);
        }
        Ok(())
    }

    /// Returns indices of the two parents according to `self.parent_selection`.
    /// May return the same index twice, which is rare and allowed, since a
    /// tournament draws its candidates with replacement.
    fn select_parent_indices(&self, rng: &mut impl rand::Rng) -> (usize, usize)
    where
        P::Solution: Distance,
    {
        match self.parent_selection {
            ParentSelection::Tournament => self.tournament_indices(rng),
            ParentSelection::DistantTopK { top_k } => self.distant_top_k_indices(rng, top_k),
            ParentSelection::BiasedFitness { .. } => self.biased_fitness_indices(rng),
        }
    }

    /// Two independent binary tournaments on biased fitness (lower wins).
    ///
    /// Falls back to the cost-only tournament when the population is not the
    /// ranked kind. That cannot happen through the public API, the
    /// representation being chosen from the strategy, but keeping this total
    /// beats panicking on a state the type system does not forbid.
    fn biased_fitness_indices(&self, rng: &mut impl rand::Rng) -> (usize, usize) {
        let Members::Ranked(pop) = &self.population else {
            return self.tournament_indices(rng);
        };
        let candidates: Vec<(usize, f64)> = (0..pop.len()).map(|i| (i, pop.fitness(i))).collect();
        let a = binary_tournament(&candidates, rng);
        let b = binary_tournament(&candidates, rng);
        match (a, b) {
            (Some(a), Some(b)) => (a, b),
            _ => (0, 0),
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

    /// Adds `member` to whichever representation the population holds, without
    /// deciding whether it deserves a place. That is
    /// [`insert_into_population`](Self::insert_into_population)'s job, and this
    /// is what both it and the initial seeding go through.
    fn admit(&mut self, member: P::Solution) {
        match &mut self.population {
            Members::Plain(v) => v.push(member),
            Members::Ranked(p) => p.push(member),
        }
    }

    /// Index of the best member by the problem's own ranking.
    fn best_member_idx(&self) -> Option<usize> {
        self.population
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| crate::trait_defs::rank_cmp(*a, *b))
            .map(|(i, _)| i)
    }

    /// Inserts `offspring` into the population.
    ///
    /// Which member leaves depends on the strategy, and it has to: ranking
    /// parents by diversity while evicting purely on cost would let the
    /// population converge anyway, one eviction at a time.
    ///
    /// - `Plain`: replace the worst member, if `offspring` beats it.
    /// - `Ranked`: admit unconditionally, then trim back to capacity by biased
    ///   fitness, which evicts clones first, so a newcomer that duplicates an
    ///   incumbent simply displaces it.
    fn insert_into_population(&mut self, offspring: P::Solution) {
        if let Members::Plain(members) = &mut self.population
            && members.len() >= self.population_size
        {
            let worst_idx = members
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| crate::trait_defs::rank_cmp(*a, *b))
                .map(|(i, _)| i)
                .expect("a full population is non-empty");
            if offspring.is_better_than(&members[worst_idx]) {
                members[worst_idx] = offspring;
            }
            return;
        }
        self.admit(offspring);
        if let Members::Ranked(p) = &mut self.population {
            p.trim_to(self.population_size);
        }
    }
}

impl<P, C> Heuristic<P> for GeneticAlgorithm<P, C>
where
    P: ProblemTrait,
    P::Solution: Distance + Evaluate + 'static,
    C: Crossover<P>,
{
    /// Clears the population so the next `run` starts fresh.
    fn clear(&mut self) {
        self.population.clear();
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        if self.population.len() < self.population_size {
            self.initialize_population(state)?;
        }

        let (i_a, i_b) = self.select_parent_indices(&mut state.rng);
        let offspring = self.crossover.crossover(
            state.instance,
            &self.population[i_a],
            &self.population[i_b],
            &mut state.rng,
        )?;

        let mutated = Self::improve_via_sub_run(state, offspring, self.mutation.as_mut())?;

        self.insert_into_population(mutated);

        let best = self
            .best_member_idx()
            .expect("the population was filled above");
        state.solution = self.population[best].clone();
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
    use rand::SeedableRng;

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

    fn triangle_plus_tail() -> MaxCut {
        MaxCut::new(Graph::from_edges([
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 2, 1.0),
            (1, 3, 1.0),
            (2, 3, 1.0),
            (3, 4, 1.0),
            (4, 5, 1.0),
        ]))
    }

    /// Every selection strategy, and the optional initial-improvement pass,
    /// has to carry a generation through without leaving the population the
    /// wrong size. This is a smoke test and says so: the interesting claim
    /// about `DistantTopK` is the one below it.
    #[test]
    fn every_parent_selection_completes_a_generation() {
        for (name, selection, init_improvement) in [
            ("tournament", ParentSelection::Tournament, false),
            ("tournament + init", ParentSelection::Tournament, true),
            (
                "distant top k",
                ParentSelection::DistantTopK { top_k: 2 },
                false,
            ),
        ] {
            let mc = triangle_plus_tail();
            let mut state = SearchState::new_with_seed(&mc, 7);
            let mut ga = GeneticAlgorithm::new(
                StopCondition::iterations(20),
                4,
                CloneFirstParent,
                Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
                    StopCondition::failed_updates(1),
                )),
                selection,
            );
            if init_improvement {
                ga = ga.with_init_improvement(Box::new(TabuSearch::<MaxCutFlipNeighbor>::new(
                    StopCondition::failed_updates(1),
                    (1, 5),
                )));
            }

            ga.run(&mut state).unwrap();

            assert_eq!(ga.population.len(), 4, "{name}");
            assert_eq!(
                state.best_solution.objective,
                mc.calculate_cut_size(&state.best_solution.x),
                "{name}: the reported best is not the cut it carries"
            );
        }
    }

    /// With `top_k = 1` the second parent is determined: it is the member
    /// furthest from the first. A selection that ignored the distances, or
    /// sorted them the wrong way round, would still fill the population and
    /// still report a best, which is all the smoke test above can see.
    #[test]
    fn distant_top_k_of_one_picks_the_furthest_member() {
        let mc = triangle_plus_tail();
        // Hamming distances chosen so each member has a unique furthest peer.
        let assignments = [
            vec![false, false, false, false, false, false],
            vec![true, false, false, false, false, false],
            vec![true, true, false, false, false, false],
            vec![true, true, true, true, true, true],
        ];

        let mut ga = GeneticAlgorithm::new(
            StopCondition::iterations(1),
            4,
            CloneFirstParent,
            Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
                StopCondition::failed_updates(1),
            )),
            ParentSelection::DistantTopK { top_k: 1 },
        );
        for x in &assignments {
            ga.admit(MaxCutSolution::new_from_assignment(&mc, x.clone()));
        }

        let mut rng = rand::rngs::SmallRng::seed_from_u64(3);
        let mut seen_first_parents = std::collections::HashSet::new();
        for _ in 0..50 {
            let (a, b) = ga.distant_top_k_indices(&mut rng, 1);
            seen_first_parents.insert(a);
            let furthest = (0..4)
                .filter(|&j| j != a)
                .max_by_key(|&j| ga.population[a].distance(&ga.population[j]))
                .unwrap();
            assert_eq!(b, furthest, "first parent {a} paired with {b}");
        }
        assert_eq!(
            seen_first_parents.len(),
            4,
            "every member should have been drawn as the first parent"
        );
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
            ParentSelection::Tournament,
        );
    }
}
