//! A problem ordered by `Rankable` alone, with no `Evaluate` anywhere, runs
//! every heuristic that only compares solutions.
//!
//! The order here is violations first and value second, the shape a
//! constrained problem takes when its constraints are not priced. Nothing
//! below reports a number, so a heuristic that still read one would not
//! compile against it.

use optopus::building_blocks::search::{TabuKey, TabuMemory};
use optopus::error::OptError;
use optopus::heuristic::{
    AdaptivePerturbation, BeamSearch, BreakoutLocalSearch, GeneticAlgorithm, Heuristic, Iterated,
    LocalSearch, ParentSelection, RandomWalk, StopCondition, TabuSearch,
};
use optopus::rand::Rng;
use optopus::rand::rngs::SmallRng;
use optopus::search_state::SearchState;
use optopus::trait_defs::{
    Crossover, Distance, EnabledTabu, MoveToNeighbor, ProblemTrait, Rankable,
};

/// Pick at most `LIMIT` items, as valuable as possible. Every item past the
/// limit is one violation.
struct Pick {
    values: Vec<u32>,
}

const LIMIT: usize = 4;

/// Fewer violations first, then more value.
#[derive(Clone, Debug, Rankable)]
#[rank(min = violations(), max = value)]
struct Picked {
    x: Vec<bool>,
    count: usize,
    value: u32,
}

impl Picked {
    fn violations(&self) -> usize {
        self.count.saturating_sub(LIMIT)
    }
}

impl Distance for Picked {
    fn distance(&self, other: &Self) -> usize {
        self.x.iter().zip(&other.x).filter(|(a, b)| a != b).count()
    }
}

impl ProblemTrait for Pick {
    type Solution = Picked;

    fn new_solution(&self, rng: &mut impl Rng) -> Picked {
        let x: Vec<bool> = (0..self.values.len())
            .map(|_| rng.random_bool(0.5))
            .collect();
        self.picked(x)
    }
}

impl Pick {
    fn new(n: usize) -> Self {
        Self {
            values: (0..n as u32).map(|i| (i * 7) % 11 + 1).collect(),
        }
    }

    fn picked(&self, x: Vec<bool>) -> Picked {
        let count = x.iter().filter(|&&b| b).count();
        let value = x
            .iter()
            .zip(&self.values)
            .filter(|(b, _)| **b)
            .map(|(_, v)| v)
            .sum();
        Picked { x, count, value }
    }

    /// The most valuable feasible pick, to check a search against.
    fn optimum(&self) -> u32 {
        let mut v = self.values.clone();
        v.sort_unstable_by(|a, b| b.cmp(a));
        v.iter().take(LIMIT).sum()
    }
}

/// Flip one item, carrying the solution it leads to so that two flips of the
/// same solution compare by where they land.
#[derive(Clone)]
struct Flip {
    i: usize,
    after: Picked,
}

impl Rankable for Flip {
    fn is_better_than(&self, other: &Self) -> bool {
        self.after.is_better_than(&other.after)
    }
}

impl MoveToNeighbor<Pick> for Flip {
    fn apply_to_solution(&self, _prob: &Pick, sol: &mut Picked) -> Result<(), OptError> {
        sol.clone_from(&self.after);
        Ok(())
    }

    fn iter(prob: &Pick, sol: &Picked) -> impl Iterator<Item = Self> + Send {
        let sol = sol.clone();
        let values = prob.values.clone();
        (0..sol.x.len()).map(move |i| {
            let mut after = sol.clone();
            after.x[i] = !after.x[i];
            if after.x[i] {
                after.count += 1;
                after.value += values[i];
            } else {
                after.count -= 1;
                after.value -= values[i];
            }
            Flip { i, after }
        })
    }

    fn move_to_be_better_than(&self, _prob: &Pick, _src: &Picked, other: &Picked) -> bool {
        self.after.is_better_than(other)
    }

    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }
}

impl EnabledTabu for Flip {
    fn is_move_enabled(&self, tabu: &TabuMemory, iter: u64) -> bool {
        tabu.is_enabled(TabuKey::Var(self.i), iter)
    }

    fn add_to_tabu_map(&self, tabu: &mut TabuMemory, iter: u64, rng: &mut SmallRng) {
        tabu.forbid(TabuKey::Var(self.i), iter, rng);
    }
}

/// Uniform crossover, each item from either parent.
struct Uniform;

impl Crossover<Pick> for Uniform {
    fn crossover(
        &mut self,
        prob: &Pick,
        a: &Picked,
        b: &Picked,
        rng: &mut SmallRng,
    ) -> Result<Picked, OptError> {
        let x =
            a.x.iter()
                .zip(&b.x)
                .map(|(&p, &q)| if rng.random_bool(0.5) { p } else { q })
                .collect();
        Ok(prob.picked(x))
    }
}

fn assert_optimal(name: &str, prob: &Pick, state: &SearchState<Pick>) {
    let best = &state.best_solution;
    assert_eq!(best.violations(), 0, "{name} ended infeasible");
    assert_eq!(best.value, prob.optimum(), "{name} missed the optimum");
}

fn descent() -> LocalSearch<Flip> {
    LocalSearch::new(StopCondition::iterations(1_000))
}

#[test]
fn comparison_only_heuristics_run_on_an_order_without_a_number() {
    let prob = Pick::new(12);

    let mut state = SearchState::new_with_seed(&prob, 1);
    TabuSearch::<Flip>::new(StopCondition::iterations(500), (2, 4))
        .run(&mut state)
        .unwrap();
    assert_optimal("TabuSearch", &prob, &state);

    let mut state = SearchState::new_with_seed(&prob, 2);
    BeamSearch::<Pick, Flip>::new(StopCondition::iterations(50), 8)
        .run(&mut state)
        .unwrap();
    assert_optimal("BeamSearch", &prob, &state);

    let mut state = SearchState::new_with_seed(&prob, 3);
    Iterated::new(
        StopCondition::iterations(2_000),
        Box::new(descent()),
        Box::new(RandomWalk::<Flip>::new(StopCondition::iterations(3))),
    )
    .run(&mut state)
    .unwrap();
    assert_optimal("Iterated", &prob, &state);

    let mut state = SearchState::new_with_seed(&prob, 4);
    GeneticAlgorithm::new(
        StopCondition::iterations(2_000),
        8,
        Uniform,
        Box::new(descent()),
        ParentSelection::BiasedFitness {
            n_elite: 2,
            n_closest: 3,
        },
    )
    .run(&mut state)
    .unwrap();
    assert_optimal("GeneticAlgorithm", &prob, &state);

    let mut state = SearchState::new_with_seed(&prob, 5);
    BreakoutLocalSearch::new(
        StopCondition::iterations(2_000),
        (2, 4),
        Box::new(descent()),
        AdaptivePerturbation::new(
            50,
            2,
            0.5,
            Box::new(RandomWalk::<Flip>::new(StopCondition::iterations(1))),
            vec![(
                Box::new(TabuSearch::<Flip>::new(
                    StopCondition::iterations(1),
                    (2, 4),
                )) as Box<dyn Heuristic<Pick>>,
                1.0,
            )],
        ),
    )
    .run(&mut state)
    .unwrap();
    assert_optimal("BreakoutLocalSearch", &prob, &state);
}
