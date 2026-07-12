//! The generic config-driven heuristic factory.
//!
//! [`build_heuristic`] turns a [`HeuristicConfig`] into a `Box<dyn Heuristic<P>>`
//! for any [`ConfigurableProblem`]. The base-heuristic dispatch is written
//! exactly once ([`BaseBuilder`]); per-problem information comes from the
//! [`ConfigurableProblem`] impls in [`super::problems`].

use super::config::{HeuristicConfig, NeighborKind};
use super::problems::{BenchmarkProblem, BenchmarkSolution};
use crate::error::OptError;
use crate::heuristic::{
    GeneticAlgorithm, Heuristic, Iterated, LateAcceptanceHillClimbing, LocalSearch,
    ParentSelection, Restart, RewardShaping, RlSearch, Sequential, SimulatedAnnealing,
    StopCondition, TabuSearch,
};
use crate::search_state::{Crossover, Distance, MoveToNeighbor, ProblemTrait};
use crate::trait_defs::{EnabledTabu, Evaluate, Rankable};

// ---------------------------------------------------------------------------
// ConfigurableProblem: per-problem registration for the config-driven factory
// ---------------------------------------------------------------------------

/// Bound bundle every config-selectable neighborhood move must satisfy.
///
/// Blanket-implemented -- problems never implement it by hand. If a future
/// neighbor type cannot satisfy one of the bounds (e.g. `Evaluate`), split
/// the bundle instead of writing stub impls.
pub(crate) trait ConfigNeighbor<P: ProblemTrait>:
    MoveToNeighbor<P> + Rankable + Evaluate + EnabledTabu + Clone + 'static
{
}
impl<P: ProblemTrait, T> ConfigNeighbor<P> for T where
    T: MoveToNeighbor<P> + Rankable + Evaluate + EnabledTabu + Clone + 'static
{
}

/// Callback invoked with the concrete neighbor type chosen from config.
pub(crate) trait NeighborVisitor<P: ProblemTrait> {
    type Output;
    fn visit<N: ConfigNeighbor<P>>(self) -> Self::Output;
}

/// Per-problem registration point for the benchmark factory.
///
/// Adding a new problem to the benchmark requires exactly:
/// 1. a variant in [`ProblemKind`],
/// 2. an arm in [`with_problem`],
/// 3. an impl of this trait (plus [`BenchmarkProblem`] / [`BenchmarkSolution`]).
pub(crate) trait ConfigurableProblem: BenchmarkProblem + Send + Sync + 'static
where
    Self::Solution: BenchmarkSolution + Distance,
{
    /// Problem name used in error messages (matches the `ProblemKind` variant).
    const NAME: &'static str;
    /// Whether the problem is minimized (drives report statistics).
    const MINIMIZE: bool;
    /// The neighborhood kinds this problem supports.
    const VALID_NEIGHBORS: &'static [NeighborKind];

    /// The neighbor registry: maps a [`NeighborKind`] to the concrete move
    /// type by invoking `visitor` with it.
    fn with_neighbor<V: NeighborVisitor<Self>>(
        kind: &NeighborKind,
        visitor: V,
    ) -> Result<V::Output, OptError>;

    /// Problem-specific heuristics (BLS on MaxCut, LKH on TSP).
    fn build_special_heuristic(
        config: &HeuristicConfig,
        _cond: StopCondition,
    ) -> Result<Box<dyn Heuristic<Self>>, OptError> {
        Err(OptError::Config(format!(
            "heuristic '{}' is not supported for {}",
            config.kind_name(),
            Self::NAME
        )))
    }

    /// Builds the crossover operator selected by `crossover_kind`
    /// (with a problem-specific default when `None`).
    fn build_crossover(kind: Option<&str>) -> Result<Box<dyn Crossover<Self>>, OptError>;
}

/// Shared error for a `(problem, neighbor)` mismatch.
pub(crate) fn invalid_neighbor<P>(kind: &NeighborKind) -> OptError
where
    P: ConfigurableProblem,
    P::Solution: BenchmarkSolution + Distance,
{
    OptError::Config(format!(
        "Invalid neighbor {:?} for {} (valid: {:?})",
        kind,
        P::NAME,
        P::VALID_NEIGHBORS
    ))
}

// ---------------------------------------------------------------------------
// Generic heuristic builder
// ---------------------------------------------------------------------------

/// Builds the neighbor-parameterized base heuristics once the concrete
/// neighbor type is known. One arm per base kind: adding a new base heuristic
/// means adding a [`HeuristicConfig`] variant and following the compile
/// errors -- this match is written once, not per problem.
struct BaseBuilder<'c> {
    config: &'c HeuristicConfig,
    cond: StopCondition,
}

impl<'c, P> NeighborVisitor<P> for BaseBuilder<'c>
where
    P: ConfigurableProblem,
    P::Solution: BenchmarkSolution + Distance,
{
    type Output = Result<Box<dyn Heuristic<P>>, OptError>;

    fn visit<N: ConfigNeighbor<P>>(self) -> Self::Output {
        match self.config {
            HeuristicConfig::LocalSearch { .. } => Ok(Box::new(LocalSearch::<N>::new(self.cond))),
            HeuristicConfig::TabuSearch { tabu_tenure, .. } => {
                if tabu_tenure.0 > tabu_tenure.1 {
                    return Err(OptError::Config(format!(
                        "invalid tabu_tenure range [{}, {}]: min must be <= max",
                        tabu_tenure.0, tabu_tenure.1
                    )));
                }
                Ok(Box::new(TabuSearch::<N>::new(
                    self.cond,
                    *tabu_tenure,
                    None,
                )))
            }
            HeuristicConfig::SimulatedAnnealing {
                initial_temperature,
                cooling_rate,
                ..
            } => Ok(Box::new(SimulatedAnnealing::<N>::new(
                self.cond,
                *initial_temperature,
                *cooling_rate,
            ))),
            HeuristicConfig::LateAcceptanceHillClimbing { history_length, .. } => {
                if *history_length == 0 {
                    return Err(OptError::Config(
                        "'history_length' must be at least 1".to_string(),
                    ));
                }
                Ok(Box::new(LateAcceptanceHillClimbing::<N>::new(
                    self.cond,
                    *history_length,
                )))
            }
            HeuristicConfig::RlSearch { .. } => build_rl_search::<P, N>(self.config, self.cond),
            _ => unreachable!("non-neighbor kinds are dispatched before with_neighbor"),
        }
    }
}

fn build_rl_search<P, N>(
    config: &HeuristicConfig,
    cond: StopCondition,
) -> Result<Box<dyn Heuristic<P>>, OptError>
where
    P: ProblemTrait + 'static,
    N: MoveToNeighbor<P> + Evaluate + Clone + 'static,
{
    use crate::heuristic::reinforcement_learning::feature::NUM_FEATURES;

    let HeuristicConfig::RlSearch {
        learning_rate,
        discount,
        softmax_temperature,
        reward_shaping,
        policy_weights,
        max_candidates,
        ..
    } = config
    else {
        unreachable!("build_rl_search called with a non-RlSearch config");
    };

    let reward = match reward_shaping.as_deref().unwrap_or("Normalized") {
        "Raw" => RewardShaping::Raw,
        "Normalized" => RewardShaping::Normalized,
        "BestImprovement" => RewardShaping::BestImprovement,
        other => {
            return Err(OptError::Config(format!(
                "Unknown reward_shaping '{other}' (expected Raw, Normalized, or BestImprovement)"
            )));
        }
    };

    if discount.is_some() {
        tracing::warn!(
            "'discount' is deprecated and ignored: RlSearch uses single-step \
             REINFORCE, which has no discount factor"
        );
    }
    if max_candidates == &Some(0) {
        return Err(OptError::Config(
            "'max_candidates' must be at least 1 when set".to_string(),
        ));
    }

    let mut rl = RlSearch::<N>::new(
        cond,
        learning_rate.unwrap_or(0.01),
        softmax_temperature.unwrap_or(1.0),
        reward,
        *max_candidates,
    );
    if let Some(v) = policy_weights {
        if v.len() != NUM_FEATURES {
            return Err(OptError::Config(format!(
                "policy_weights must have exactly {} elements, got {}",
                NUM_FEATURES,
                v.len()
            )));
        }
        let mut arr = [0.0; NUM_FEATURES];
        arr.copy_from_slice(v);
        rl = rl.with_policy_weights(arr);
    }
    Ok(Box::new(rl))
}

/// Builds a `Box<dyn Heuristic<P>>` from a [`HeuristicConfig`].
///
/// Meta-heuristics (Sequential / Iterated / Restart / GeneticAlgorithm) are
/// handled generically here; problem-specific kinds go to
/// [`ConfigurableProblem::build_special_heuristic`]; neighbor-parameterized
/// base kinds go through [`ConfigurableProblem::with_neighbor`] +
/// [`BaseBuilder`].
pub(crate) fn build_heuristic<P>(
    config: &HeuristicConfig,
) -> Result<Box<dyn Heuristic<P>>, OptError>
where
    P: ConfigurableProblem,
    P::Solution: BenchmarkSolution + Distance,
{
    let cond = config.stop_condition().to_stop_condition();
    match config {
        HeuristicConfig::Sequential { steps, .. } => {
            let sub: Result<Vec<Box<dyn Heuristic<P>>>, OptError> =
                steps.iter().map(build_heuristic::<P>).collect();
            Ok(Box::new(Sequential::new(cond, sub?)))
        }
        HeuristicConfig::Iterated { steps, .. } => {
            if steps.len() != 2 {
                return Err(OptError::Config(format!(
                    "Iterated requires exactly 2 steps (search + perturbation), but got {}",
                    steps.len()
                )));
            }
            let search = build_heuristic::<P>(&steps[0])?;
            let perturbation = build_heuristic::<P>(&steps[1])?;
            Ok(Box::new(Iterated::new(cond, search, perturbation)))
        }
        HeuristicConfig::Restart {
            steps,
            restart_condition,
            ..
        } => {
            if steps.len() != 1 {
                return Err(OptError::Config(format!(
                    "Restart requires exactly 1 step (inner heuristic), but got {}",
                    steps.len()
                )));
            }
            let inner = build_heuristic::<P>(&steps[0])?;
            Ok(Box::new(Restart::new(
                cond,
                inner,
                restart_condition.to_stop_condition(),
            )))
        }
        HeuristicConfig::GeneticAlgorithm {
            population_size,
            steps,
            crossover_kind,
            parent_selection,
            parent_top_k,
            ..
        } => {
            if steps.is_empty() || steps.len() > 2 {
                return Err(OptError::Config(format!(
                    "GeneticAlgorithm requires 1 or 2 steps (steps[0] = mutation, optional steps[1] = init_improvement), but got {}",
                    steps.len()
                )));
            }
            if *population_size < 2 {
                return Err(OptError::Config(
                    "'population_size' must be at least 2".to_string(),
                ));
            }
            let mutation = build_heuristic::<P>(&steps[0])?;
            let init_improvement = match steps.get(1) {
                Some(c) => Some(build_heuristic::<P>(c)?),
                None => None,
            };
            let crossover = P::build_crossover(crossover_kind.as_deref())?;

            let parent_selection = match parent_selection.as_deref().unwrap_or("Tournament") {
                "Tournament" => ParentSelection::Tournament,
                "DistantTopK" => {
                    let top_k = parent_top_k.ok_or_else(|| {
                        OptError::Config(
                            "'parent_top_k' required for parent_selection = 'DistantTopK'"
                                .to_string(),
                        )
                    })?;
                    if top_k == 0 {
                        return Err(OptError::Config("'parent_top_k' must be >= 1".to_string()));
                    }
                    ParentSelection::DistantTopK { top_k }
                }
                other => {
                    return Err(OptError::Config(format!(
                        "Unknown parent_selection '{other}' (expected 'Tournament' or 'DistantTopK')"
                    )));
                }
            };

            let ga = GeneticAlgorithm::new_with_init(
                cond,
                *population_size,
                crossover,
                mutation,
                init_improvement,
            )
            .with_parent_selection(parent_selection);
            Ok(Box::new(ga))
        }
        HeuristicConfig::BreakoutLocalSearch { .. }
        | HeuristicConfig::PopulationAnnealingForMaxCut { .. }
        | HeuristicConfig::RlBreakoutLocalSearch { .. }
        | HeuristicConfig::LinKernighanHelsgaun { .. }
        | HeuristicConfig::AdaptiveLargeNeighborhoodSearch { .. } => {
            P::build_special_heuristic(config, cond)
        }
        HeuristicConfig::LocalSearch { neighbor, .. }
        | HeuristicConfig::TabuSearch { neighbor, .. }
        | HeuristicConfig::SimulatedAnnealing { neighbor, .. }
        | HeuristicConfig::LateAcceptanceHillClimbing { neighbor, .. }
        | HeuristicConfig::RlSearch { neighbor, .. } => {
            P::with_neighbor(neighbor, BaseBuilder { config, cond })?
        }
    }
}

#[cfg(test)]
mod factory_tests {
    use super::*;
    use crate::benchmark::config::{ProblemKind, StopConditionConfig};
    use crate::benchmark::problems::{ProblemVisitor, with_problem};

    /// Visitor that builds the heuristic and reports success/failure only.
    struct BuildCheck<'a> {
        config: &'a HeuristicConfig,
    }
    impl ProblemVisitor for BuildCheck<'_> {
        type Output = Result<(), OptError>;
        fn visit<P>(self) -> Result<(), OptError>
        where
            P: ConfigurableProblem,
            P::Solution: BenchmarkSolution + Distance,
        {
            build_heuristic::<P>(self.config).map(|_| ())
        }
    }

    fn try_build(kind: &ProblemKind, config: &HeuristicConfig) -> Result<(), OptError> {
        with_problem(kind, BuildCheck { config })
    }

    const ALL_PROBLEMS: &[ProblemKind] = &[
        ProblemKind::MaxCut,
        ProblemKind::Qubo,
        ProblemKind::Sat,
        ProblemKind::Tsp,
        ProblemKind::VertexCover,
        ProblemKind::JobShop,
        ProblemKind::Vrp,
    ];

    fn base_kinds_for(neighbor: NeighborKind) -> Vec<HeuristicConfig> {
        let sc = StopConditionConfig::default;
        vec![
            HeuristicConfig::LocalSearch {
                neighbor: neighbor.clone(),
                stop_condition: sc(),
            },
            HeuristicConfig::TabuSearch {
                neighbor: neighbor.clone(),
                tabu_tenure: (1, 5),
                stop_condition: sc(),
            },
            HeuristicConfig::SimulatedAnnealing {
                neighbor: neighbor.clone(),
                initial_temperature: 1.0,
                cooling_rate: 0.99,
                stop_condition: sc(),
            },
            HeuristicConfig::LateAcceptanceHillClimbing {
                neighbor: neighbor.clone(),
                history_length: 10,
                stop_condition: sc(),
            },
            HeuristicConfig::RlSearch {
                neighbor,
                learning_rate: None,
                discount: None,
                softmax_temperature: None,
                reward_shaping: None,
                policy_weights: None,
                max_candidates: None,
                stop_condition: sc(),
            },
        ]
    }

    /// Every (problem x base kind x valid neighbor) combination must build.
    #[test]
    fn all_base_kinds_build_for_all_valid_neighbors() {
        for problem in ALL_PROBLEMS {
            for neighbor in problem.valid_neighbors() {
                for config in base_kinds_for(neighbor.clone()) {
                    try_build(problem, &config).unwrap_or_else(|e| {
                        panic!(
                            "{problem:?} x {} x {neighbor:?} must build: {e}",
                            config.kind_name()
                        )
                    });
                }
            }
        }
    }

    #[test]
    fn invalid_neighbor_reports_problem_and_valid_set() {
        let config = HeuristicConfig::LocalSearch {
            neighbor: NeighborKind::TwoOpt,
            stop_condition: StopConditionConfig::default(),
        };
        let err = try_build(&ProblemKind::MaxCut, &config).expect_err("TwoOpt invalid for MaxCut");
        let msg = err.to_string();
        assert!(msg.contains("MaxCut"), "{msg}");
        assert!(msg.contains("TwoOpt"), "{msg}");
        assert!(msg.contains("Flip"), "{msg}");
    }

    #[test]
    fn problem_specific_kinds_are_rejected_elsewhere() {
        let bls = HeuristicConfig::BreakoutLocalSearch {
            tabu_tenure: (1, 5),
            t: 100,
            l0: 5,
            p0: 0.8,
            q: 0.5,
            plateau_prob: None,
            stop_condition: StopConditionConfig::default(),
        };
        try_build(&ProblemKind::MaxCut, &bls).expect("BLS builds for MaxCut");
        let err = try_build(&ProblemKind::Qubo, &bls).expect_err("BLS invalid for Qubo");
        assert!(err.to_string().contains("Qubo"), "{err}");

        let rl_bls = HeuristicConfig::RlBreakoutLocalSearch {
            tabu_tenure: (1, 5),
            t: 100,
            l0: 5,
            strength_bins: None,
            learning_rate: None,
            softmax_temperature: None,
            exploration: None,
            policy_weights: None,
            stop_condition: StopConditionConfig::default(),
        };
        try_build(&ProblemKind::MaxCut, &rl_bls).expect("RL-BLS builds for MaxCut");
        let err = try_build(&ProblemKind::Qubo, &rl_bls).expect_err("RL-BLS invalid for Qubo");
        assert!(err.to_string().contains("Qubo"), "{err}");

        let lkh = HeuristicConfig::LinKernighanHelsgaun {
            num_neighbors: None,
            max_depth: None,
            stop_condition: StopConditionConfig::default(),
        };
        try_build(&ProblemKind::Tsp, &lkh).expect("LKH builds for Tsp");
        let err = try_build(&ProblemKind::MaxCut, &lkh).expect_err("LKH invalid for MaxCut");
        assert!(err.to_string().contains("MaxCut"), "{err}");
    }

    #[test]
    fn genetic_algorithm_uses_problem_specific_crossover_defaults() {
        let ga = |crossover_kind: Option<String>| HeuristicConfig::GeneticAlgorithm {
            population_size: 4,
            steps: vec![HeuristicConfig::LocalSearch {
                neighbor: NeighborKind::TwoOpt,
                stop_condition: StopConditionConfig::default(),
            }],
            crossover_kind,
            parent_selection: None,
            parent_top_k: None,
            stop_condition: StopConditionConfig::default(),
        };
        // TSP defaults to Order crossover.
        try_build(&ProblemKind::Tsp, &ga(None)).expect("GA on Tsp with default crossover");
        let err = try_build(&ProblemKind::Tsp, &ga(Some("NoSuch".to_string())))
            .expect_err("unknown crossover_kind must fail");
        assert!(err.to_string().contains("NoSuch"), "{err}");
    }

    #[test]
    fn genetic_algorithm_builds_sub_problem_crossover_for_max_cut() {
        let ga = |crossover_kind: Option<String>| HeuristicConfig::GeneticAlgorithm {
            population_size: 4,
            steps: vec![HeuristicConfig::LocalSearch {
                neighbor: NeighborKind::Flip,
                stop_condition: StopConditionConfig::default(),
            }],
            crossover_kind,
            parent_selection: None,
            parent_top_k: None,
            stop_condition: StopConditionConfig::default(),
        };
        try_build(&ProblemKind::MaxCut, &ga(Some("SubProblem".to_string())))
            .expect("GA on MaxCut with SubProblem crossover");
        // Not registered for problems without a SubProblemExtractable-backed
        // entry (e.g. VertexCover keeps 'Uniform' only).
        let err = try_build(
            &ProblemKind::VertexCover,
            &ga(Some("SubProblem".to_string())),
        )
        .expect_err("SubProblem crossover is MaxCut-only for now");
        assert!(err.to_string().contains("SubProblem"), "{err}");
    }

    #[test]
    fn minimize_truth_table() {
        assert!(!ProblemKind::MaxCut.minimize());
        assert!(!ProblemKind::Sat.minimize());
        assert!(ProblemKind::Qubo.minimize());
        assert!(ProblemKind::Tsp.minimize());
        assert!(ProblemKind::VertexCover.minimize());
        assert!(ProblemKind::JobShop.minimize());
        assert!(ProblemKind::Vrp.minimize());
    }
}
