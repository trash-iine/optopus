//! TOML-facing configuration types and config validation.

use serde::{Deserialize, Serialize};

use crate::error::OptError;
use crate::heuristic::StopCondition;

// ---------------------------------------------------------------------------
// Config types (Deserialize + Serialize) — used as input from a TOML file
// ---------------------------------------------------------------------------

/// Problem type discriminant used in config files.
///
/// `FormulaProblem` is intentionally absent: it is library-only, constructed
/// in code from an [`Expr`](crate::problem::Expr) AST, and has no instance
/// file format for the benchmark runner to load.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ProblemKind {
    MaxCut,
    Qubo,
    Sat,
    Tsp,
    VertexCover,
    JobShop,
    Vrp,
}

/// Stop condition as expressed in a config file (duration in seconds instead of `Duration`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StopConditionConfig {
    pub max_iteration: Option<u64>,
    pub max_duration_secs: Option<f64>,
    pub max_failed_update: Option<u64>,
}

impl StopConditionConfig {
    pub fn to_stop_condition(&self) -> StopCondition {
        StopCondition::new(
            self.max_iteration,
            self.max_duration_secs
                .map(std::time::Duration::from_secs_f64),
            self.max_failed_update,
        )
    }
}

/// Neighborhood move type used in config files.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum NeighborKind {
    Flip,
    Swap,
    TwoOpt,
    Relocate,
}

/// Heuristic configuration as expressed in a config file.
///
/// Internally tagged by `kind`, so a TOML entry looks like:
///
/// ```toml
/// [[heuristics]]
/// kind = "TabuSearch"
/// neighbor = "Flip"
/// tabu_tenure = [5, 10]
///
/// [heuristics.stop_condition]
/// max_iteration = 100000
/// ```
///
/// Missing required fields and unknown `kind` values are rejected at parse
/// time. The problem type is inferred from the instance being benchmarked and
/// does not need to be specified here.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum HeuristicConfig {
    LocalSearch {
        neighbor: NeighborKind,
        #[serde(default)]
        stop_condition: StopConditionConfig,
    },
    TabuSearch {
        neighbor: NeighborKind,
        /// Tabu tenure range `(min, max)` in iterations.
        tabu_tenure: (u64, u64),
        #[serde(default)]
        stop_condition: StopConditionConfig,
    },
    SimulatedAnnealing {
        neighbor: NeighborKind,
        initial_temperature: f64,
        cooling_rate: f64,
        #[serde(default)]
        stop_condition: StopConditionConfig,
    },
    LateAcceptanceHillClimbing {
        neighbor: NeighborKind,
        history_length: usize,
        #[serde(default)]
        stop_condition: StopConditionConfig,
    },
    /// REINFORCE policy-gradient move selection.
    RlSearch {
        neighbor: NeighborKind,
        /// Learning rate (0.0 = evaluation mode). Default: 0.01.
        #[serde(skip_serializing_if = "Option::is_none")]
        learning_rate: Option<f64>,
        /// Deprecated and ignored (single-step REINFORCE has no discount
        /// factor); accepted for config compatibility, warns at build time.
        #[serde(skip_serializing_if = "Option::is_none")]
        discount: Option<f64>,
        /// Softmax temperature. Default: 1.0.
        #[serde(skip_serializing_if = "Option::is_none")]
        softmax_temperature: Option<f64>,
        /// Reward shaping strategy: "Raw", "Normalized" (default), "BestImprovement".
        #[serde(skip_serializing_if = "Option::is_none")]
        reward_shaping: Option<String>,
        /// Pre-trained policy weights (`NUM_FEATURES` = 21 elements).
        #[serde(skip_serializing_if = "Option::is_none")]
        policy_weights: Option<Vec<f64>>,
        /// Max candidate moves sampled (before evaluation) per step.
        #[serde(skip_serializing_if = "Option::is_none")]
        max_candidates: Option<usize>,
        #[serde(default)]
        stop_condition: StopConditionConfig,
    },
    /// Breakout Local Search (MaxCut only).
    BreakoutLocalSearch {
        tabu_tenure: (u64, u64),
        t: u64,
        l0: u64,
        p0: f64,
        q: f64,
        /// Probability that a weak perturbation flips a connected cluster of
        /// zero-gain vertices instead (plateau traversal). Default: 0.0
        /// (original Benlic & Hao behavior).
        #[serde(skip_serializing_if = "Option::is_none")]
        plateau_prob: Option<f64>,
        #[serde(default)]
        stop_condition: StopConditionConfig,
    },
    /// Population Annealing Monte Carlo with non-local cluster moves
    /// (MaxCut only).
    PopulationAnnealingForMaxCut {
        /// Number of replicas `R` (>= 2).
        population_size: usize,
        /// Starting inverse temperature. Default: 0.1.
        #[serde(skip_serializing_if = "Option::is_none")]
        initial_beta: Option<f64>,
        /// Inverse-temperature increment per step. Default: 0.02.
        #[serde(skip_serializing_if = "Option::is_none")]
        delta_beta: Option<f64>,
        /// Metropolis sweeps per replica per step. Default: 50.
        #[serde(skip_serializing_if = "Option::is_none")]
        sweeps_per_step: Option<usize>,
        /// Reset `β` to `initial_beta` every this many steps. Default: 400
        /// (set to 0 to disable resets).
        #[serde(skip_serializing_if = "Option::is_none")]
        reset_period: Option<usize>,
        /// Enable the non-local cluster (iso-site) move. Default: true.
        #[serde(skip_serializing_if = "Option::is_none")]
        cluster_moves: Option<bool>,
        #[serde(default)]
        stop_condition: StopConditionConfig,
    },
    /// Breakout Local Search with a learned (contextual-bandit) perturbation
    /// policy (MaxCut only).
    RlBreakoutLocalSearch {
        tabu_tenure: (u64, u64),
        /// Omega normalization period for the stagnation features.
        t: u64,
        /// Base perturbation length; actions scale it by a strength bin.
        l0: u64,
        /// Strength multipliers of `l0`. Default: `[1.0, 2.0, 4.0]`.
        #[serde(skip_serializing_if = "Option::is_none")]
        strength_bins: Option<Vec<f64>>,
        /// Bandit step size (0.0 = frozen-policy evaluation). Default: 0.1.
        #[serde(skip_serializing_if = "Option::is_none")]
        learning_rate: Option<f64>,
        /// Bandit softmax temperature. Default: 1.0.
        #[serde(skip_serializing_if = "Option::is_none")]
        softmax_temperature: Option<f64>,
        /// ε-uniform exploration floor. Default: 0.05.
        #[serde(skip_serializing_if = "Option::is_none")]
        exploration: Option<f64>,
        /// Pre-trained bandit weights, row-major
        /// `(5 × strength_bins.len()) × NUM_CONTEXT_FEATURES` elements.
        /// Weights saved before the plateau operators were added (3 types ×
        /// 7 features) are rejected at parse time with a size error.
        #[serde(skip_serializing_if = "Option::is_none")]
        policy_weights: Option<Vec<f64>>,
        #[serde(default)]
        stop_condition: StopConditionConfig,
    },
    /// Lin-Kernighan-Helsgaun (TSP only).
    LinKernighanHelsgaun {
        /// Candidate neighbors per city. Default: 5.
        #[serde(skip_serializing_if = "Option::is_none")]
        num_neighbors: Option<usize>,
        /// Max LK move depth. Default: 5.
        #[serde(skip_serializing_if = "Option::is_none")]
        max_depth: Option<usize>,
        #[serde(default)]
        stop_condition: StopConditionConfig,
    },
    /// Adaptive Large Neighborhood Search (VRP only).
    AdaptiveLargeNeighborhoodSearch {
        /// Fraction of customers ruined each iteration. Default: 0.15.
        #[serde(skip_serializing_if = "Option::is_none")]
        removal_fraction: Option<f64>,
        /// Geometric cooling factor applied per iteration. Default: 0.9995.
        #[serde(skip_serializing_if = "Option::is_none")]
        cooling_rate: Option<f64>,
        #[serde(default)]
        stop_condition: StopConditionConfig,
    },
    /// Repeats its `steps` cycle until `stop_condition` is met.
    Sequential {
        steps: Vec<HeuristicConfig>,
        #[serde(default)]
        stop_condition: StopConditionConfig,
    },
    /// ILS: `steps[0]` = search phase, `steps[1]` = perturbation phase.
    Iterated {
        steps: Vec<HeuristicConfig>,
        #[serde(default)]
        stop_condition: StopConditionConfig,
    },
    /// Runs `steps[0]`, resetting to a fresh random solution whenever
    /// `restart_condition` is met.
    Restart {
        steps: Vec<HeuristicConfig>,
        restart_condition: StopConditionConfig,
        #[serde(default)]
        stop_condition: StopConditionConfig,
    },
    /// `steps[0]` = mutation, optional `steps[1]` = init_improvement (HEA pattern).
    GeneticAlgorithm {
        /// Must be >= 2.
        population_size: usize,
        steps: Vec<HeuristicConfig>,
        /// Crossover operator. Defaults: "Uniform" ("Order" for TSP, "Ppx" for JobShop).
        /// MaxCut also accepts "SubProblem" (memetic recombination: solves the
        /// sub-MaxCut of disagreeing variables with an internal bounded BLS).
        #[serde(skip_serializing_if = "Option::is_none")]
        crossover_kind: Option<String>,
        /// "Tournament" (default) or "DistantTopK".
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_selection: Option<String>,
        /// `top_k` for `parent_selection = "DistantTopK"` (must be >= 1).
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_top_k: Option<usize>,
        #[serde(default)]
        stop_condition: StopConditionConfig,
    },
}

impl HeuristicConfig {
    /// The `kind` tag as it appears in config files.
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::LocalSearch { .. } => "LocalSearch",
            Self::TabuSearch { .. } => "TabuSearch",
            Self::SimulatedAnnealing { .. } => "SimulatedAnnealing",
            Self::LateAcceptanceHillClimbing { .. } => "LateAcceptanceHillClimbing",
            Self::RlSearch { .. } => "RlSearch",
            Self::BreakoutLocalSearch { .. } => "BreakoutLocalSearch",
            Self::PopulationAnnealingForMaxCut { .. } => "PopulationAnnealingForMaxCut",
            Self::RlBreakoutLocalSearch { .. } => "RlBreakoutLocalSearch",
            Self::LinKernighanHelsgaun { .. } => "LinKernighanHelsgaun",
            Self::AdaptiveLargeNeighborhoodSearch { .. } => "AdaptiveLargeNeighborhoodSearch",
            Self::Sequential { .. } => "Sequential",
            Self::Iterated { .. } => "Iterated",
            Self::Restart { .. } => "Restart",
            Self::GeneticAlgorithm { .. } => "GeneticAlgorithm",
        }
    }

    /// The neighborhood move this config selects, when the kind uses one.
    pub fn neighbor(&self) -> Option<&NeighborKind> {
        match self {
            Self::LocalSearch { neighbor, .. }
            | Self::TabuSearch { neighbor, .. }
            | Self::SimulatedAnnealing { neighbor, .. }
            | Self::LateAcceptanceHillClimbing { neighbor, .. }
            | Self::RlSearch { neighbor, .. } => Some(neighbor),
            _ => None,
        }
    }

    /// Nested sub-heuristics (empty for non-composite kinds).
    pub fn steps(&self) -> &[HeuristicConfig] {
        match self {
            Self::Sequential { steps, .. }
            | Self::Iterated { steps, .. }
            | Self::Restart { steps, .. }
            | Self::GeneticAlgorithm { steps, .. } => steps,
            _ => &[],
        }
    }

    /// The outer stop condition (every kind carries one).
    pub fn stop_condition(&self) -> &StopConditionConfig {
        match self {
            Self::LocalSearch { stop_condition, .. }
            | Self::TabuSearch { stop_condition, .. }
            | Self::SimulatedAnnealing { stop_condition, .. }
            | Self::LateAcceptanceHillClimbing { stop_condition, .. }
            | Self::RlSearch { stop_condition, .. }
            | Self::BreakoutLocalSearch { stop_condition, .. }
            | Self::PopulationAnnealingForMaxCut { stop_condition, .. }
            | Self::RlBreakoutLocalSearch { stop_condition, .. }
            | Self::LinKernighanHelsgaun { stop_condition, .. }
            | Self::AdaptiveLargeNeighborhoodSearch { stop_condition, .. }
            | Self::Sequential { stop_condition, .. }
            | Self::Iterated { stop_condition, .. }
            | Self::Restart { stop_condition, .. }
            | Self::GeneticAlgorithm { stop_condition, .. } => stop_condition,
        }
    }
}

/// A single instance entry in the config file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstanceConfig {
    /// Path to the instance file; glob patterns (e.g. `"data/instances/max_cut/G[1-9]*"`) are supported.
    pub path: String,
    pub problem: ProblemKind,
}

/// Top-level benchmark configuration read from a TOML file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub num_runs: usize,
    pub instances: Vec<InstanceConfig>,
    pub heuristics: Vec<HeuristicConfig>,
    /// Optional master seed. When set, each `(instance, heuristic, run_index)`
    /// triple is run with a deterministic per-run seed derived from this master,
    /// so re-running the same config produces bit-identical TOML output.
    /// When `None`, every run is entropy-seeded (the original behavior).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// Recursively validates that every `neighbor` field in `h` (including in
/// nested `steps` from Sequential / Iterated / Restart / GeneticAlgorithm)
/// is supported by `problem`.
fn validate_heuristic_neighbors(
    h: &HeuristicConfig,
    problem: &ProblemKind,
    instance_path: &str,
) -> Result<(), OptError> {
    if let Some(n) = h.neighbor() {
        let valid = problem.valid_neighbors();
        if !valid.contains(n) {
            return Err(OptError::Config(format!(
                "instance '{}' ({:?}) does not support neighbor {:?} for heuristic '{}'. \
                 Valid neighbors: {:?}",
                instance_path,
                problem,
                n,
                h.kind_name(),
                valid,
            )));
        }
    }
    for step in h.steps() {
        validate_heuristic_neighbors(step, problem, instance_path)?;
    }
    Ok(())
}

pub(crate) fn validate_config(config: &BenchmarkConfig) -> Result<(), OptError> {
    if config.num_runs == 0 {
        return Err(OptError::Config(
            "'num_runs' must be at least 1".to_string(),
        ));
    }
    if config.instances.is_empty() {
        return Err(OptError::Config(
            "at least one [[instances]] entry is required".to_string(),
        ));
    }
    if config.heuristics.is_empty() {
        return Err(OptError::Config(
            "at least one [[heuristics]] entry is required".to_string(),
        ));
    }
    // Reject (problem, neighbor) mismatches early — before opening any file —
    // so a typo in a long benchmark TOML fails at startup, not mid-run.
    for inst in &config.instances {
        for h in &config.heuristics {
            validate_heuristic_neighbors(h, &inst.problem, &inst.path)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod validate_tests {
    use super::*;

    fn instance(problem: ProblemKind) -> InstanceConfig {
        InstanceConfig {
            path: "dummy".to_string(),
            problem,
        }
    }

    fn local_search(neighbor: NeighborKind) -> HeuristicConfig {
        HeuristicConfig::LocalSearch {
            neighbor,
            stop_condition: StopConditionConfig::default(),
        }
    }

    fn cfg(instances: Vec<InstanceConfig>, heuristics: Vec<HeuristicConfig>) -> BenchmarkConfig {
        BenchmarkConfig {
            num_runs: 1,
            instances,
            heuristics,
            seed: None,
        }
    }

    #[test]
    fn validate_accepts_compatible_problem_and_neighbor() {
        let c = cfg(
            vec![instance(ProblemKind::MaxCut)],
            vec![local_search(NeighborKind::Flip)],
        );
        validate_config(&c).expect("MaxCut x Flip is valid");
    }

    #[test]
    fn validate_rejects_tsp_with_flip() {
        let c = cfg(
            vec![instance(ProblemKind::Tsp)],
            vec![local_search(NeighborKind::Flip)],
        );
        let err = validate_config(&c).expect_err("Tsp x Flip must fail");
        let msg = format!("{err}");
        assert!(msg.contains("Tsp"), "error mentions Tsp: {msg}");
        assert!(msg.contains("Flip"), "error mentions Flip: {msg}");
        assert!(
            msg.contains("TwoOpt"),
            "error suggests valid neighbors: {msg}"
        );
    }

    #[test]
    fn validate_rejects_jobshop_with_flip() {
        let c = cfg(
            vec![instance(ProblemKind::JobShop)],
            vec![local_search(NeighborKind::Flip)],
        );
        validate_config(&c).expect_err("JobShop x Flip must fail");
    }

    #[test]
    fn validate_recurses_into_nested_steps() {
        // Iterated whose search step has an invalid neighbor for the problem.
        let outer = HeuristicConfig::Iterated {
            steps: vec![
                local_search(NeighborKind::Flip),
                local_search(NeighborKind::Flip),
            ],
            stop_condition: StopConditionConfig::default(),
        };
        let c = cfg(vec![instance(ProblemKind::Tsp)], vec![outer]);
        let err = validate_config(&c).expect_err("nested invalid neighbor must fail");
        let msg = format!("{err}");
        assert!(msg.contains("Flip"), "error mentions Flip: {msg}");
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;

    /// Every documented TOML shape must keep parsing unchanged -- this is the
    /// compatibility contract for the internally-tagged enum.
    #[test]
    fn parses_flat_base_heuristic_toml() {
        let h: HeuristicConfig = toml::from_str(
            r#"
kind = "TabuSearch"
neighbor = "Flip"
tabu_tenure = [5, 10]

[stop_condition]
max_iteration = 1000
"#,
        )
        .expect("TabuSearch TOML parses");
        match &h {
            HeuristicConfig::TabuSearch {
                neighbor,
                tabu_tenure,
                stop_condition,
            } => {
                assert_eq!(*neighbor, NeighborKind::Flip);
                assert_eq!(*tabu_tenure, (5, 10));
                assert_eq!(stop_condition.max_iteration, Some(1000));
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn parses_nested_steps_toml() {
        let h: HeuristicConfig = toml::from_str(
            r#"
kind = "Iterated"

[stop_condition]
max_iteration = 100

[[steps]]
kind = "LocalSearch"
neighbor = "Flip"

[[steps]]
kind = "SimulatedAnnealing"
neighbor = "Flip"
initial_temperature = 1.0
cooling_rate = 0.99
"#,
        )
        .expect("Iterated TOML parses");
        assert_eq!(h.kind_name(), "Iterated");
        assert_eq!(h.steps().len(), 2);
        assert_eq!(h.steps()[1].kind_name(), "SimulatedAnnealing");
    }

    #[test]
    fn missing_required_field_fails_at_parse_time() {
        let err = toml::from_str::<HeuristicConfig>(
            r#"
kind = "TabuSearch"
neighbor = "Flip"
"#,
        )
        .expect_err("missing tabu_tenure must fail");
        assert!(
            err.to_string().contains("tabu_tenure"),
            "error names the missing field: {err}"
        );
    }

    #[test]
    fn unknown_kind_fails_at_parse_time() {
        toml::from_str::<HeuristicConfig>(r#"kind = "NoSuchHeuristic""#)
            .expect_err("unknown kind must fail");
    }

    #[test]
    fn serialization_round_trips() {
        let h = HeuristicConfig::SimulatedAnnealing {
            neighbor: NeighborKind::Swap,
            initial_temperature: 2.5,
            cooling_rate: 0.9,
            stop_condition: StopConditionConfig {
                max_iteration: Some(42),
                max_duration_secs: None,
                max_failed_update: None,
            },
        };
        let toml_str = toml::to_string(&h).expect("serializes");
        assert!(
            toml_str.contains("kind = \"SimulatedAnnealing\""),
            "tag serializes as kind: {toml_str}"
        );
        let back: HeuristicConfig = toml::from_str(&toml_str).expect("round-trips");
        assert_eq!(back.kind_name(), "SimulatedAnnealing");
        assert_eq!(back.stop_condition().max_iteration, Some(42));
    }
}
