//! Benchmark runner for comparing heuristics across problem instances.
//!
//! [`Benchmark::run_from_config`] runs every `(instance, heuristic)` pair from
//! a [`BenchmarkConfig`] (parsed from TOML) and returns a [`BenchmarkReport`]
//! that can be serialized back to TOML for offline analysis.

use rayon::prelude::*;

use crate::{
    error::OptError,
    heuristic::{
        BreakoutLocalSearchForMaxCut, GeneticAlgorithm, Heuristic, Iterated,
        LateAcceptanceHillClimbing, LinKernighanHelsgottForTsp, LocalSearch, ParentSelection,
        RLSearch, Restart, RewardShaping, Sequential, SimulatedAnnealing, StopCondition,
        TabuSearch,
    },
    problem::{
        JobShopPpxCrossover, JobShopRelocateNeighbor, JobShopScheduling, JobShopSolution,
        JobShopSwapNeighbor, MaxCutFlipNeighbor, MaxCutSolution, MaxCutSwapNeighbor,
        MaxCutUniformCrossover, QuboFlipNeighbor, QuboSwapNeighbor, QuboUniformCrossover,
        SatUniformCrossover, TspOrderCrossover, VertexCover, VertexCoverFlipNeighbor,
        VertexCoverSolution, VertexCoverSwapNeighbor, VertexCoverUniformCrossover,
        max_cut::MaxCut,
        qubo::{Qubo, QuboSolution},
        sat::{Sat, SatFlipNeighbor, SatSolution, SatSwapNeighbor},
        tsp_2d::{TspRelocateNeighbor, TspSolution, TspTwoOptNeighbor, TspWithCoordinates},
    },
    search_state::{Crossover, Distance, MoveToNeighbor, ProblemTrait, SearchState},
    trait_defs::{EnabledTabu, Evaluate, Rankable},
};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// BenchmarkProblem / BenchmarkSolution traits
// ---------------------------------------------------------------------------

/// Problem types that can load an instance from a file path.
pub trait BenchmarkProblem: ProblemTrait + Sized {
    fn load_instance(path: &str) -> Result<Self, OptError>;
}

/// Solution types that expose generic metrics needed by the benchmark runner.
pub trait BenchmarkSolution: Clone {
    fn best_objective_f64(&self) -> f64;
    fn encode_as_indices(&self) -> Vec<usize>;
}

impl BenchmarkProblem for MaxCut {
    fn load_instance(path: &str) -> Result<Self, OptError> {
        MaxCut::load_file(path)
    }
}

impl BenchmarkSolution for MaxCutSolution {
    fn best_objective_f64(&self) -> f64 {
        self.objective as f64
    }
    fn encode_as_indices(&self) -> Vec<usize> {
        self.x
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v)
            .map(|(i, _)| i)
            .collect()
    }
}

impl BenchmarkProblem for Qubo {
    fn load_instance(path: &str) -> Result<Self, OptError> {
        Qubo::load_file(path)
    }
}

impl BenchmarkSolution for QuboSolution {
    fn best_objective_f64(&self) -> f64 {
        self.objective as f64
    }
    fn encode_as_indices(&self) -> Vec<usize> {
        self.x
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v)
            .map(|(i, _)| i)
            .collect()
    }
}

impl BenchmarkProblem for Sat {
    fn load_instance(path: &str) -> Result<Self, OptError> {
        Sat::load_file(path)
    }
}

impl BenchmarkSolution for SatSolution {
    fn best_objective_f64(&self) -> f64 {
        self.n_satisfied as f64
    }
    fn encode_as_indices(&self) -> Vec<usize> {
        self.x
            .iter()
            .enumerate()
            .filter(|&(_, v)| *v)
            .map(|(i, _)| i)
            .collect()
    }
}

impl BenchmarkProblem for TspWithCoordinates {
    fn load_instance(path: &str) -> Result<Self, OptError> {
        TspWithCoordinates::load_file(path)
    }
}

impl BenchmarkSolution for TspSolution {
    fn best_objective_f64(&self) -> f64 {
        self.objective
    }
    fn encode_as_indices(&self) -> Vec<usize> {
        self.tour.clone()
    }
}

impl BenchmarkProblem for VertexCover {
    fn load_instance(path: &str) -> Result<Self, OptError> {
        VertexCover::load_file(path)
    }
}

impl BenchmarkSolution for VertexCoverSolution {
    fn best_objective_f64(&self) -> f64 {
        // Use penalty-augmented objective so infeasible solutions are correctly penalized.
        self.objective as f64
    }
    fn encode_as_indices(&self) -> Vec<usize> {
        self.x
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v)
            .map(|(i, _)| i)
            .collect()
    }
}

impl BenchmarkProblem for JobShopScheduling {
    fn load_instance(path: &str) -> Result<Self, OptError> {
        JobShopScheduling::load_file(path)
    }
}

impl BenchmarkSolution for JobShopSolution {
    fn best_objective_f64(&self) -> f64 {
        self.objective as f64
    }
    fn encode_as_indices(&self) -> Vec<usize> {
        self.operations.clone()
    }
}

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
    RLSearch {
        neighbor: NeighborKind,
        /// Learning rate (0.0 = evaluation mode). Default: 0.01.
        #[serde(skip_serializing_if = "Option::is_none")]
        learning_rate: Option<f64>,
        /// Discount factor. Default: 0.99.
        #[serde(skip_serializing_if = "Option::is_none")]
        discount: Option<f64>,
        /// Softmax temperature. Default: 1.0.
        #[serde(skip_serializing_if = "Option::is_none")]
        softmax_temperature: Option<f64>,
        /// Reward shaping strategy: "Raw", "Normalized" (default), "BestImprovement".
        #[serde(skip_serializing_if = "Option::is_none")]
        reward_shaping: Option<String>,
        /// Pre-trained policy weights.
        #[serde(skip_serializing_if = "Option::is_none")]
        policy_weights: Option<Vec<f64>>,
        /// Max candidate moves evaluated per step.
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
        #[serde(default)]
        stop_condition: StopConditionConfig,
    },
    /// Lin-Kernighan-Helsgaun (TSP only).
    LinKernighanHelsgott {
        /// Candidate neighbors per city. Default: 5.
        #[serde(skip_serializing_if = "Option::is_none")]
        num_neighbors: Option<usize>,
        /// Max LK move depth. Default: 5.
        #[serde(skip_serializing_if = "Option::is_none")]
        max_depth: Option<usize>,
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
            Self::RLSearch { .. } => "RLSearch",
            Self::BreakoutLocalSearch { .. } => "BreakoutLocalSearch",
            Self::LinKernighanHelsgott { .. } => "LinKernighanHelsgott",
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
            | Self::RLSearch { neighbor, .. } => Some(neighbor),
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
            | Self::RLSearch { stop_condition, .. }
            | Self::BreakoutLocalSearch { stop_condition, .. }
            | Self::LinKernighanHelsgott { stop_condition, .. }
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

// ---------------------------------------------------------------------------
// Result types (Serialize only) — written to the output TOML report
// ---------------------------------------------------------------------------

/// Result of a single heuristic run on a single instance.
#[derive(Serialize)]
pub struct SingleRunResult {
    pub run_index: usize,
    pub status: String,
    pub best_objective: f64,
    pub best_iteration: u64,
    pub time_to_best_secs: f64,
    pub total_time_secs: f64,
    /// Objective value of the random initial solution; absent on failed runs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial_objective: Option<f64>,
    /// `best_objective - initial_objective`, sign-corrected so that positive
    /// values always mean improvement (regardless of optimization direction).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub improvement: Option<f64>,
    /// Number of moves the heuristic accepted during the run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_accepted: Option<u64>,
    /// Number of iterations the heuristic advanced without applying a move.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_rejected: Option<u64>,
    /// Number of times the best solution was strictly improved.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_best_updates: Option<u64>,
    /// Per-run seed actually used. Set only when the benchmark config provided a master `seed`.
    /// `SearchState::new_with_seed(instance, seed)` reproduces this single run exactly.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    pub solution: Vec<usize>,
}

/// Summary statistics across all runs for one (instance, heuristic) combination.
///
/// Only successful runs are included in the statistics.
#[derive(Serialize)]
pub struct Summary {
    /// Number of successful runs used to compute the statistics.
    pub num_successful_runs: usize,
    pub best_objective: f64,
    pub avg_objective: f64,
    pub worst_objective: f64,
    /// Population standard deviation of the objective across runs.
    pub std_objective: f64,
    pub best_time_to_best_secs: f64,
    pub avg_time_to_best_secs: f64,
    pub avg_total_time_secs: f64,
    /// Average objective value of the random initial solution.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_initial_objective: Option<f64>,
    /// Average improvement (sign-corrected) from initial to best across runs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_improvement: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_n_accepted: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_n_rejected: Option<f64>,
    /// Average acceptance rate across runs:
    /// `n_accepted / (n_accepted + n_rejected)`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_acceptance_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_n_best_updates: Option<f64>,
}

/// All runs for one (instance, heuristic) combination.
#[derive(Serialize)]
pub struct InstanceHeuristicResult {
    pub instance_path: String,
    pub heuristic: HeuristicConfig,
    pub summary: Summary,
    pub runs: Vec<SingleRunResult>,
}

/// Top-level benchmark report written to the output TOML file.
#[derive(Serialize)]
pub struct BenchmarkReport {
    pub timestamp: String,
    pub config_file: String,
    pub results: Vec<InstanceHeuristicResult>,
}

// ---------------------------------------------------------------------------
// ConfigurableProblem: per-problem registration for the config-driven factory
// ---------------------------------------------------------------------------

/// Bound bundle every config-selectable neighborhood move must satisfy.
///
/// Blanket-implemented -- problems never implement it by hand. If a future
/// neighbor type cannot satisfy one of the bounds (e.g. `Evaluate`), split
/// the bundle instead of writing stub impls.
trait ConfigNeighbor<P: ProblemTrait>:
    MoveToNeighbor<P> + Rankable + Evaluate + EnabledTabu + Clone + 'static
{
}
impl<P: ProblemTrait, T> ConfigNeighbor<P> for T where
    T: MoveToNeighbor<P> + Rankable + Evaluate + EnabledTabu + Clone + 'static
{
}

/// Callback invoked with the concrete neighbor type chosen from config.
trait NeighborVisitor<P: ProblemTrait> {
    type Output;
    fn visit<N: ConfigNeighbor<P>>(self) -> Self::Output;
}

/// Per-problem registration point for the benchmark factory.
///
/// Adding a new problem to the benchmark requires exactly:
/// 1. a variant in [`ProblemKind`],
/// 2. an arm in [`with_problem`],
/// 3. an impl of this trait (plus [`BenchmarkProblem`] / [`BenchmarkSolution`]).
trait ConfigurableProblem: BenchmarkProblem + 'static
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
fn invalid_neighbor<P>(kind: &NeighborKind) -> OptError
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

impl ConfigurableProblem for MaxCut {
    const NAME: &'static str = "MaxCut";
    const MINIMIZE: bool = false;
    const VALID_NEIGHBORS: &'static [NeighborKind] = &[NeighborKind::Flip, NeighborKind::Swap];

    fn with_neighbor<V: NeighborVisitor<Self>>(
        kind: &NeighborKind,
        visitor: V,
    ) -> Result<V::Output, OptError> {
        match kind {
            NeighborKind::Flip => Ok(visitor.visit::<MaxCutFlipNeighbor>()),
            NeighborKind::Swap => Ok(visitor.visit::<MaxCutSwapNeighbor>()),
            other => Err(invalid_neighbor::<Self>(other)),
        }
    }

    fn build_special_heuristic(
        config: &HeuristicConfig,
        cond: StopCondition,
    ) -> Result<Box<dyn Heuristic<Self>>, OptError> {
        match config {
            HeuristicConfig::BreakoutLocalSearch {
                tabu_tenure,
                t,
                l0,
                p0,
                q,
                ..
            } => Ok(Box::new(BreakoutLocalSearchForMaxCut::new(
                cond,
                *tabu_tenure,
                *t,
                *l0,
                *p0,
                *q,
            ))),
            _ => Err(OptError::Config(format!(
                "heuristic '{}' is not supported for MaxCut",
                config.kind_name()
            ))),
        }
    }

    fn build_crossover(kind: Option<&str>) -> Result<Box<dyn Crossover<Self>>, OptError> {
        match kind.unwrap_or("Uniform") {
            "Uniform" => Ok(Box::new(MaxCutUniformCrossover)),
            other => Err(OptError::Config(format!(
                "Unknown crossover_kind '{other}' for MaxCut (expected 'Uniform')"
            ))),
        }
    }
}

impl ConfigurableProblem for Qubo {
    const NAME: &'static str = "Qubo";
    const MINIMIZE: bool = true;
    const VALID_NEIGHBORS: &'static [NeighborKind] = &[NeighborKind::Flip, NeighborKind::Swap];

    fn with_neighbor<V: NeighborVisitor<Self>>(
        kind: &NeighborKind,
        visitor: V,
    ) -> Result<V::Output, OptError> {
        match kind {
            NeighborKind::Flip => Ok(visitor.visit::<QuboFlipNeighbor>()),
            NeighborKind::Swap => Ok(visitor.visit::<QuboSwapNeighbor>()),
            other => Err(invalid_neighbor::<Self>(other)),
        }
    }

    fn build_crossover(kind: Option<&str>) -> Result<Box<dyn Crossover<Self>>, OptError> {
        match kind.unwrap_or("Uniform") {
            "Uniform" => Ok(Box::new(QuboUniformCrossover)),
            other => Err(OptError::Config(format!(
                "Unknown crossover_kind '{other}' for Qubo (expected 'Uniform')"
            ))),
        }
    }
}

impl ConfigurableProblem for Sat {
    const NAME: &'static str = "Sat";
    const MINIMIZE: bool = false;
    const VALID_NEIGHBORS: &'static [NeighborKind] = &[NeighborKind::Flip, NeighborKind::Swap];

    fn with_neighbor<V: NeighborVisitor<Self>>(
        kind: &NeighborKind,
        visitor: V,
    ) -> Result<V::Output, OptError> {
        match kind {
            NeighborKind::Flip => Ok(visitor.visit::<SatFlipNeighbor>()),
            NeighborKind::Swap => Ok(visitor.visit::<SatSwapNeighbor>()),
            other => Err(invalid_neighbor::<Self>(other)),
        }
    }

    fn build_crossover(kind: Option<&str>) -> Result<Box<dyn Crossover<Self>>, OptError> {
        match kind.unwrap_or("Uniform") {
            "Uniform" => Ok(Box::new(SatUniformCrossover)),
            other => Err(OptError::Config(format!(
                "Unknown crossover_kind '{other}' for Sat (expected 'Uniform')"
            ))),
        }
    }
}

impl ConfigurableProblem for TspWithCoordinates {
    const NAME: &'static str = "Tsp";
    const MINIMIZE: bool = true;
    const VALID_NEIGHBORS: &'static [NeighborKind] =
        &[NeighborKind::TwoOpt, NeighborKind::Relocate];

    fn with_neighbor<V: NeighborVisitor<Self>>(
        kind: &NeighborKind,
        visitor: V,
    ) -> Result<V::Output, OptError> {
        match kind {
            NeighborKind::TwoOpt => Ok(visitor.visit::<TspTwoOptNeighbor>()),
            NeighborKind::Relocate => Ok(visitor.visit::<TspRelocateNeighbor>()),
            other => Err(invalid_neighbor::<Self>(other)),
        }
    }

    fn build_special_heuristic(
        config: &HeuristicConfig,
        cond: StopCondition,
    ) -> Result<Box<dyn Heuristic<Self>>, OptError> {
        match config {
            HeuristicConfig::LinKernighanHelsgott {
                num_neighbors,
                max_depth,
                ..
            } => Ok(Box::new(LinKernighanHelsgottForTsp::new(
                cond,
                num_neighbors.unwrap_or(5),
                max_depth.unwrap_or(5),
            ))),
            _ => Err(OptError::Config(format!(
                "heuristic '{}' is not supported for Tsp",
                config.kind_name()
            ))),
        }
    }

    fn build_crossover(kind: Option<&str>) -> Result<Box<dyn Crossover<Self>>, OptError> {
        match kind.unwrap_or("Order") {
            "Order" => Ok(Box::new(TspOrderCrossover)),
            other => Err(OptError::Config(format!(
                "Unknown crossover_kind '{other}' for Tsp (expected 'Order')"
            ))),
        }
    }
}

impl ConfigurableProblem for VertexCover {
    const NAME: &'static str = "VertexCover";
    const MINIMIZE: bool = true;
    const VALID_NEIGHBORS: &'static [NeighborKind] = &[NeighborKind::Flip, NeighborKind::Swap];

    fn with_neighbor<V: NeighborVisitor<Self>>(
        kind: &NeighborKind,
        visitor: V,
    ) -> Result<V::Output, OptError> {
        match kind {
            NeighborKind::Flip => Ok(visitor.visit::<VertexCoverFlipNeighbor>()),
            NeighborKind::Swap => Ok(visitor.visit::<VertexCoverSwapNeighbor>()),
            other => Err(invalid_neighbor::<Self>(other)),
        }
    }

    fn build_crossover(kind: Option<&str>) -> Result<Box<dyn Crossover<Self>>, OptError> {
        match kind.unwrap_or("Uniform") {
            "Uniform" => Ok(Box::new(VertexCoverUniformCrossover)),
            other => Err(OptError::Config(format!(
                "Unknown crossover_kind '{other}' for VertexCover (expected 'Uniform')"
            ))),
        }
    }
}

impl ConfigurableProblem for JobShopScheduling {
    const NAME: &'static str = "JobShop";
    const MINIMIZE: bool = true;
    const VALID_NEIGHBORS: &'static [NeighborKind] = &[NeighborKind::Swap, NeighborKind::Relocate];

    fn with_neighbor<V: NeighborVisitor<Self>>(
        kind: &NeighborKind,
        visitor: V,
    ) -> Result<V::Output, OptError> {
        match kind {
            NeighborKind::Swap => Ok(visitor.visit::<JobShopSwapNeighbor>()),
            NeighborKind::Relocate => Ok(visitor.visit::<JobShopRelocateNeighbor>()),
            other => Err(invalid_neighbor::<Self>(other)),
        }
    }

    fn build_crossover(kind: Option<&str>) -> Result<Box<dyn Crossover<Self>>, OptError> {
        match kind.unwrap_or("Ppx") {
            "Ppx" => Ok(Box::new(JobShopPpxCrossover)),
            other => Err(OptError::Config(format!(
                "Unknown crossover_kind '{other}' for JobShop (expected 'Ppx')"
            ))),
        }
    }
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
            HeuristicConfig::RLSearch { .. } => build_rl_search::<P, N>(self.config, self.cond),
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

    let HeuristicConfig::RLSearch {
        learning_rate,
        discount,
        softmax_temperature,
        reward_shaping,
        policy_weights,
        max_candidates,
        ..
    } = config
    else {
        unreachable!("build_rl_search called with a non-RLSearch config");
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

    let mut rl = RLSearch::<N>::new(
        cond,
        learning_rate.unwrap_or(0.01),
        discount.unwrap_or(0.99),
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
fn build_heuristic<P>(config: &HeuristicConfig) -> Result<Box<dyn Heuristic<P>>, OptError>
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
        | HeuristicConfig::LinKernighanHelsgott { .. } => P::build_special_heuristic(config, cond),
        HeuristicConfig::LocalSearch { neighbor, .. }
        | HeuristicConfig::TabuSearch { neighbor, .. }
        | HeuristicConfig::SimulatedAnnealing { neighbor, .. }
        | HeuristicConfig::LateAcceptanceHillClimbing { neighbor, .. }
        | HeuristicConfig::RLSearch { neighbor, .. } => {
            P::with_neighbor(neighbor, BaseBuilder { config, cond })?
        }
    }
}

// ---------------------------------------------------------------------------
// Runtime problem dispatch -- the single place mapping ProblemKind to types
// ---------------------------------------------------------------------------

/// Callback invoked with the concrete problem type for a [`ProblemKind`].
trait ProblemVisitor {
    type Output;
    fn visit<P>(self) -> Self::Output
    where
        P: ConfigurableProblem,
        P::Solution: BenchmarkSolution + Distance;
}

/// Maps the runtime [`ProblemKind`] to the concrete problem type.
fn with_problem<V: ProblemVisitor>(kind: &ProblemKind, visitor: V) -> V::Output {
    match kind {
        ProblemKind::MaxCut => visitor.visit::<MaxCut>(),
        ProblemKind::Qubo => visitor.visit::<Qubo>(),
        ProblemKind::Sat => visitor.visit::<Sat>(),
        ProblemKind::Tsp => visitor.visit::<TspWithCoordinates>(),
        ProblemKind::VertexCover => visitor.visit::<VertexCover>(),
        ProblemKind::JobShop => visitor.visit::<JobShopScheduling>(),
    }
}

struct MinimizeVisitor;
impl ProblemVisitor for MinimizeVisitor {
    type Output = bool;
    fn visit<P>(self) -> bool
    where
        P: ConfigurableProblem,
        P::Solution: BenchmarkSolution + Distance,
    {
        P::MINIMIZE
    }
}

struct ValidNeighborsVisitor;
impl ProblemVisitor for ValidNeighborsVisitor {
    type Output = &'static [NeighborKind];
    fn visit<P>(self) -> &'static [NeighborKind]
    where
        P: ConfigurableProblem,
        P::Solution: BenchmarkSolution + Distance,
    {
        P::VALID_NEIGHBORS
    }
}

impl ProblemKind {
    /// Whether this problem is minimized (drives report statistics).
    pub fn minimize(&self) -> bool {
        with_problem(self, MinimizeVisitor)
    }

    /// The neighborhood kinds supported by this problem.
    pub fn valid_neighbors(&self) -> &'static [NeighborKind] {
        with_problem(self, ValidNeighborsVisitor)
    }
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

/// Namespace for the benchmark entry point ([`Benchmark::run_from_config`]).
pub struct Benchmark;

// ---------------------------------------------------------------------------
// Generic run functions
// ---------------------------------------------------------------------------

struct RunMetrics {
    status: String,
    best_objective: f64,
    best_iteration: u64,
    time_to_best_secs: f64,
    total_time_secs: f64,
    initial_objective: Option<f64>,
    improvement: Option<f64>,
    n_accepted: Option<u64>,
    n_rejected: Option<u64>,
    n_best_updates: Option<u64>,
    seed: Option<u64>,
    solution: Vec<usize>,
}

fn empty_metrics(status: String, seed: Option<u64>) -> RunMetrics {
    RunMetrics {
        status,
        best_objective: 0.0,
        best_iteration: 0,
        time_to_best_secs: 0.0,
        total_time_secs: 0.0,
        initial_objective: None,
        improvement: None,
        n_accepted: None,
        n_rejected: None,
        n_best_updates: None,
        seed,
        solution: Vec::new(),
    }
}

/// Derives a per-run u64 seed from the master seed and the run coordinates.
/// Using a hash keeps the seed deterministic and well-mixed even when the master
/// is small (e.g. `seed = 0`).
fn derive_run_seed(
    master: u64,
    instance_path: &str,
    heuristic_idx: usize,
    run_index: usize,
) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    master.hash(&mut h);
    instance_path.hash(&mut h);
    (heuristic_idx as u64).hash(&mut h);
    (run_index as u64).hash(&mut h);
    // Clear the top bit so the seed fits in TOML's signed i64 range when
    // serialized into the benchmark report.
    h.finish() & 0x7FFF_FFFF_FFFF_FFFF
}

struct RunVisitor<'a> {
    config: &'a HeuristicConfig,
    instance_path: &'a str,
    seed: Option<u64>,
}

impl ProblemVisitor for RunVisitor<'_> {
    type Output = RunMetrics;
    fn visit<P>(self) -> RunMetrics
    where
        P: ConfigurableProblem,
        P::Solution: BenchmarkSolution + Distance,
    {
        run_typed::<P>(self.instance_path, self.config, self.seed)
    }
}

fn run_for_problem_kind(
    problem_kind: &ProblemKind,
    config: &HeuristicConfig,
    instance_path: &str,
    seed: Option<u64>,
) -> RunMetrics {
    with_problem(
        problem_kind,
        RunVisitor {
            config,
            instance_path,
            seed,
        },
    )
}

fn run_typed<P>(instance_path: &str, config: &HeuristicConfig, seed: Option<u64>) -> RunMetrics
where
    P: ConfigurableProblem,
    P::Solution: BenchmarkSolution + Distance,
{
    let heuristic = match build_heuristic::<P>(config) {
        Ok(h) => h,
        Err(e) => {
            let msg = match e {
                OptError::Config(m) => m,
                other => other.to_string(),
            };
            return empty_metrics(format!("config error: {msg}"), seed);
        }
    };
    run_problem::<P>(instance_path, heuristic, P::MINIMIZE, seed)
}

fn run_problem<P>(
    instance_path: &str,
    mut heuristic: Box<dyn Heuristic<P>>,
    minimize: bool,
    seed: Option<u64>,
) -> RunMetrics
where
    P: BenchmarkProblem,
    P::Solution: BenchmarkSolution,
{
    let instance = match P::load_instance(instance_path) {
        Ok(v) => v,
        Err(e) => return empty_metrics(format!("error loading instance: {}", e), seed),
    };
    let mut state = match seed {
        Some(s) => SearchState::new_with_seed(&instance, s),
        None => SearchState::new(&instance),
    };
    let initial_objective = state.initial_solution.best_objective_f64();
    let start = std::time::Instant::now();
    let status = heuristic.run(&mut state);
    let total_time = start.elapsed();
    let best_objective = state.best_solution.best_objective_f64();
    let raw_diff = best_objective - initial_objective;
    let improvement = if minimize { -raw_diff } else { raw_diff };
    RunMetrics {
        status: status_str(status),
        best_objective,
        best_iteration: state.best_iteration,
        time_to_best_secs: (state.best_time - state.start_time).as_secs_f64(),
        total_time_secs: total_time.as_secs_f64(),
        initial_objective: Some(initial_objective),
        improvement: Some(improvement),
        n_accepted: Some(state.n_accepted),
        n_rejected: Some(state.n_rejected),
        n_best_updates: Some(state.n_best_updates),
        seed,
        solution: state.best_solution.encode_as_indices(),
    }
}

fn status_str(r: Result<(), crate::error::OptError>) -> String {
    match r {
        Ok(_) => "success".to_string(),
        Err(e) => format!("error: {}", e),
    }
}

fn compute_summary(runs: &[SingleRunResult], minimize: bool) -> Summary {
    let successful: Vec<&SingleRunResult> = runs.iter().filter(|r| r.status == "success").collect();
    let n = successful.len();
    if n == 0 {
        return Summary {
            num_successful_runs: 0,
            best_objective: f64::NAN,
            avg_objective: f64::NAN,
            worst_objective: f64::NAN,
            std_objective: f64::NAN,
            best_time_to_best_secs: f64::NAN,
            avg_time_to_best_secs: f64::NAN,
            avg_total_time_secs: f64::NAN,
            avg_initial_objective: None,
            avg_improvement: None,
            avg_n_accepted: None,
            avg_n_rejected: None,
            avg_acceptance_rate: None,
            avg_n_best_updates: None,
        };
    }
    let objectives: Vec<f64> = successful.iter().map(|r| r.best_objective).collect();
    let best = if minimize {
        objectives.iter().cloned().fold(f64::INFINITY, f64::min)
    } else {
        objectives.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    };
    let worst = if minimize {
        objectives.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    } else {
        objectives.iter().cloned().fold(f64::INFINITY, f64::min)
    };
    let avg = objectives.iter().sum::<f64>() / n as f64;
    let variance = objectives.iter().map(|&x| (x - avg).powi(2)).sum::<f64>() / n as f64;
    let std = variance.sqrt();

    let times_to_best: Vec<f64> = successful.iter().map(|r| r.time_to_best_secs).collect();
    let best_ttb = times_to_best.iter().cloned().fold(f64::INFINITY, f64::min);
    let avg_ttb = times_to_best.iter().sum::<f64>() / n as f64;
    let avg_total = successful.iter().map(|r| r.total_time_secs).sum::<f64>() / n as f64;

    let avg_opt = |xs: Vec<f64>| -> Option<f64> {
        if xs.len() == n {
            Some(xs.iter().sum::<f64>() / n as f64)
        } else {
            None
        }
    };

    let avg_initial_objective = avg_opt(
        successful
            .iter()
            .filter_map(|r| r.initial_objective)
            .collect(),
    );
    let avg_improvement = avg_opt(successful.iter().filter_map(|r| r.improvement).collect());
    let accepted_vals: Vec<u64> = successful.iter().filter_map(|r| r.n_accepted).collect();
    let rejected_vals: Vec<u64> = successful.iter().filter_map(|r| r.n_rejected).collect();
    let best_vals: Vec<u64> = successful.iter().filter_map(|r| r.n_best_updates).collect();
    let avg_n_accepted = if accepted_vals.len() == n {
        Some(accepted_vals.iter().map(|&v| v as f64).sum::<f64>() / n as f64)
    } else {
        None
    };
    let avg_n_rejected = if rejected_vals.len() == n {
        Some(rejected_vals.iter().map(|&v| v as f64).sum::<f64>() / n as f64)
    } else {
        None
    };
    let avg_acceptance_rate = if accepted_vals.len() == n && rejected_vals.len() == n {
        let rates: Vec<f64> = accepted_vals
            .iter()
            .zip(rejected_vals.iter())
            .map(|(&a, &r)| {
                let total = a + r;
                if total == 0 {
                    0.0
                } else {
                    a as f64 / total as f64
                }
            })
            .collect();
        Some(rates.iter().sum::<f64>() / n as f64)
    } else {
        None
    };
    let avg_n_best_updates = if best_vals.len() == n {
        Some(best_vals.iter().map(|&v| v as f64).sum::<f64>() / n as f64)
    } else {
        None
    };

    Summary {
        num_successful_runs: n,
        best_objective: best,
        avg_objective: avg,
        worst_objective: worst,
        std_objective: std,
        best_time_to_best_secs: best_ttb,
        avg_time_to_best_secs: avg_ttb,
        avg_total_time_secs: avg_total,
        avg_initial_objective,
        avg_improvement,
        avg_n_accepted,
        avg_n_rejected,
        avg_acceptance_rate,
        avg_n_best_updates,
    }
}

fn to_single_run_result(run_index: usize, m: RunMetrics) -> SingleRunResult {
    SingleRunResult {
        run_index,
        status: m.status,
        best_objective: m.best_objective,
        best_iteration: m.best_iteration,
        time_to_best_secs: m.time_to_best_secs,
        total_time_secs: m.total_time_secs,
        initial_objective: m.initial_objective,
        improvement: m.improvement,
        n_accepted: m.n_accepted,
        n_rejected: m.n_rejected,
        n_best_updates: m.n_best_updates,
        seed: m.seed,
        solution: m.solution,
    }
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

fn validate_config(config: &BenchmarkConfig) -> Result<(), OptError> {
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

fn expand_instance_paths(config: &BenchmarkConfig) -> Result<Vec<(String, ProblemKind)>, OptError> {
    let mut instance_paths: Vec<(String, ProblemKind)> = Vec::new();

    for inst in &config.instances {
        let paths = glob::glob(&inst.path).map_err(|e| {
            OptError::Config(format!("invalid glob pattern '{}': {}", inst.path, e))
        })?;

        let mut expanded: Vec<_> = paths.collect::<Result<Vec<_>, _>>().map_err(|e| {
            OptError::Config(format!("glob entry error for '{}': {}", inst.path, e))
        })?;
        expanded.sort();

        if expanded.is_empty() {
            return Err(OptError::Config(format!(
                "instance pattern '{}' matched no files",
                inst.path
            )));
        }

        for path in expanded {
            let Some(path_str) = path.to_str() else {
                return Err(OptError::Config(format!(
                    "instance path '{}' is not valid UTF-8",
                    path.display()
                )));
            };
            instance_paths.push((path_str.to_string(), inst.problem.clone()));
        }
    }

    Ok(instance_paths)
}

impl Benchmark {
    /// Runs all (instance x heuristic) combinations from a `BenchmarkConfig` and returns a report.
    ///
    /// Instance paths support glob patterns (e.g. `"data/instances/max_cut/G[1-9]*"`).
    /// For each combination, the heuristic is run `config.num_runs` times.
    pub fn run_from_config(
        config: BenchmarkConfig,
        config_file: &str,
    ) -> Result<BenchmarkReport, OptError> {
        validate_config(&config)?;
        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
        let mut results: Vec<InstanceHeuristicResult> = Vec::new();
        let instance_paths = expand_instance_paths(&config)?;

        for (instance_path, problem_kind) in &instance_paths {
            for (heuristic_idx, heuristic_cfg) in config.heuristics.iter().enumerate() {
                tracing::info!(
                    instance = %instance_path,
                    heuristic = heuristic_cfg.kind_name(),
                    num_runs = config.num_runs,
                    max_iteration = ?heuristic_cfg.stop_condition().max_iteration,
                    max_duration_secs = ?heuristic_cfg.stop_condition().max_duration_secs,
                    max_failed_update = ?heuristic_cfg.stop_condition().max_failed_update,
                    seed = ?config.seed,
                    "Start:"
                );

                let master_seed = config.seed;
                let mut runs: Vec<SingleRunResult> = (0..config.num_runs)
                    .into_par_iter()
                    .map(|run_index| {
                        let run_seed = master_seed
                            .map(|m| derive_run_seed(m, instance_path, heuristic_idx, run_index));
                        let metrics = run_for_problem_kind(
                            problem_kind,
                            heuristic_cfg,
                            instance_path,
                            run_seed,
                        );

                        tracing::info!(
                            run = run_index + 1,
                            objective = metrics.best_objective,
                            best_iteration = metrics.best_iteration,
                            time_to_best_secs = metrics.time_to_best_secs,
                            total_time_secs = metrics.total_time_secs,
                            "Completed:"
                        );

                        to_single_run_result(run_index, metrics)
                    })
                    .collect();
                runs.sort_by_key(|r| r.run_index);

                let summary = compute_summary(&runs, problem_kind.minimize());
                tracing::info!(
                    instance = %instance_path,
                    heuristic = heuristic_cfg.kind_name(),
                    num_runs = summary.num_successful_runs,
                    best = summary.best_objective,
                    avg = summary.avg_objective,
                    worst = summary.worst_objective,
                    std = summary.std_objective,
                    avg_time_to_best_secs = summary.avg_time_to_best_secs,
                    avg_total_time_secs = summary.avg_total_time_secs,
                    "Summary:"
                );
                results.push(InstanceHeuristicResult {
                    instance_path: instance_path.clone(),
                    heuristic: heuristic_cfg.clone(),
                    summary,
                    runs,
                });
            }
        }

        Ok(BenchmarkReport {
            timestamp,
            config_file: config_file.to_string(),
            results,
        })
    }
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

#[cfg(test)]
mod factory_tests {
    use super::*;

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
            HeuristicConfig::RLSearch {
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
            stop_condition: StopConditionConfig::default(),
        };
        try_build(&ProblemKind::MaxCut, &bls).expect("BLS builds for MaxCut");
        let err = try_build(&ProblemKind::Qubo, &bls).expect_err("BLS invalid for Qubo");
        assert!(err.to_string().contains("Qubo"), "{err}");

        let lkh = HeuristicConfig::LinKernighanHelsgott {
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
    fn minimize_truth_table() {
        assert!(!ProblemKind::MaxCut.minimize());
        assert!(!ProblemKind::Sat.minimize());
        assert!(ProblemKind::Qubo.minimize());
        assert!(ProblemKind::Tsp.minimize());
        assert!(ProblemKind::VertexCover.minimize());
        assert!(ProblemKind::JobShop.minimize());
    }
}
