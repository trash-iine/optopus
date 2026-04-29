//! Benchmark runner for comparing heuristics across problem instances.
//!
//! [`Benchmark`] accumulates [`BenchmarkResult`] records produced by [`Benchmark::run`].
//! Each result captures the configuration, best objective value, time-to-best, and solution.
//! Results can be serialized to TOML (or any serde format) for offline analysis.

use rayon::prelude::*;

use crate::{
    common::Graph,
    error::OptError,
    heuristic::{
        BreakoutLocalSearchForMaxCut, GeneticAlgorithm, Heuristic, Iterated,
        LateAcceptanceHillClimbing, LinKernighanHelsgottForTsp, LocalSearch, ParentSelection,
        RLSearch, Restart, RewardShaping, Sequential, SimulatedAnnealing, StopCondition,
        TabuSearch,
    },
    problem::{
        MaxCutFlipNeighbor, MaxCutSolution, MaxCutSwapNeighbor, MaxCutUniformCrossover,
        QuboFlipNeighbor, QuboSwapNeighbor, QuboUniformCrossover, SatUniformCrossover,
        TspOrderCrossover, VertexCover, VertexCoverFlipNeighbor, VertexCoverSolution,
        VertexCoverSwapNeighbor, VertexCoverUniformCrossover,
        max_cut::MaxCut,
        qubo::{Qubo, QuboSolution},
        sat::{Sat, SatFlipNeighbor, SatSolution, SatSwapNeighbor},
        tsp_2d::{TspRelocateNeighbor, TspSolution, TspTwoOptNeighbor, TspWithCoordinates},
    },
    search_state::{Crossover, Distance, MoveToNeighbor, ProblemTrait, SearchState},
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
        Graph::load_from_file(path).map(MaxCut::new)
    }
}

impl BenchmarkSolution for MaxCutSolution {
    fn best_objective_f64(&self) -> f64 {
        self.objective as f64
    }
    fn encode_as_indices(&self) -> Vec<usize> {
        self.cut
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
        Graph::load_from_file(path).map(VertexCover::new)
    }
}

impl BenchmarkSolution for VertexCoverSolution {
    fn best_objective_f64(&self) -> f64 {
        // Use penalty-augmented objective so infeasible solutions are correctly penalized.
        self.objective as f64
    }
    fn encode_as_indices(&self) -> Vec<usize> {
        self.cover
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v)
            .map(|(i, _)| i)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Config types (Deserialize + Serialize) — used as input from a TOML file
// ---------------------------------------------------------------------------

/// Problem type discriminant used in config files.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ProblemKind {
    MaxCut,
    Qubo,
    Sat,
    Tsp,
    VertexCover,
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
/// Uses a flat struct with optional fields; the `kind` string selects the algorithm.
///
/// Valid `kind` values:
/// - `"LocalSearch"`, `"TabuSearch"`, `"SimulatedAnnealing"`, `"LateAcceptanceHillClimbing"`, `"BreakoutLocalSearch"` (MaxCut only)
/// - `"RLSearch"` — reinforcement learning move selection with REINFORCE policy gradient
/// - `"Sequential"` — repeats its `steps` cycle until `stop_condition` is met
/// - `"Iterated"` — `steps\[0\]` = search phase, `steps\[1\]` = perturbation phase (ILS)
/// - `"Restart"` — runs `steps\[0\]` then resets to a new random solution when `restart_condition` is met
/// - `"GeneticAlgorithm"` — `steps\[0\]` = mutation, optional `steps\[1\]` = init_improvement (HEA pattern).
///   Requires `population_size`. Optional fields: `crossover_kind` (default `"Uniform"`,
///   TSP defaults to `"Order"`), `parent_selection` (`"Tournament"` default or `"HammingTopK"`),
///   `parent_top_k` (required when `parent_selection = "HammingTopK"`).
///
/// The problem type is inferred from the instance being benchmarked and does not need
/// to be specified here.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeuristicConfig {
    /// Algorithm name.
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neighbor: Option<NeighborKind>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tabu_tenure: Option<(u64, u64)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial_temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cooling_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub t: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub l0: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p0: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_neighbors: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_depth: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub history_length: Option<usize>,
    #[serde(default)]
    pub stop_condition: StopConditionConfig,
    /// Sub-heuristics for `"Sequential"`, `"Iterated"`, and `"Restart"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps: Option<Vec<HeuristicConfig>>,
    /// Restart trigger for `kind = "Restart"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub restart_condition: Option<StopConditionConfig>,
    /// Learning rate for `kind = "RLSearch"` (0.0 = evaluation mode). Default: 0.01.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learning_rate: Option<f64>,
    /// Discount factor γ for `kind = "RLSearch"`. Default: 0.99.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub discount: Option<f64>,
    /// Softmax temperature for `kind = "RLSearch"`. Default: 1.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub softmax_temperature: Option<f64>,
    /// Reward shaping strategy for `kind = "RLSearch"`: "Raw", "Normalized", "BestImprovement".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reward_shaping: Option<String>,
    /// Pre-trained policy weights for `kind = "RLSearch"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub policy_weights: Option<Vec<f64>>,
    /// Max candidate moves to evaluate per step for `kind = "RLSearch"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_candidates: Option<usize>,
    // ----- GeneticAlgorithm fields -----
    /// Population size for `kind = "GeneticAlgorithm"`. Required, must be >= 2.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub population_size: Option<usize>,
    /// Crossover operator for `kind = "GeneticAlgorithm"`. Currently `"Uniform"`
    /// is supported for every problem; TSP additionally accepts `"Order"`.
    /// Defaults to `"Uniform"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crossover_kind: Option<String>,
    /// Parent selection strategy for `kind = "GeneticAlgorithm"`:
    /// `"Tournament"` (default) or `"HammingTopK"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_selection: Option<String>,
    /// `top_k` for `parent_selection = "HammingTopK"` (must be >= 1).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_top_k: Option<usize>,
}

/// A single instance entry in the config file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstanceConfig {
    /// Path to the instance file; glob patterns (e.g. `"data/max_cut/G[1-9]*"`) are supported.
    pub path: String,
    pub problem: ProblemKind,
}

/// Top-level benchmark configuration read from a TOML file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub num_runs: usize,
    pub instances: Vec<InstanceConfig>,
    pub heuristics: Vec<HeuristicConfig>,
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
// BenchmarkResult
// ---------------------------------------------------------------------------

/// The result of a single experiment run (configuration + metrics).
#[derive(Serialize)]
pub struct BenchmarkResult {
    pub instance_path: String,
    pub problem: ProblemKind,
    pub heuristic: HeuristicConfig,
    /// Run status: `"success"` or `"error: <message>"`.
    pub status: String,
    /// Best objective value found (maximized for MaxCut/SAT, minimized for QUBO/TSP).
    pub best_objective: f64,
    /// Iteration at which the best solution was found.
    pub best_iteration: u64,
    /// Elapsed time (seconds) until the best solution was found.
    pub time_to_best_secs: f64,
    /// Total elapsed time (seconds) for the run.
    pub total_time_secs: f64,
    /// Best solution encoded as a list of indices (0-indexed):
    /// - MaxCut: vertex indices on the cut side
    /// - QUBO: variable indices set to 1
    /// - SAT: variable indices set to `true`
    /// - TSP: city visit order
    pub solution: Vec<usize>,
}

// ---------------------------------------------------------------------------
// BenchmarkableProblem trait + per-problem implementations
// ---------------------------------------------------------------------------

/// Extends [`BenchmarkProblem`] with the ability to build heuristics from config.
///
/// Each problem implements `build_base_heuristic` to dispatch on [`NeighborKind`]
/// and construct the concrete heuristic. Meta-heuristics (Sequential, Iterated,
/// Restart) are handled generically by [`build_heuristic`].
trait BenchmarkableProblem: BenchmarkProblem
where
    Self::Solution: BenchmarkSolution + Distance,
{
    fn build_base_heuristic(
        config: &HeuristicConfig,
        cond: StopCondition,
    ) -> Result<Box<dyn Heuristic<Self>>, String>;

    /// Builds the crossover operator selected by `config.crossover_kind`.
    ///
    /// Default: returns the problem's uniform crossover. Override to add
    /// problem-specific operators (e.g. TSP order crossover).
    fn build_crossover(config: &HeuristicConfig) -> Result<Box<dyn Crossover<Self>>, String>;
}

impl HeuristicConfig {
    fn req_neighbor(&self, problem: &str) -> Result<&NeighborKind, String> {
        self.neighbor
            .as_ref()
            .ok_or_else(|| format!("'neighbor' required for {} {}", problem, self.kind))
    }
    fn req_tabu(&self, problem: &str) -> Result<(u64, u64), String> {
        self.tabu_tenure
            .ok_or_else(|| format!("'tabu_tenure' required for {} {}", problem, self.kind))
    }
    fn req_temp(&self, problem: &str) -> Result<f64, String> {
        self.initial_temperature.ok_or_else(|| {
            format!(
                "'initial_temperature' required for {} {}",
                problem, self.kind
            )
        })
    }
    fn req_cooling(&self, problem: &str) -> Result<f64, String> {
        self.cooling_rate
            .ok_or_else(|| format!("'cooling_rate' required for {} {}", problem, self.kind))
    }
    fn req_history_length(&self, problem: &str) -> Result<usize, String> {
        let len = self
            .history_length
            .ok_or_else(|| format!("'history_length' required for {} {}", problem, self.kind))?;
        if len == 0 {
            return Err(format!(
                "'history_length' must be at least 1 for {} {}",
                problem, self.kind
            ));
        }
        Ok(len)
    }
    fn parse_reward_shaping(&self) -> Result<RewardShaping, String> {
        match self.reward_shaping.as_deref().unwrap_or("Normalized") {
            "Raw" => Ok(RewardShaping::Raw),
            "Normalized" => Ok(RewardShaping::Normalized),
            "BestImprovement" => Ok(RewardShaping::BestImprovement),
            other => Err(format!(
                "Unknown reward_shaping '{}' (expected Raw, Normalized, or BestImprovement)",
                other
            )),
        }
    }
    fn parse_policy_weights(
        &self,
    ) -> Result<
        Option<[f64; crate::heuristic::reinforcement_learning::feature::NUM_FEATURES]>,
        String,
    > {
        use crate::heuristic::reinforcement_learning::feature::NUM_FEATURES;
        match &self.policy_weights {
            None => Ok(None),
            Some(v) => {
                if v.len() != NUM_FEATURES {
                    return Err(format!(
                        "policy_weights must have exactly {} elements, got {}",
                        NUM_FEATURES,
                        v.len()
                    ));
                }
                let mut arr = [0.0; NUM_FEATURES];
                arr.copy_from_slice(v);
                Ok(Some(arr))
            }
        }
    }
}

fn build_rl_search_from_config<P, N>(
    config: &HeuristicConfig,
    cond: StopCondition,
) -> Result<Box<dyn Heuristic<P>>, String>
where
    P: ProblemTrait + 'static,
    N: MoveToNeighbor<P> + crate::search_state::Evaluate + Clone + 'static,
{
    let reward = config.parse_reward_shaping()?;
    let mut rl = RLSearch::<N>::new(
        cond,
        config.learning_rate.unwrap_or(0.01),
        config.discount.unwrap_or(0.99),
        config.softmax_temperature.unwrap_or(1.0),
        reward,
        config.max_candidates,
    );
    if let Some(w) = config.parse_policy_weights()? {
        rl = rl.with_policy_weights(w);
    }
    Ok(Box::new(rl))
}

impl BenchmarkableProblem for MaxCut {
    fn build_base_heuristic(
        config: &HeuristicConfig,
        cond: StopCondition,
    ) -> Result<Box<dyn Heuristic<Self>>, String> {
        match config.kind.as_str() {
            "LocalSearch" => match config.req_neighbor("MaxCut")? {
                NeighborKind::Flip => Ok(Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(cond))),
                NeighborKind::Swap => Ok(Box::new(LocalSearch::<MaxCutSwapNeighbor>::new(cond))),
                n => Err(format!(
                    "Invalid neighbor {:?} for MaxCut (use Flip or Swap)",
                    n
                )),
            },
            "TabuSearch" => {
                let tenure = config.req_tabu("MaxCut")?;
                match config.req_neighbor("MaxCut")? {
                    NeighborKind::Flip => Ok(Box::new(TabuSearch::<MaxCutFlipNeighbor>::new(
                        cond, tenure, None,
                    ))),
                    NeighborKind::Swap => Ok(Box::new(TabuSearch::<MaxCutSwapNeighbor>::new(
                        cond, tenure, None,
                    ))),
                    n => Err(format!(
                        "Invalid neighbor {:?} for MaxCut (use Flip or Swap)",
                        n
                    )),
                }
            }
            "SimulatedAnnealing" => {
                let temp = config.req_temp("MaxCut")?;
                let cooling = config.req_cooling("MaxCut")?;
                match config.req_neighbor("MaxCut")? {
                    NeighborKind::Flip => Ok(Box::new(
                        SimulatedAnnealing::<MaxCutFlipNeighbor>::new(cond, temp, cooling),
                    )),
                    NeighborKind::Swap => Ok(Box::new(
                        SimulatedAnnealing::<MaxCutSwapNeighbor>::new(cond, temp, cooling),
                    )),
                    n => Err(format!(
                        "Invalid neighbor {:?} for MaxCut (use Flip or Swap)",
                        n
                    )),
                }
            }
            "LateAcceptanceHillClimbing" => {
                let history_length = config.req_history_length("MaxCut")?;
                match config.req_neighbor("MaxCut")? {
                    NeighborKind::Flip => Ok(Box::new(LateAcceptanceHillClimbing::<
                        MaxCutFlipNeighbor,
                    >::new(
                        cond, history_length
                    ))),
                    NeighborKind::Swap => Ok(Box::new(LateAcceptanceHillClimbing::<
                        MaxCutSwapNeighbor,
                    >::new(
                        cond, history_length
                    ))),
                    n => Err(format!(
                        "Invalid neighbor {:?} for MaxCut (use Flip or Swap)",
                        n
                    )),
                }
            }
            "BreakoutLocalSearch" => {
                let tenure = config.req_tabu("MaxCut")?;
                let t = config
                    .t
                    .ok_or("'t' required for MaxCut BreakoutLocalSearch")?;
                let l0 = config
                    .l0
                    .ok_or("'l0' required for MaxCut BreakoutLocalSearch")?;
                let p0 = config
                    .p0
                    .ok_or("'p0' required for MaxCut BreakoutLocalSearch")?;
                let q = config
                    .q
                    .ok_or("'q' required for MaxCut BreakoutLocalSearch")?;
                Ok(Box::new(BreakoutLocalSearchForMaxCut::new(
                    tenure, cond, t, l0, p0, q,
                )))
            }
            "RLSearch" => match config.req_neighbor("MaxCut")? {
                NeighborKind::Flip => {
                    build_rl_search_from_config::<MaxCut, MaxCutFlipNeighbor>(config, cond)
                }
                NeighborKind::Swap => {
                    build_rl_search_from_config::<MaxCut, MaxCutSwapNeighbor>(config, cond)
                }
                n => Err(format!(
                    "Invalid neighbor {:?} for MaxCut RLSearch (use Flip or Swap)",
                    n
                )),
            },
            k => Err(format!("Unknown kind '{}' for MaxCut", k)),
        }
    }

    fn build_crossover(config: &HeuristicConfig) -> Result<Box<dyn Crossover<Self>>, String> {
        match config.crossover_kind.as_deref().unwrap_or("Uniform") {
            "Uniform" => Ok(Box::new(MaxCutUniformCrossover)),
            other => Err(format!(
                "Unknown crossover_kind '{other}' for MaxCut (expected 'Uniform')"
            )),
        }
    }
}

impl BenchmarkableProblem for Qubo {
    fn build_base_heuristic(
        config: &HeuristicConfig,
        cond: StopCondition,
    ) -> Result<Box<dyn Heuristic<Self>>, String> {
        match config.kind.as_str() {
            "LocalSearch" => match config.req_neighbor("Qubo")? {
                NeighborKind::Flip => Ok(Box::new(LocalSearch::<QuboFlipNeighbor>::new(cond))),
                NeighborKind::Swap => Ok(Box::new(LocalSearch::<QuboSwapNeighbor>::new(cond))),
                n => Err(format!(
                    "Invalid neighbor {:?} for Qubo (use Flip or Swap)",
                    n
                )),
            },
            "TabuSearch" => {
                let tenure = config.req_tabu("Qubo")?;
                match config.req_neighbor("Qubo")? {
                    NeighborKind::Flip => Ok(Box::new(TabuSearch::<QuboFlipNeighbor>::new(
                        cond, tenure, None,
                    ))),
                    NeighborKind::Swap => Ok(Box::new(TabuSearch::<QuboSwapNeighbor>::new(
                        cond, tenure, None,
                    ))),
                    n => Err(format!(
                        "Invalid neighbor {:?} for Qubo (use Flip or Swap)",
                        n
                    )),
                }
            }
            "SimulatedAnnealing" => {
                let temp = config.req_temp("Qubo")?;
                let cooling = config.req_cooling("Qubo")?;
                match config.req_neighbor("Qubo")? {
                    NeighborKind::Flip => Ok(Box::new(
                        SimulatedAnnealing::<QuboFlipNeighbor>::new(cond, temp, cooling),
                    )),
                    NeighborKind::Swap => Ok(Box::new(
                        SimulatedAnnealing::<QuboSwapNeighbor>::new(cond, temp, cooling),
                    )),
                    n => Err(format!(
                        "Invalid neighbor {:?} for Qubo (use Flip or Swap)",
                        n
                    )),
                }
            }
            "LateAcceptanceHillClimbing" => {
                let history_length = config.req_history_length("Qubo")?;
                match config.req_neighbor("Qubo")? {
                    NeighborKind::Flip => Ok(Box::new(LateAcceptanceHillClimbing::<
                        QuboFlipNeighbor,
                    >::new(
                        cond, history_length
                    ))),
                    NeighborKind::Swap => Ok(Box::new(LateAcceptanceHillClimbing::<
                        QuboSwapNeighbor,
                    >::new(
                        cond, history_length
                    ))),
                    n => Err(format!(
                        "Invalid neighbor {:?} for Qubo (use Flip or Swap)",
                        n
                    )),
                }
            }
            "RLSearch" => match config.req_neighbor("Qubo")? {
                NeighborKind::Flip => {
                    build_rl_search_from_config::<Qubo, QuboFlipNeighbor>(config, cond)
                }
                NeighborKind::Swap => {
                    build_rl_search_from_config::<Qubo, QuboSwapNeighbor>(config, cond)
                }
                n => Err(format!(
                    "Invalid neighbor {:?} for Qubo RLSearch (use Flip or Swap)",
                    n
                )),
            },
            k => Err(format!("Unknown kind '{}' for Qubo", k)),
        }
    }

    fn build_crossover(config: &HeuristicConfig) -> Result<Box<dyn Crossover<Self>>, String> {
        match config.crossover_kind.as_deref().unwrap_or("Uniform") {
            "Uniform" => Ok(Box::new(QuboUniformCrossover)),
            other => Err(format!(
                "Unknown crossover_kind '{other}' for Qubo (expected 'Uniform')"
            )),
        }
    }
}

impl BenchmarkableProblem for Sat {
    fn build_base_heuristic(
        config: &HeuristicConfig,
        cond: StopCondition,
    ) -> Result<Box<dyn Heuristic<Self>>, String> {
        match config.kind.as_str() {
            "LocalSearch" => match config.req_neighbor("Sat")? {
                NeighborKind::Flip => Ok(Box::new(LocalSearch::<SatFlipNeighbor>::new(cond))),
                NeighborKind::Swap => Ok(Box::new(LocalSearch::<SatSwapNeighbor>::new(cond))),
                n => Err(format!(
                    "Invalid neighbor {:?} for Sat (use Flip or Swap)",
                    n
                )),
            },
            "TabuSearch" => {
                let tenure = config.req_tabu("Sat")?;
                match config.req_neighbor("Sat")? {
                    NeighborKind::Flip => Ok(Box::new(TabuSearch::<SatFlipNeighbor>::new(
                        cond, tenure, None,
                    ))),
                    NeighborKind::Swap => Ok(Box::new(TabuSearch::<SatSwapNeighbor>::new(
                        cond, tenure, None,
                    ))),
                    n => Err(format!(
                        "Invalid neighbor {:?} for Sat (use Flip or Swap)",
                        n
                    )),
                }
            }
            "SimulatedAnnealing" => {
                let temp = config.req_temp("Sat")?;
                let cooling = config.req_cooling("Sat")?;
                match config.req_neighbor("Sat")? {
                    NeighborKind::Flip => Ok(Box::new(SimulatedAnnealing::<SatFlipNeighbor>::new(
                        cond, temp, cooling,
                    ))),
                    NeighborKind::Swap => Ok(Box::new(SimulatedAnnealing::<SatSwapNeighbor>::new(
                        cond, temp, cooling,
                    ))),
                    n => Err(format!(
                        "Invalid neighbor {:?} for Sat (use Flip or Swap)",
                        n
                    )),
                }
            }
            "LateAcceptanceHillClimbing" => {
                let history_length = config.req_history_length("Sat")?;
                match config.req_neighbor("Sat")? {
                    NeighborKind::Flip => Ok(Box::new(
                        LateAcceptanceHillClimbing::<SatFlipNeighbor>::new(cond, history_length),
                    )),
                    NeighborKind::Swap => Ok(Box::new(
                        LateAcceptanceHillClimbing::<SatSwapNeighbor>::new(cond, history_length),
                    )),
                    n => Err(format!(
                        "Invalid neighbor {:?} for Sat (use Flip or Swap)",
                        n
                    )),
                }
            }
            "RLSearch" => match config.req_neighbor("Sat")? {
                NeighborKind::Flip => {
                    build_rl_search_from_config::<Sat, SatFlipNeighbor>(config, cond)
                }
                NeighborKind::Swap => {
                    build_rl_search_from_config::<Sat, SatSwapNeighbor>(config, cond)
                }
                n => Err(format!(
                    "Invalid neighbor {:?} for Sat RLSearch (use Flip or Swap)",
                    n
                )),
            },
            k => Err(format!("Unknown kind '{}' for Sat", k)),
        }
    }

    fn build_crossover(config: &HeuristicConfig) -> Result<Box<dyn Crossover<Self>>, String> {
        match config.crossover_kind.as_deref().unwrap_or("Uniform") {
            "Uniform" => Ok(Box::new(SatUniformCrossover)),
            other => Err(format!(
                "Unknown crossover_kind '{other}' for Sat (expected 'Uniform')"
            )),
        }
    }
}

impl BenchmarkableProblem for TspWithCoordinates {
    fn build_base_heuristic(
        config: &HeuristicConfig,
        cond: StopCondition,
    ) -> Result<Box<dyn Heuristic<Self>>, String> {
        match config.kind.as_str() {
            "LocalSearch" => match config.req_neighbor("Tsp")? {
                NeighborKind::TwoOpt => Ok(Box::new(LocalSearch::<TspTwoOptNeighbor>::new(cond))),
                NeighborKind::Relocate => {
                    Ok(Box::new(LocalSearch::<TspRelocateNeighbor>::new(cond)))
                }
                n => Err(format!(
                    "Invalid neighbor {:?} for Tsp (use TwoOpt or Relocate)",
                    n
                )),
            },
            "TabuSearch" => {
                let tenure = config.req_tabu("Tsp")?;
                match config.req_neighbor("Tsp")? {
                    NeighborKind::TwoOpt => Ok(Box::new(TabuSearch::<TspTwoOptNeighbor>::new(
                        cond, tenure, None,
                    ))),
                    NeighborKind::Relocate => Ok(Box::new(TabuSearch::<TspRelocateNeighbor>::new(
                        cond, tenure, None,
                    ))),
                    n => Err(format!(
                        "Invalid neighbor {:?} for Tsp (use TwoOpt or Relocate)",
                        n
                    )),
                }
            }
            "SimulatedAnnealing" => {
                let temp = config.req_temp("Tsp")?;
                let cooling = config.req_cooling("Tsp")?;
                match config.req_neighbor("Tsp")? {
                    NeighborKind::TwoOpt => Ok(Box::new(
                        SimulatedAnnealing::<TspTwoOptNeighbor>::new(cond, temp, cooling),
                    )),
                    NeighborKind::Relocate => Ok(Box::new(
                        SimulatedAnnealing::<TspRelocateNeighbor>::new(cond, temp, cooling),
                    )),
                    n => Err(format!(
                        "Invalid neighbor {:?} for Tsp (use TwoOpt or Relocate)",
                        n
                    )),
                }
            }
            "LinKernighanHelsgott" => {
                let num_neighbors = config.num_neighbors.unwrap_or(5);
                let max_depth = config.max_depth.unwrap_or(5);
                Ok(Box::new(LinKernighanHelsgottForTsp::new(
                    cond,
                    num_neighbors,
                    max_depth,
                )))
            }
            "LateAcceptanceHillClimbing" => {
                let history_length = config.req_history_length("Tsp")?;
                match config.req_neighbor("Tsp")? {
                    NeighborKind::TwoOpt => Ok(Box::new(LateAcceptanceHillClimbing::<
                        TspTwoOptNeighbor,
                    >::new(
                        cond, history_length
                    ))),
                    NeighborKind::Relocate => Ok(Box::new(LateAcceptanceHillClimbing::<
                        TspRelocateNeighbor,
                    >::new(
                        cond, history_length
                    ))),
                    n => Err(format!(
                        "Invalid neighbor {:?} for Tsp (use TwoOpt or Relocate)",
                        n
                    )),
                }
            }
            "RLSearch" => match config.req_neighbor("Tsp")? {
                NeighborKind::TwoOpt => build_rl_search_from_config::<
                    TspWithCoordinates,
                    TspTwoOptNeighbor,
                >(config, cond),
                NeighborKind::Relocate => build_rl_search_from_config::<
                    TspWithCoordinates,
                    TspRelocateNeighbor,
                >(config, cond),
                n => Err(format!(
                    "Invalid neighbor {:?} for Tsp RLSearch (use TwoOpt or Relocate)",
                    n
                )),
            },
            k => Err(format!("Unknown kind '{}' for Tsp", k)),
        }
    }

    fn build_crossover(config: &HeuristicConfig) -> Result<Box<dyn Crossover<Self>>, String> {
        match config.crossover_kind.as_deref().unwrap_or("Order") {
            "Order" => Ok(Box::new(TspOrderCrossover)),
            other => Err(format!(
                "Unknown crossover_kind '{other}' for Tsp (expected 'Order')"
            )),
        }
    }
}

impl BenchmarkableProblem for VertexCover {
    fn build_base_heuristic(
        config: &HeuristicConfig,
        cond: StopCondition,
    ) -> Result<Box<dyn Heuristic<Self>>, String> {
        match config.kind.as_str() {
            "LocalSearch" => match config.req_neighbor("VertexCover")? {
                NeighborKind::Flip => {
                    Ok(Box::new(LocalSearch::<VertexCoverFlipNeighbor>::new(cond)))
                }
                NeighborKind::Swap => {
                    Ok(Box::new(LocalSearch::<VertexCoverSwapNeighbor>::new(cond)))
                }
                n => Err(format!(
                    "Invalid neighbor {:?} for VertexCover (use Flip or Swap)",
                    n
                )),
            },
            "TabuSearch" => {
                let tenure = config.req_tabu("VertexCover")?;
                match config.req_neighbor("VertexCover")? {
                    NeighborKind::Flip => Ok(Box::new(TabuSearch::<VertexCoverFlipNeighbor>::new(
                        cond, tenure, None,
                    ))),
                    NeighborKind::Swap => Ok(Box::new(TabuSearch::<VertexCoverSwapNeighbor>::new(
                        cond, tenure, None,
                    ))),
                    n => Err(format!(
                        "Invalid neighbor {:?} for VertexCover (use Flip or Swap)",
                        n
                    )),
                }
            }
            "SimulatedAnnealing" => {
                let temp = config.req_temp("VertexCover")?;
                let cooling = config.req_cooling("VertexCover")?;
                match config.req_neighbor("VertexCover")? {
                    NeighborKind::Flip => Ok(Box::new(
                        SimulatedAnnealing::<VertexCoverFlipNeighbor>::new(cond, temp, cooling),
                    )),
                    NeighborKind::Swap => Ok(Box::new(
                        SimulatedAnnealing::<VertexCoverSwapNeighbor>::new(cond, temp, cooling),
                    )),
                    n => Err(format!(
                        "Invalid neighbor {:?} for VertexCover (use Flip or Swap)",
                        n
                    )),
                }
            }
            k => Err(format!("Unknown kind '{}' for VertexCover", k)),
        }
    }

    fn build_crossover(config: &HeuristicConfig) -> Result<Box<dyn Crossover<Self>>, String> {
        match config.crossover_kind.as_deref().unwrap_or("Uniform") {
            "Uniform" => Ok(Box::new(VertexCoverUniformCrossover)),
            other => Err(format!(
                "Unknown crossover_kind '{other}' for VertexCover (expected 'Uniform')"
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Generic heuristic builder
// ---------------------------------------------------------------------------

/// Builds a `Box<dyn Heuristic<P>>` from a [`HeuristicConfig`].
///
/// Meta-heuristics (Sequential, Iterated, Restart) are handled generically here.
/// Base algorithms are dispatched to [`BenchmarkableProblem::build_base_heuristic`].
fn build_heuristic<P: BenchmarkableProblem + 'static>(
    config: &HeuristicConfig,
) -> Result<Box<dyn Heuristic<P>>, String>
where
    P::Solution: BenchmarkSolution + Distance,
{
    let cond = config.stop_condition.to_stop_condition();
    match config.kind.as_str() {
        "Sequential" => {
            let steps = config
                .steps
                .as_ref()
                .ok_or("'steps' required for Sequential")?;
            let sub: Result<Vec<Box<dyn Heuristic<P>>>, String> =
                steps.iter().map(build_heuristic::<P>).collect();
            Ok(Box::new(Sequential::new(cond, sub?)))
        }
        "Iterated" => {
            let steps = config
                .steps
                .as_ref()
                .ok_or("'steps' required for Iterated")?;
            if steps.len() != 2 {
                return Err(format!(
                    "Iterated requires exactly 2 steps (search + perturbation), but got {}",
                    steps.len()
                ));
            }
            let search = build_heuristic::<P>(&steps[0])?;
            let perturbation = build_heuristic::<P>(&steps[1])?;
            Ok(Box::new(Iterated::new(cond, search, perturbation)))
        }
        "Restart" => {
            let steps = config
                .steps
                .as_ref()
                .ok_or("'steps' required for Restart")?;
            if steps.len() != 1 {
                return Err(format!(
                    "Restart requires exactly 1 step (inner heuristic), but got {}",
                    steps.len()
                ));
            }
            let inner = build_heuristic::<P>(&steps[0])?;
            let rc = config
                .restart_condition
                .as_ref()
                .ok_or("'restart_condition' required for Restart")?;
            Ok(Box::new(Restart::new(cond, inner, rc.to_stop_condition())))
        }
        "GeneticAlgorithm" => {
            let population_size = config
                .population_size
                .ok_or("'population_size' required for GeneticAlgorithm")?;
            let steps = config
                .steps
                .as_ref()
                .ok_or("'steps' required for GeneticAlgorithm (steps[0] = mutation, optional steps[1] = init_improvement)")?;
            if steps.is_empty() || steps.len() > 2 {
                return Err(format!(
                    "GeneticAlgorithm requires 1 or 2 steps (steps[0] = mutation, optional steps[1] = init_improvement), but got {}",
                    steps.len()
                ));
            }
            let mutation = build_heuristic::<P>(&steps[0])?;
            let init_improvement = match steps.get(1) {
                Some(c) => Some(build_heuristic::<P>(c)?),
                None => None,
            };
            let crossover = P::build_crossover(config)?;

            let parent_selection = match config.parent_selection.as_deref().unwrap_or("Tournament")
            {
                "Tournament" => ParentSelection::Tournament,
                "HammingTopK" => {
                    let top_k = config.parent_top_k.ok_or(
                        "'parent_top_k' required for parent_selection = 'HammingTopK'",
                    )?;
                    if top_k == 0 {
                        return Err("'parent_top_k' must be >= 1".to_string());
                    }
                    ParentSelection::HammingTopK { top_k }
                }
                other => {
                    return Err(format!(
                        "Unknown parent_selection '{other}' (expected 'Tournament' or 'HammingTopK')"
                    ));
                }
            };

            let ga = GeneticAlgorithm::new_with_init(
                cond,
                population_size,
                crossover,
                mutation,
                init_improvement,
            )
            .with_parent_selection(parent_selection);
            Ok(Box::new(ga))
        }
        _ => P::build_base_heuristic(config, cond),
    }
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

/// Accumulates benchmark results from multiple runs.
pub struct Benchmark {
    pub results: Vec<BenchmarkResult>,
}

impl Default for Benchmark {
    fn default() -> Self {
        Self::new()
    }
}

impl Benchmark {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Runs a single experiment and appends the result.
    pub fn run(
        &mut self,
        instance_path: &str,
        problem: &ProblemKind,
        heuristic_config: &HeuristicConfig,
    ) {
        let metrics = run_for_problem_kind(problem, heuristic_config, instance_path);
        self.results.push(BenchmarkResult {
            instance_path: instance_path.to_string(),
            problem: problem.clone(),
            heuristic: heuristic_config.clone(),
            status: metrics.status,
            best_objective: metrics.best_objective,
            best_iteration: metrics.best_iteration,
            time_to_best_secs: metrics.time_to_best_secs,
            total_time_secs: metrics.total_time_secs,
            solution: metrics.solution,
        });
    }
}

// ---------------------------------------------------------------------------
// Generic run functions
// ---------------------------------------------------------------------------

struct RunMetrics {
    status: String,
    best_objective: f64,
    best_iteration: u64,
    time_to_best_secs: f64,
    total_time_secs: f64,
    solution: Vec<usize>,
}

fn run_for_problem_kind(
    problem_kind: &ProblemKind,
    config: &HeuristicConfig,
    instance_path: &str,
) -> RunMetrics {
    match problem_kind {
        ProblemKind::MaxCut => run_typed::<MaxCut>(instance_path, config),
        ProblemKind::Qubo => run_typed::<Qubo>(instance_path, config),
        ProblemKind::Sat => run_typed::<Sat>(instance_path, config),
        ProblemKind::Tsp => run_typed::<TspWithCoordinates>(instance_path, config),
        ProblemKind::VertexCover => run_typed::<VertexCover>(instance_path, config),
    }
}

fn run_typed<P: BenchmarkableProblem + 'static>(
    instance_path: &str,
    config: &HeuristicConfig,
) -> RunMetrics
where
    P::Solution: BenchmarkSolution + Distance,
{
    let heuristic = match build_heuristic::<P>(config) {
        Ok(h) => h,
        Err(e) => {
            return RunMetrics {
                status: format!("config error: {}", e),
                best_objective: 0.0,
                best_iteration: 0,
                time_to_best_secs: 0.0,
                total_time_secs: 0.0,
                solution: Vec::new(),
            };
        }
    };
    run_problem::<P>(instance_path, heuristic)
}

fn run_problem<P>(instance_path: &str, mut heuristic: Box<dyn Heuristic<P>>) -> RunMetrics
where
    P: BenchmarkProblem,
    P::Solution: BenchmarkSolution,
{
    let instance = match P::load_instance(instance_path) {
        Ok(v) => v,
        Err(e) => {
            return RunMetrics {
                status: format!("error loading instance: {}", e),
                best_objective: 0.0,
                best_iteration: 0,
                time_to_best_secs: 0.0,
                total_time_secs: 0.0,
                solution: Vec::new(),
            };
        }
    };
    let mut state = SearchState::new(&instance);
    let start = std::time::Instant::now();
    let status = heuristic.run(&mut state);
    let total_time = start.elapsed();
    RunMetrics {
        status: status_str(status),
        best_objective: state.best_solution.best_objective_f64(),
        best_iteration: state.best_iteration,
        time_to_best_secs: (state.best_time - state.start_time).as_secs_f64(),
        total_time_secs: total_time.as_secs_f64(),
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

    Summary {
        num_successful_runs: n,
        best_objective: best,
        avg_objective: avg,
        worst_objective: worst,
        std_objective: std,
        best_time_to_best_secs: best_ttb,
        avg_time_to_best_secs: avg_ttb,
        avg_total_time_secs: avg_total,
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
        solution: m.solution,
    }
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
    /// Instance paths support glob patterns (e.g. `"data/max_cut/G[1-9]*"`).
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
            for heuristic_cfg in &config.heuristics {
                tracing::info!(
                    instance = %instance_path,
                    heuristic = %heuristic_cfg.kind,
                    num_runs = config.num_runs,
                    max_iteration = ?heuristic_cfg.stop_condition.max_iteration,
                    max_duration_secs = ?heuristic_cfg.stop_condition.max_duration_secs,
                    max_failed_update = ?heuristic_cfg.stop_condition.max_failed_update,
                    "Start:"
                );

                let mut runs: Vec<SingleRunResult> = (0..config.num_runs)
                    .into_par_iter()
                    .map(|run_index| {
                        let metrics =
                            run_for_problem_kind(problem_kind, heuristic_cfg, instance_path);

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

                let minimize = matches!(
                    problem_kind,
                    ProblemKind::Qubo | ProblemKind::Tsp | ProblemKind::VertexCover
                );
                let summary = compute_summary(&runs, minimize);
                tracing::info!(
                    instance = %instance_path,
                    heuristic = %heuristic_cfg.kind,
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
