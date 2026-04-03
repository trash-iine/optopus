//! Benchmark runner for comparing heuristics across problem instances.
//!
//! [`Benchmark`] accumulates [`BenchmarkResult`] records produced by [`Benchmark::run`].
//! Each result captures the configuration, best objective value, time-to-best, and solution.
//! Results can be serialized to TOML (or any serde format) for offline analysis.

use crate::{
    error::OptError,
    heuristic::{
        BreakoutLocalSearchForMaxCut, Heuristic, Iterated, LocalSearch, Restart, Sequential,
        SimulatedAnnealing, StopCondition, TabuSearch,
    },
    problem::{
        MaxCutFlipNeighbor, MaxCutSolution, MaxCutSwapNeighbor, QuboFlipNeighbor, QuboSwapNeighbor,
        max_cut::MaxCut,
        qubo::{Qubo, QuboSolution},
        sat::{Sat, SatFlipNeighbor, SatSolution, SatSwapNeighbor},
        tsp_2d::{TspRelocateNeighbor, TspSolution, TspTwoOptNeighbor, TspWithCoordinates},
    },
    search_state::{ProblemTrait, SearchState},
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
        MaxCut::load_from_file(path).map_err(|e| OptError::Parse(e.to_string()))
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
        Qubo::load_file_as_max_cut(path).map_err(|e| OptError::Parse(e.to_string()))
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
        Sat::load_file(path).map_err(|e| OptError::Parse(e.to_string()))
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
        TspWithCoordinates::load_file(path).map_err(|e| OptError::Parse(e.to_string()))
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
/// - `"LocalSearch"`, `"TabuSearch"`, `"SimulatedAnnealing"`, `"BreakoutLocalSearch"` (MaxCut only)
/// - `"Sequential"` — repeats its `steps` cycle until `stop_condition` is met
/// - `"Iterated"` — `steps\[0\]` = search phase, `steps\[1\]` = perturbation phase (ILS)
/// - `"Restart"` — runs `steps\[0\]` then resets to a new random solution when `restart_condition` is met
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
    #[serde(default)]
    pub stop_condition: StopConditionConfig,
    /// Sub-heuristics for `"Sequential"`, `"Iterated"`, and `"Restart"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps: Option<Vec<HeuristicConfig>>,
    /// Restart trigger for `kind = "Restart"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub restart_condition: Option<StopConditionConfig>,
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
// Common algorithm parameter structs
// ---------------------------------------------------------------------------

/// Parameters for simulated annealing.
#[derive(Clone, Serialize)]
pub struct SimulatedAnnealingSetting {
    pub initial_temperature: f64,
    pub cooling_rate: f64,
}

/// Parameters for tabu search.
#[derive(Clone, Serialize)]
pub struct TabuSearchSetting {
    pub tabu_tenure: (u64, u64),
}

/// Parameters for Breakout Local Search (MaxCut only).
#[derive(Clone, Serialize)]
pub struct BreakoutLocalSearchSetting {
    pub tabu_tenure: (u64, u64),
    pub t: u64,
    pub l0: u64,
    pub p0: f64,
    pub q: f64,
}

// ---------------------------------------------------------------------------
// MaxCut
// ---------------------------------------------------------------------------

/// Neighborhood type for MaxCut benchmarks.
#[derive(Clone, Serialize)]
pub enum MaxCutNeighborKind {
    Flip,
    Swap,
}

/// A single step in a [`Sequential`] meta-heuristic, parameterised by the heuristic setting type.
#[derive(Clone, Serialize)]
pub struct SequentialStep<S> {
    pub heuristic: S,
    pub stop_condition: StopCondition,
}

pub type MaxCutSequentialStep = SequentialStep<MaxCutHeuristicSetting>;

/// Heuristic configuration for MaxCut benchmarks.
#[derive(Clone, Serialize)]
pub enum MaxCutHeuristicSetting {
    LocalSearch(MaxCutNeighborKind),
    TabuSearch(TabuSearchSetting, MaxCutNeighborKind),
    SimulatedAnnealing(SimulatedAnnealingSetting, MaxCutNeighborKind),
    BreakoutLocalSearch(BreakoutLocalSearchSetting),
    /// Run each step in sequence, repeating the cycle until `cond` is satisfied.
    Sequential(Vec<MaxCutSequentialStep>),
    /// ILS: `steps[0]` = search phase, `steps[1]` = perturbation phase.
    Iterated(Vec<MaxCutSequentialStep>),
    /// Restart from a new random solution when `restart_condition` is met.
    Restart {
        inner: Box<MaxCutSequentialStep>,
        restart_condition: StopCondition,
    },
}

impl MaxCutHeuristicSetting {
    fn build(&self, cond: StopCondition) -> Box<dyn Heuristic<MaxCut>> {
        match self {
            Self::LocalSearch(MaxCutNeighborKind::Flip) => {
                Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(cond))
            }
            Self::LocalSearch(MaxCutNeighborKind::Swap) => {
                Box::new(LocalSearch::<MaxCutSwapNeighbor>::new(cond))
            }
            Self::TabuSearch(s, MaxCutNeighborKind::Flip) => Box::new(TabuSearch::<
                MaxCutFlipNeighbor,
            >::new(
                cond, s.tabu_tenure, None
            )),
            Self::TabuSearch(s, MaxCutNeighborKind::Swap) => Box::new(TabuSearch::<
                MaxCutSwapNeighbor,
            >::new(
                cond, s.tabu_tenure, None
            )),
            Self::SimulatedAnnealing(s, MaxCutNeighborKind::Flip) => {
                Box::new(SimulatedAnnealing::<MaxCutFlipNeighbor>::new(
                    cond,
                    s.initial_temperature,
                    s.cooling_rate,
                ))
            }
            Self::SimulatedAnnealing(s, MaxCutNeighborKind::Swap) => {
                Box::new(SimulatedAnnealing::<MaxCutSwapNeighbor>::new(
                    cond,
                    s.initial_temperature,
                    s.cooling_rate,
                ))
            }
            Self::BreakoutLocalSearch(s) => Box::new(BreakoutLocalSearchForMaxCut::new(
                s.tabu_tenure,
                cond,
                s.t,
                s.l0,
                s.p0,
                s.q,
            )),
            Self::Sequential(steps) => {
                let sub: Vec<Box<dyn Heuristic<MaxCut>>> = steps
                    .iter()
                    .map(|s| s.heuristic.build(s.stop_condition.clone()))
                    .collect();
                Box::new(Sequential::new(cond, sub))
            }
            Self::Iterated(steps) => {
                assert_eq!(
                    steps.len(),
                    2,
                    "Iterated requires exactly 2 steps (search, perturbation)"
                );
                let search = steps[0].heuristic.build(steps[0].stop_condition.clone());
                let perturbation = steps[1].heuristic.build(steps[1].stop_condition.clone());
                Box::new(Iterated::new(cond, search, perturbation))
            }
            Self::Restart {
                inner,
                restart_condition,
            } => {
                let heuristic = inner.heuristic.build(inner.stop_condition.clone());
                Box::new(Restart::new(cond, heuristic, restart_condition.clone()))
            }
        }
    }
}

/// Full benchmark configuration for a single run, parameterised by the heuristic setting type.
#[derive(Clone, Serialize)]
pub struct GenericBenchmarkSetting<H> {
    pub instance_path: String,
    pub heuristic: H,
    pub stop_condition: StopCondition,
}

pub type MaxCutBenchmarkSetting = GenericBenchmarkSetting<MaxCutHeuristicSetting>;

// ---------------------------------------------------------------------------
// QUBO
// ---------------------------------------------------------------------------

/// Neighborhood type for QUBO benchmarks.
#[derive(Clone, Serialize)]
pub enum QuboNeighborKind {
    Flip,
    Swap,
}

pub type QuboSequentialStep = SequentialStep<QuboHeuristicSetting>;

/// Heuristic configuration for QUBO benchmarks.
#[derive(Clone, Serialize)]
pub enum QuboHeuristicSetting {
    LocalSearch(QuboNeighborKind),
    TabuSearch(TabuSearchSetting, QuboNeighborKind),
    SimulatedAnnealing(SimulatedAnnealingSetting, QuboNeighborKind),
    Sequential(Vec<QuboSequentialStep>),
    Iterated(Vec<QuboSequentialStep>),
    Restart {
        inner: Box<QuboSequentialStep>,
        restart_condition: StopCondition,
    },
}

impl QuboHeuristicSetting {
    fn build(&self, cond: StopCondition) -> Box<dyn Heuristic<Qubo>> {
        match self {
            Self::LocalSearch(QuboNeighborKind::Flip) => {
                Box::new(LocalSearch::<QuboFlipNeighbor>::new(cond))
            }
            Self::LocalSearch(QuboNeighborKind::Swap) => {
                Box::new(LocalSearch::<QuboSwapNeighbor>::new(cond))
            }
            Self::TabuSearch(s, QuboNeighborKind::Flip) => Box::new(
                TabuSearch::<QuboFlipNeighbor>::new(cond, s.tabu_tenure, None),
            ),
            Self::TabuSearch(s, QuboNeighborKind::Swap) => Box::new(
                TabuSearch::<QuboSwapNeighbor>::new(cond, s.tabu_tenure, None),
            ),
            Self::SimulatedAnnealing(s, QuboNeighborKind::Flip) => {
                Box::new(SimulatedAnnealing::<QuboFlipNeighbor>::new(
                    cond,
                    s.initial_temperature,
                    s.cooling_rate,
                ))
            }
            Self::SimulatedAnnealing(s, QuboNeighborKind::Swap) => {
                Box::new(SimulatedAnnealing::<QuboSwapNeighbor>::new(
                    cond,
                    s.initial_temperature,
                    s.cooling_rate,
                ))
            }
            Self::Sequential(steps) => {
                let sub: Vec<Box<dyn Heuristic<Qubo>>> = steps
                    .iter()
                    .map(|s| s.heuristic.build(s.stop_condition.clone()))
                    .collect();
                Box::new(Sequential::new(cond, sub))
            }
            Self::Iterated(steps) => {
                assert_eq!(steps.len(), 2, "Iterated requires exactly 2 steps");
                let search = steps[0].heuristic.build(steps[0].stop_condition.clone());
                let perturbation = steps[1].heuristic.build(steps[1].stop_condition.clone());
                Box::new(Iterated::new(cond, search, perturbation))
            }
            Self::Restart {
                inner,
                restart_condition,
            } => {
                let heuristic = inner.heuristic.build(inner.stop_condition.clone());
                Box::new(Restart::new(cond, heuristic, restart_condition.clone()))
            }
        }
    }
}

pub type QuboBenchmarkSetting = GenericBenchmarkSetting<QuboHeuristicSetting>;

// ---------------------------------------------------------------------------
// SAT
// ---------------------------------------------------------------------------

/// Neighborhood type for SAT benchmarks.
#[derive(Clone, Serialize)]
pub enum SatNeighborKind {
    Flip,
    Swap,
}

pub type SatSequentialStep = SequentialStep<SatHeuristicSetting>;

/// Heuristic configuration for SAT benchmarks.
#[derive(Clone, Serialize)]
pub enum SatHeuristicSetting {
    LocalSearch(SatNeighborKind),
    TabuSearch(TabuSearchSetting, SatNeighborKind),
    SimulatedAnnealing(SimulatedAnnealingSetting, SatNeighborKind),
    Sequential(Vec<SatSequentialStep>),
    Iterated(Vec<SatSequentialStep>),
    Restart {
        inner: Box<SatSequentialStep>,
        restart_condition: StopCondition,
    },
}

impl SatHeuristicSetting {
    fn build(&self, cond: StopCondition) -> Box<dyn Heuristic<Sat>> {
        match self {
            Self::LocalSearch(SatNeighborKind::Flip) => {
                Box::new(LocalSearch::<SatFlipNeighbor>::new(cond))
            }
            Self::LocalSearch(SatNeighborKind::Swap) => {
                Box::new(LocalSearch::<SatSwapNeighbor>::new(cond))
            }
            Self::TabuSearch(s, SatNeighborKind::Flip) => Box::new(
                TabuSearch::<SatFlipNeighbor>::new(cond, s.tabu_tenure, None),
            ),
            Self::TabuSearch(s, SatNeighborKind::Swap) => Box::new(
                TabuSearch::<SatSwapNeighbor>::new(cond, s.tabu_tenure, None),
            ),
            Self::SimulatedAnnealing(s, SatNeighborKind::Flip) => {
                Box::new(SimulatedAnnealing::<SatFlipNeighbor>::new(
                    cond,
                    s.initial_temperature,
                    s.cooling_rate,
                ))
            }
            Self::SimulatedAnnealing(s, SatNeighborKind::Swap) => {
                Box::new(SimulatedAnnealing::<SatSwapNeighbor>::new(
                    cond,
                    s.initial_temperature,
                    s.cooling_rate,
                ))
            }
            Self::Sequential(steps) => {
                let sub: Vec<Box<dyn Heuristic<Sat>>> = steps
                    .iter()
                    .map(|s| s.heuristic.build(s.stop_condition.clone()))
                    .collect();
                Box::new(Sequential::new(cond, sub))
            }
            Self::Iterated(steps) => {
                assert_eq!(steps.len(), 2, "Iterated requires exactly 2 steps");
                let search = steps[0].heuristic.build(steps[0].stop_condition.clone());
                let perturbation = steps[1].heuristic.build(steps[1].stop_condition.clone());
                Box::new(Iterated::new(cond, search, perturbation))
            }
            Self::Restart {
                inner,
                restart_condition,
            } => {
                let heuristic = inner.heuristic.build(inner.stop_condition.clone());
                Box::new(Restart::new(cond, heuristic, restart_condition.clone()))
            }
        }
    }
}

pub type SatBenchmarkSetting = GenericBenchmarkSetting<SatHeuristicSetting>;

// ---------------------------------------------------------------------------
// TSP
// ---------------------------------------------------------------------------

/// Neighborhood type for TSP benchmarks.
#[derive(Clone, Serialize)]
pub enum TspNeighborKind {
    TwoOpt,
    Relocate,
}

pub type TspSequentialStep = SequentialStep<TspHeuristicSetting>;

/// Heuristic configuration for TSP benchmarks.
#[derive(Clone, Serialize)]
pub enum TspHeuristicSetting {
    LocalSearch(TspNeighborKind),
    TabuSearch(TabuSearchSetting, TspNeighborKind),
    SimulatedAnnealing(SimulatedAnnealingSetting, TspNeighborKind),
    Sequential(Vec<TspSequentialStep>),
    Iterated(Vec<TspSequentialStep>),
    Restart {
        inner: Box<TspSequentialStep>,
        restart_condition: StopCondition,
    },
}

impl TspHeuristicSetting {
    fn build(&self, cond: StopCondition) -> Box<dyn Heuristic<TspWithCoordinates>> {
        match self {
            Self::LocalSearch(TspNeighborKind::TwoOpt) => {
                Box::new(LocalSearch::<TspTwoOptNeighbor>::new(cond))
            }
            Self::LocalSearch(TspNeighborKind::Relocate) => {
                Box::new(LocalSearch::<TspRelocateNeighbor>::new(cond))
            }
            Self::TabuSearch(s, TspNeighborKind::TwoOpt) => Box::new(
                TabuSearch::<TspTwoOptNeighbor>::new(cond, s.tabu_tenure, None),
            ),
            Self::TabuSearch(s, TspNeighborKind::Relocate) => Box::new(TabuSearch::<
                TspRelocateNeighbor,
            >::new(
                cond, s.tabu_tenure, None
            )),
            Self::SimulatedAnnealing(s, TspNeighborKind::TwoOpt) => {
                Box::new(SimulatedAnnealing::<TspTwoOptNeighbor>::new(
                    cond,
                    s.initial_temperature,
                    s.cooling_rate,
                ))
            }
            Self::SimulatedAnnealing(s, TspNeighborKind::Relocate) => {
                Box::new(SimulatedAnnealing::<TspRelocateNeighbor>::new(
                    cond,
                    s.initial_temperature,
                    s.cooling_rate,
                ))
            }
            Self::Sequential(steps) => {
                let sub: Vec<Box<dyn Heuristic<TspWithCoordinates>>> = steps
                    .iter()
                    .map(|s| s.heuristic.build(s.stop_condition.clone()))
                    .collect();
                Box::new(Sequential::new(cond, sub))
            }
            Self::Iterated(steps) => {
                assert_eq!(steps.len(), 2, "Iterated requires exactly 2 steps");
                let search = steps[0].heuristic.build(steps[0].stop_condition.clone());
                let perturbation = steps[1].heuristic.build(steps[1].stop_condition.clone());
                Box::new(Iterated::new(cond, search, perturbation))
            }
            Self::Restart {
                inner,
                restart_condition,
            } => {
                let heuristic = inner.heuristic.build(inner.stop_condition.clone());
                Box::new(Restart::new(cond, heuristic, restart_condition.clone()))
            }
        }
    }
}

pub type TspBenchmarkSetting = GenericBenchmarkSetting<TspHeuristicSetting>;

// ---------------------------------------------------------------------------
// Master BenchmarkSetting
// ---------------------------------------------------------------------------

/// All information needed to reproduce a single experiment run.
#[derive(Clone, Serialize)]
pub enum BenchmarkSetting {
    MaxCut(MaxCutBenchmarkSetting),
    Qubo(QuboBenchmarkSetting),
    Sat(SatBenchmarkSetting),
    Tsp(TspBenchmarkSetting),
}

// ---------------------------------------------------------------------------
// HeuristicConfig → BenchmarkSetting conversion
// ---------------------------------------------------------------------------

impl HeuristicConfig {
    /// Converts this config entry into a `BenchmarkSetting` for the given instance.
    ///
    /// `problem` is supplied by the caller (from `InstanceConfig`) rather than stored
    /// in the heuristic config itself to avoid duplication.
    pub fn to_benchmark_setting(
        &self,
        instance_path: &str,
        problem: &ProblemKind,
    ) -> Result<BenchmarkSetting, String> {
        let stop_condition = self.stop_condition.to_stop_condition();
        match problem {
            ProblemKind::MaxCut => Ok(BenchmarkSetting::MaxCut(MaxCutBenchmarkSetting {
                instance_path: instance_path.to_string(),
                heuristic: self.to_max_cut_heuristic_setting()?,
                stop_condition,
            })),
            ProblemKind::Qubo => Ok(BenchmarkSetting::Qubo(QuboBenchmarkSetting {
                instance_path: instance_path.to_string(),
                heuristic: self.to_qubo_heuristic_setting()?,
                stop_condition,
            })),
            ProblemKind::Sat => Ok(BenchmarkSetting::Sat(SatBenchmarkSetting {
                instance_path: instance_path.to_string(),
                heuristic: self.to_sat_heuristic_setting()?,
                stop_condition,
            })),
            ProblemKind::Tsp => Ok(BenchmarkSetting::Tsp(TspBenchmarkSetting {
                instance_path: instance_path.to_string(),
                heuristic: self.to_tsp_heuristic_setting()?,
                stop_condition,
            })),
        }
    }

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

    fn to_max_cut_heuristic_setting(&self) -> Result<MaxCutHeuristicSetting, String> {
        match self.kind.as_str() {
            "LocalSearch" => Ok(MaxCutHeuristicSetting::LocalSearch(to_max_cut_neighbor(
                self.req_neighbor("MaxCut")?,
            )?)),
            "TabuSearch" => Ok(MaxCutHeuristicSetting::TabuSearch(
                TabuSearchSetting {
                    tabu_tenure: self.req_tabu("MaxCut")?,
                },
                to_max_cut_neighbor(self.req_neighbor("MaxCut")?)?,
            )),
            "SimulatedAnnealing" => Ok(MaxCutHeuristicSetting::SimulatedAnnealing(
                SimulatedAnnealingSetting {
                    initial_temperature: self.req_temp("MaxCut")?,
                    cooling_rate: self.req_cooling("MaxCut")?,
                },
                to_max_cut_neighbor(self.req_neighbor("MaxCut")?)?,
            )),
            "BreakoutLocalSearch" => Ok(MaxCutHeuristicSetting::BreakoutLocalSearch(
                BreakoutLocalSearchSetting {
                    tabu_tenure: self.req_tabu("MaxCut")?,
                    t: self
                        .t
                        .ok_or("'t' required for MaxCut BreakoutLocalSearch")?,
                    l0: self
                        .l0
                        .ok_or("'l0' required for MaxCut BreakoutLocalSearch")?,
                    p0: self
                        .p0
                        .ok_or("'p0' required for MaxCut BreakoutLocalSearch")?,
                    q: self
                        .q
                        .ok_or("'q' required for MaxCut BreakoutLocalSearch")?,
                },
            )),
            "Sequential" => {
                let steps = self
                    .steps
                    .as_ref()
                    .ok_or("'steps' required for Sequential")?;
                let converted: Result<Vec<MaxCutSequentialStep>, String> = steps
                    .iter()
                    .map(|s| {
                        Ok(MaxCutSequentialStep {
                            heuristic: s.to_max_cut_heuristic_setting()?,
                            stop_condition: s.stop_condition.to_stop_condition(),
                        })
                    })
                    .collect();
                Ok(MaxCutHeuristicSetting::Sequential(converted?))
            }
            "Iterated" => {
                let steps = self.steps.as_ref().ok_or("'steps' required for Iterated")?;
                if steps.len() != 2 {
                    return Err("Iterated requires exactly 2 steps".into());
                }
                let converted: Result<Vec<MaxCutSequentialStep>, String> = steps
                    .iter()
                    .map(|s| {
                        Ok(MaxCutSequentialStep {
                            heuristic: s.to_max_cut_heuristic_setting()?,
                            stop_condition: s.stop_condition.to_stop_condition(),
                        })
                    })
                    .collect();
                Ok(MaxCutHeuristicSetting::Iterated(converted?))
            }
            "Restart" => {
                let steps = self.steps.as_ref().ok_or("'steps' required for Restart")?;
                if steps.len() != 1 {
                    return Err("Restart requires exactly 1 step".into());
                }
                let rc = self
                    .restart_condition
                    .as_ref()
                    .ok_or("'restart_condition' required for Restart")?;
                Ok(MaxCutHeuristicSetting::Restart {
                    inner: Box::new(MaxCutSequentialStep {
                        heuristic: steps[0].to_max_cut_heuristic_setting()?,
                        stop_condition: steps[0].stop_condition.to_stop_condition(),
                    }),
                    restart_condition: rc.to_stop_condition(),
                })
            }
            k => Err(format!("Unknown kind '{}' for MaxCut", k)),
        }
    }

    fn to_qubo_heuristic_setting(&self) -> Result<QuboHeuristicSetting, String> {
        match self.kind.as_str() {
            "LocalSearch" => Ok(QuboHeuristicSetting::LocalSearch(to_qubo_neighbor(
                self.req_neighbor("Qubo")?,
            )?)),
            "TabuSearch" => Ok(QuboHeuristicSetting::TabuSearch(
                TabuSearchSetting {
                    tabu_tenure: self.req_tabu("Qubo")?,
                },
                to_qubo_neighbor(self.req_neighbor("Qubo")?)?,
            )),
            "SimulatedAnnealing" => Ok(QuboHeuristicSetting::SimulatedAnnealing(
                SimulatedAnnealingSetting {
                    initial_temperature: self.req_temp("Qubo")?,
                    cooling_rate: self.req_cooling("Qubo")?,
                },
                to_qubo_neighbor(self.req_neighbor("Qubo")?)?,
            )),
            "Sequential" => {
                let steps = self
                    .steps
                    .as_ref()
                    .ok_or("'steps' required for Sequential")?;
                let converted: Result<Vec<QuboSequentialStep>, String> = steps
                    .iter()
                    .map(|s| {
                        Ok(QuboSequentialStep {
                            heuristic: s.to_qubo_heuristic_setting()?,
                            stop_condition: s.stop_condition.to_stop_condition(),
                        })
                    })
                    .collect();
                Ok(QuboHeuristicSetting::Sequential(converted?))
            }
            "Iterated" => {
                let steps = self.steps.as_ref().ok_or("'steps' required for Iterated")?;
                if steps.len() != 2 {
                    return Err("Iterated requires exactly 2 steps".into());
                }
                let converted: Result<Vec<QuboSequentialStep>, String> = steps
                    .iter()
                    .map(|s| {
                        Ok(QuboSequentialStep {
                            heuristic: s.to_qubo_heuristic_setting()?,
                            stop_condition: s.stop_condition.to_stop_condition(),
                        })
                    })
                    .collect();
                Ok(QuboHeuristicSetting::Iterated(converted?))
            }
            "Restart" => {
                let steps = self.steps.as_ref().ok_or("'steps' required for Restart")?;
                if steps.len() != 1 {
                    return Err("Restart requires exactly 1 step".into());
                }
                let rc = self
                    .restart_condition
                    .as_ref()
                    .ok_or("'restart_condition' required for Restart")?;
                Ok(QuboHeuristicSetting::Restart {
                    inner: Box::new(QuboSequentialStep {
                        heuristic: steps[0].to_qubo_heuristic_setting()?,
                        stop_condition: steps[0].stop_condition.to_stop_condition(),
                    }),
                    restart_condition: rc.to_stop_condition(),
                })
            }
            k => Err(format!("Unknown kind '{}' for Qubo", k)),
        }
    }

    fn to_sat_heuristic_setting(&self) -> Result<SatHeuristicSetting, String> {
        match self.kind.as_str() {
            "LocalSearch" => Ok(SatHeuristicSetting::LocalSearch(to_sat_neighbor(
                self.req_neighbor("Sat")?,
            )?)),
            "TabuSearch" => Ok(SatHeuristicSetting::TabuSearch(
                TabuSearchSetting {
                    tabu_tenure: self.req_tabu("Sat")?,
                },
                to_sat_neighbor(self.req_neighbor("Sat")?)?,
            )),
            "SimulatedAnnealing" => Ok(SatHeuristicSetting::SimulatedAnnealing(
                SimulatedAnnealingSetting {
                    initial_temperature: self.req_temp("Sat")?,
                    cooling_rate: self.req_cooling("Sat")?,
                },
                to_sat_neighbor(self.req_neighbor("Sat")?)?,
            )),
            "Sequential" => {
                let steps = self
                    .steps
                    .as_ref()
                    .ok_or("'steps' required for Sequential")?;
                let converted: Result<Vec<SatSequentialStep>, String> = steps
                    .iter()
                    .map(|s| {
                        Ok(SatSequentialStep {
                            heuristic: s.to_sat_heuristic_setting()?,
                            stop_condition: s.stop_condition.to_stop_condition(),
                        })
                    })
                    .collect();
                Ok(SatHeuristicSetting::Sequential(converted?))
            }
            "Iterated" => {
                let steps = self.steps.as_ref().ok_or("'steps' required for Iterated")?;
                if steps.len() != 2 {
                    return Err("Iterated requires exactly 2 steps".into());
                }
                let converted: Result<Vec<SatSequentialStep>, String> = steps
                    .iter()
                    .map(|s| {
                        Ok(SatSequentialStep {
                            heuristic: s.to_sat_heuristic_setting()?,
                            stop_condition: s.stop_condition.to_stop_condition(),
                        })
                    })
                    .collect();
                Ok(SatHeuristicSetting::Iterated(converted?))
            }
            "Restart" => {
                let steps = self.steps.as_ref().ok_or("'steps' required for Restart")?;
                if steps.len() != 1 {
                    return Err("Restart requires exactly 1 step".into());
                }
                let rc = self
                    .restart_condition
                    .as_ref()
                    .ok_or("'restart_condition' required for Restart")?;
                Ok(SatHeuristicSetting::Restart {
                    inner: Box::new(SatSequentialStep {
                        heuristic: steps[0].to_sat_heuristic_setting()?,
                        stop_condition: steps[0].stop_condition.to_stop_condition(),
                    }),
                    restart_condition: rc.to_stop_condition(),
                })
            }
            k => Err(format!("Unknown kind '{}' for Sat", k)),
        }
    }

    fn to_tsp_heuristic_setting(&self) -> Result<TspHeuristicSetting, String> {
        match self.kind.as_str() {
            "LocalSearch" => Ok(TspHeuristicSetting::LocalSearch(to_tsp_neighbor(
                self.req_neighbor("Tsp")?,
            )?)),
            "TabuSearch" => Ok(TspHeuristicSetting::TabuSearch(
                TabuSearchSetting {
                    tabu_tenure: self.req_tabu("Tsp")?,
                },
                to_tsp_neighbor(self.req_neighbor("Tsp")?)?,
            )),
            "SimulatedAnnealing" => Ok(TspHeuristicSetting::SimulatedAnnealing(
                SimulatedAnnealingSetting {
                    initial_temperature: self.req_temp("Tsp")?,
                    cooling_rate: self.req_cooling("Tsp")?,
                },
                to_tsp_neighbor(self.req_neighbor("Tsp")?)?,
            )),
            "Sequential" => {
                let steps = self
                    .steps
                    .as_ref()
                    .ok_or("'steps' required for Sequential")?;
                let converted: Result<Vec<TspSequentialStep>, String> = steps
                    .iter()
                    .map(|s| {
                        Ok(TspSequentialStep {
                            heuristic: s.to_tsp_heuristic_setting()?,
                            stop_condition: s.stop_condition.to_stop_condition(),
                        })
                    })
                    .collect();
                Ok(TspHeuristicSetting::Sequential(converted?))
            }
            "Iterated" => {
                let steps = self.steps.as_ref().ok_or("'steps' required for Iterated")?;
                if steps.len() != 2 {
                    return Err("Iterated requires exactly 2 steps".into());
                }
                let converted: Result<Vec<TspSequentialStep>, String> = steps
                    .iter()
                    .map(|s| {
                        Ok(TspSequentialStep {
                            heuristic: s.to_tsp_heuristic_setting()?,
                            stop_condition: s.stop_condition.to_stop_condition(),
                        })
                    })
                    .collect();
                Ok(TspHeuristicSetting::Iterated(converted?))
            }
            "Restart" => {
                let steps = self.steps.as_ref().ok_or("'steps' required for Restart")?;
                if steps.len() != 1 {
                    return Err("Restart requires exactly 1 step".into());
                }
                let rc = self
                    .restart_condition
                    .as_ref()
                    .ok_or("'restart_condition' required for Restart")?;
                Ok(TspHeuristicSetting::Restart {
                    inner: Box::new(TspSequentialStep {
                        heuristic: steps[0].to_tsp_heuristic_setting()?,
                        stop_condition: steps[0].stop_condition.to_stop_condition(),
                    }),
                    restart_condition: rc.to_stop_condition(),
                })
            }
            k => Err(format!("Unknown kind '{}' for Tsp", k)),
        }
    }
}

fn to_max_cut_neighbor(n: &NeighborKind) -> Result<MaxCutNeighborKind, String> {
    match n {
        NeighborKind::Flip => Ok(MaxCutNeighborKind::Flip),
        NeighborKind::Swap => Ok(MaxCutNeighborKind::Swap),
        _ => Err(format!(
            "Invalid neighbor {:?} for MaxCut (use Flip or Swap)",
            n
        )),
    }
}

fn to_qubo_neighbor(n: &NeighborKind) -> Result<QuboNeighborKind, String> {
    match n {
        NeighborKind::Flip => Ok(QuboNeighborKind::Flip),
        NeighborKind::Swap => Ok(QuboNeighborKind::Swap),
        _ => Err(format!(
            "Invalid neighbor {:?} for Qubo (use Flip or Swap)",
            n
        )),
    }
}

fn to_sat_neighbor(n: &NeighborKind) -> Result<SatNeighborKind, String> {
    match n {
        NeighborKind::Flip => Ok(SatNeighborKind::Flip),
        NeighborKind::Swap => Ok(SatNeighborKind::Swap),
        _ => Err(format!(
            "Invalid neighbor {:?} for Sat (use Flip or Swap)",
            n
        )),
    }
}

fn to_tsp_neighbor(n: &NeighborKind) -> Result<TspNeighborKind, String> {
    match n {
        NeighborKind::TwoOpt => Ok(TspNeighborKind::TwoOpt),
        NeighborKind::Relocate => Ok(TspNeighborKind::Relocate),
        _ => Err(format!(
            "Invalid neighbor {:?} for Tsp (use TwoOpt or Relocate)",
            n
        )),
    }
}

// ---------------------------------------------------------------------------
// BenchmarkResult
// ---------------------------------------------------------------------------

/// The result of a single experiment run (configuration + metrics).
#[derive(Serialize)]
pub struct BenchmarkResult {
    /// Configuration required to reproduce this run (instance, heuristic, stopping condition).
    pub setting: BenchmarkSetting,
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
// Benchmark runner
// ---------------------------------------------------------------------------

/// Accumulates benchmark results from multiple runs.
pub struct Benchmark {
    pub results: Vec<BenchmarkResult>,
}

impl Benchmark {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Runs a single experiment defined by `setting` and appends the result.
    pub fn run(&mut self, setting: BenchmarkSetting) {
        let metrics = match &setting {
            BenchmarkSetting::MaxCut(s) => run_problem::<MaxCut>(
                &s.instance_path,
                s.heuristic.build(s.stop_condition.clone()),
            ),
            BenchmarkSetting::Qubo(s) => run_problem::<Qubo>(
                &s.instance_path,
                s.heuristic.build(s.stop_condition.clone()),
            ),
            BenchmarkSetting::Sat(s) => run_problem::<Sat>(
                &s.instance_path,
                s.heuristic.build(s.stop_condition.clone()),
            ),
            BenchmarkSetting::Tsp(s) => run_problem::<TspWithCoordinates>(
                &s.instance_path,
                s.heuristic.build(s.stop_condition.clone()),
            ),
        };
        self.results.push(BenchmarkResult {
            setting,
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
// Generic run function
// ---------------------------------------------------------------------------

struct RunMetrics {
    status: String,
    best_objective: f64,
    best_iteration: u64,
    time_to_best_secs: f64,
    total_time_secs: f64,
    solution: Vec<usize>,
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
    /// Runs all (instance × heuristic) combinations from a `BenchmarkConfig` and returns a report.
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
                let mut runs: Vec<SingleRunResult> = Vec::new();

                for run_index in 0..config.num_runs {
                    let setting =
                        match heuristic_cfg.to_benchmark_setting(instance_path, problem_kind) {
                            Ok(s) => s,
                            Err(e) => {
                                tracing::error!("Config error for {}: {}", instance_path, e);
                                runs.push(SingleRunResult {
                                    run_index,
                                    status: format!("config error: {}", e),
                                    best_objective: 0.0,
                                    best_iteration: 0,
                                    time_to_best_secs: 0.0,
                                    total_time_secs: 0.0,
                                    solution: Vec::new(),
                                });
                                continue;
                            }
                        };

                    tracing::info!(
                        run = run_index + 1,
                        total = config.num_runs,
                        instance = %instance_path,
                        heuristic = %heuristic_cfg.kind,
                        max_iteration = ?heuristic_cfg.stop_condition.max_iteration,
                        max_duration_secs = ?heuristic_cfg.stop_condition.max_duration_secs,
                        max_failed_update = ?heuristic_cfg.stop_condition.max_failed_update,
                        "Starting run"
                    );

                    let metrics = match &setting {
                        BenchmarkSetting::MaxCut(s) => run_problem::<MaxCut>(
                            &s.instance_path,
                            s.heuristic.build(s.stop_condition.clone()),
                        ),
                        BenchmarkSetting::Qubo(s) => run_problem::<Qubo>(
                            &s.instance_path,
                            s.heuristic.build(s.stop_condition.clone()),
                        ),
                        BenchmarkSetting::Sat(s) => run_problem::<Sat>(
                            &s.instance_path,
                            s.heuristic.build(s.stop_condition.clone()),
                        ),
                        BenchmarkSetting::Tsp(s) => run_problem::<TspWithCoordinates>(
                            &s.instance_path,
                            s.heuristic.build(s.stop_condition.clone()),
                        ),
                    };

                    tracing::info!(
                        run = run_index + 1,
                        total = config.num_runs,
                        instance = %instance_path,
                        heuristic = %heuristic_cfg.kind,
                        objective = metrics.best_objective,
                        best_iteration = metrics.best_iteration,
                        time_to_best_secs = metrics.time_to_best_secs,
                        total_time_secs = metrics.total_time_secs,
                        "Run completed"
                    );

                    runs.push(to_single_run_result(run_index, metrics));
                }

                let minimize = matches!(problem_kind, ProblemKind::Qubo | ProblemKind::Tsp);
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
                    "=== Summary ==="
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
