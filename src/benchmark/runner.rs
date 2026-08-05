//! The parallel benchmark run loop.

use rayon::prelude::*;

use super::config::{BenchmarkConfig, HeuristicConfig, ProblemKind, validate_config};
use super::factory::{ConfigurableProblem, build_heuristic};
use super::problems::{BenchmarkProblem, BenchmarkSolution, ProblemVisitor, with_problem};
use super::report::{BenchmarkReport, InstanceHeuristicResult, SingleRunResult, compute_summary};
use crate::error::OptError;
use crate::heuristic::Heuristic;
use crate::search_state::{Distance, SearchState};

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
    trajectory: Vec<(f64, f64)>,
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
        trajectory: Vec::new(),
    }
}

/// Converts a recorded trajectory into `(elapsed_secs, objective)` pairs,
/// keeping only points that strictly improve the incumbent. Points merged
/// from `ClearBest`/`StartBest` sub-runs track the sub-run's *local* best,
/// which can be worse than an earlier global best; this filter restores the
/// monotone anytime curve.
fn monotone_trajectory(
    state: &SearchState<'_, impl crate::trait_defs::ProblemTrait>,
    minimize: bool,
) -> Vec<(f64, f64)> {
    let mut out: Vec<(f64, f64)> = Vec::with_capacity(state.trajectory.len());
    for p in &state.trajectory {
        let improves = out.last().is_none_or(|&(_, incumbent)| {
            if minimize {
                p.objective < incumbent
            } else {
                p.objective > incumbent
            }
        });
        if improves {
            out.push(((p.instant - state.start_time).as_secs_f64(), p.objective));
        }
    }
    out
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

/// Runs every heuristic from the config against one instance, loading the
/// instance file exactly once and sharing it across all heuristics and runs.
struct InstanceVisitor<'a> {
    config: &'a BenchmarkConfig,
    instance_path: &'a str,
    problem_kind: &'a ProblemKind,
}

impl ProblemVisitor for InstanceVisitor<'_> {
    type Output = Vec<InstanceHeuristicResult>;
    fn visit<P>(self) -> Vec<InstanceHeuristicResult>
    where
        P: ConfigurableProblem,
        P::Solution: BenchmarkSolution + Distance,
    {
        let instance = P::load_instance(self.instance_path);
        let mut results = Vec::with_capacity(self.config.heuristics.len());

        for (heuristic_idx, heuristic_cfg) in self.config.heuristics.iter().enumerate() {
            tracing::info!(
                instance = %self.instance_path,
                heuristic = heuristic_cfg.kind_name(),
                num_runs = self.config.num_runs,
                max_iteration = ?heuristic_cfg.stop_condition().max_iteration,
                max_duration_secs = ?heuristic_cfg.stop_condition().max_duration_secs,
                max_failed_update = ?heuristic_cfg.stop_condition().max_failed_update,
                seed = ?self.config.seed,
                "Start:"
            );

            let master_seed = self.config.seed;
            let mut runs: Vec<SingleRunResult> = (0..self.config.num_runs)
                .into_par_iter()
                .map(|run_index| {
                    let run_seed = master_seed
                        .map(|m| derive_run_seed(m, self.instance_path, heuristic_idx, run_index));
                    let metrics = run_typed::<P>(&instance, heuristic_cfg, run_seed);

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

            let summary = compute_summary(&runs, P::MINIMIZE);
            tracing::info!(
                instance = %self.instance_path,
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
                instance_path: self.instance_path.to_string(),
                problem: self.problem_kind.clone(),
                heuristic: heuristic_cfg.clone(),
                summary,
                runs,
            });
        }

        results
    }
}

fn run_typed<P>(
    instance: &Result<P, OptError>,
    config: &HeuristicConfig,
    seed: Option<u64>,
) -> RunMetrics
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
    let instance = match instance {
        Ok(v) => v,
        Err(e) => return empty_metrics(format!("error loading instance: {}", e), seed),
    };
    run_problem::<P>(instance, heuristic, P::MINIMIZE, seed)
}

fn run_problem<P>(
    instance: &P,
    mut heuristic: Box<dyn Heuristic<P>>,
    minimize: bool,
    seed: Option<u64>,
) -> RunMetrics
where
    P: BenchmarkProblem,
    P::Solution: BenchmarkSolution,
{
    let mut state = match seed {
        Some(s) => SearchState::new_with_seed(instance, s),
        None => SearchState::new(instance),
    };
    state.set_objective_probe(|s| s.best_objective_f64());
    let initial_objective = state.initial_solution.best_objective_f64();
    let start = std::time::Instant::now();
    let status = heuristic.run(&mut state);
    let total_time = start.elapsed();
    let best_objective = state.best_solution.best_objective_f64();
    let raw_diff = best_objective - initial_objective;
    let improvement = if minimize { -raw_diff } else { raw_diff };
    let trajectory = monotone_trajectory(&state, minimize);
    // The trajectory records the true wall-clock instant of each improvement,
    // so its last point is authoritative even when sub-run clones reset the
    // relative timers; fall back to the timer difference when no improvement
    // was ever recorded.
    let time_to_best_secs = trajectory
        .last()
        .map(|&(elapsed, _)| elapsed)
        .unwrap_or_else(|| (state.best_time - state.start_time).as_secs_f64());
    RunMetrics {
        status: status_str(status),
        best_objective,
        best_iteration: state.best_iteration,
        time_to_best_secs,
        total_time_secs: total_time.as_secs_f64(),
        initial_objective: Some(initial_objective),
        improvement: Some(improvement),
        n_accepted: Some(state.n_accepted),
        n_rejected: Some(state.n_rejected),
        n_best_updates: Some(state.n_best_updates),
        seed,
        solution: state.best_solution.encode_as_indices(),
        trajectory,
    }
}

fn status_str(r: Result<(), crate::error::OptError>) -> String {
    match r {
        Ok(_) => "success".to_string(),
        Err(e) => format!("error: {}", e),
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
        trajectory: m.trajectory,
    }
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
        let instance_paths = expand_instance_paths(&config)?;

        // Instances run in parallel with each other, and each (heuristic, run)
        // pair inside an instance runs in parallel too (rayon nests fine).
        // Per-run seeds are derived from (master, path, heuristic, run), so the
        // report is independent of scheduling order.
        let results: Vec<InstanceHeuristicResult> = instance_paths
            .par_iter()
            .map(|(instance_path, problem_kind)| {
                with_problem(
                    problem_kind,
                    InstanceVisitor {
                        config: &config,
                        instance_path,
                        problem_kind,
                    },
                )
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .collect();

        Ok(BenchmarkReport {
            timestamp,
            config_file: config_file.to_string(),
            results,
        })
    }
}
