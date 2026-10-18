//! The parallel benchmark run loop.

use rayon::prelude::*;

use super::config::{BenchmarkConfig, HeuristicConfig, ProblemKind, validate_config};
use super::factory::{ConfigurableProblem, Search, build_search};
use super::problems::{BenchmarkProblem, BenchmarkSolution, ProblemVisitor, with_problem};
use super::report::{BenchmarkReport, InstanceHeuristicResult, SingleRunResult, compute_summary};
use crate::error::OptError;
use crate::search_state::{Distance, Evaluate, SearchState};

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

/// Namespace for the benchmark entry point ([`Benchmark::run_from_config`]).
pub struct Benchmark;

// ---------------------------------------------------------------------------
// Generic run functions
// ---------------------------------------------------------------------------

/// The result of a run that never started: `status` says why.
fn failed_run(run_index: usize, status: String, seed: Option<u64>) -> SingleRunResult {
    SingleRunResult {
        run_index,
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
        proven_optimal: None,
        dual_bound: None,
        solution: Vec::new(),
        trajectory: Vec::new(),
    }
}

/// Converts a recorded trajectory into `(elapsed_secs, objective)` pairs,
/// keeping only points that strictly improve the incumbent. Points merged
/// from `ClearBest`/`StartBest` sub-runs track the sub-run's local best,
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
/// is small (e.g. `seed = 0`). An instance enters by its position among the
/// expanded instances, not its path, so the seeds depend on the config alone
/// and the same instance read from another path repeats its runs.
fn derive_run_seed(
    master: u64,
    instance_idx: usize,
    heuristic_idx: usize,
    run_index: usize,
) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    master.hash(&mut h);
    (instance_idx as u64).hash(&mut h);
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
    /// The instance's position among the expanded instances.
    instance_idx: usize,
    instance_path: &'a str,
    initial_solution: Option<&'a [usize]>,
    problem_kind: &'a ProblemKind,
}

impl ProblemVisitor for InstanceVisitor<'_> {
    type Output = Vec<InstanceHeuristicResult>;
    fn visit<P>(self) -> Vec<InstanceHeuristicResult>
    where
        P: ConfigurableProblem,
        P::Solution: BenchmarkSolution + Distance + Evaluate,
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
                        .map(|m| derive_run_seed(m, self.instance_idx, heuristic_idx, run_index));
                    let result = run_typed::<P>(
                        &instance,
                        self.initial_solution,
                        heuristic_cfg,
                        run_index,
                        run_seed,
                    );

                    tracing::info!(
                        run = run_index + 1,
                        objective = result.best_objective,
                        best_iteration = result.best_iteration,
                        time_to_best_secs = result.time_to_best_secs,
                        total_time_secs = result.total_time_secs,
                        "Completed:"
                    );

                    result
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

/// `initial_solution` is the instance's, in the report's index encoding,
/// decoded here because a solution need not be shared across threads.
fn run_typed<P>(
    instance: &Result<P, OptError>,
    initial_solution: Option<&[usize]>,
    config: &HeuristicConfig,
    run_index: usize,
    seed: Option<u64>,
) -> SingleRunResult
where
    P: ConfigurableProblem,
    P::Solution: BenchmarkSolution + Distance + Evaluate,
{
    let search = match build_search::<P>(config) {
        Ok(h) => h,
        Err(e) => {
            let msg = match e {
                OptError::Config(m) => m,
                other => other.to_string(),
            };
            return failed_run(run_index, format!("config error: {msg}"), seed);
        }
    };
    let instance = match instance {
        Ok(v) => v,
        Err(e) => {
            return failed_run(run_index, format!("error loading instance: {}", e), seed);
        }
    };
    let start = match initial_solution.map(|indices| instance.decode_indices(indices)) {
        Some(Ok(solution)) => Some(solution),
        Some(Err(e)) => {
            return failed_run(run_index, format!("initial solution error: {e}"), seed);
        }
        None => None,
    };
    run_problem::<P>(instance, start, search, P::MINIMIZE, run_index, seed)
}

/// Runs `heuristic` once, from `start` when given and from a random solution
/// otherwise.
fn run_problem<P>(
    instance: &P,
    start: Option<P::Solution>,
    mut search: Search<P>,
    minimize: bool,
    run_index: usize,
    seed: Option<u64>,
) -> SingleRunResult
where
    P: BenchmarkProblem,
    P::Solution: BenchmarkSolution,
{
    let mut state = match (start, seed) {
        (Some(solution), Some(s)) => SearchState::with_solution_and_seed(instance, solution, s),
        (Some(solution), None) => SearchState::with_solution(instance, solution),
        (None, Some(s)) => SearchState::new_with_seed(instance, s),
        (None, None) => SearchState::new(instance),
    };
    state.set_objective_probe(|s| s.best_objective_f64());
    let initial_objective = state.initial_solution.best_objective_f64();
    let status = match &mut search {
        Search::Plain(h) => h.run(&mut state),
        Search::Proving(h) => h.run(&mut state),
    };
    // Timed from `state.start_time`, the origin the trajectory points are
    // reported against, so `time_to_best_secs <= total_time_secs` holds.
    let total_time = state.duration();
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
    // A run that failed proved nothing, and one that took no node proved
    // nothing either. The bound is infinite until the search has bounded the
    // whole problem, which a report cannot carry.
    let (proven_optimal, dual_bound) = match &search {
        Search::Proving(h) if status.is_ok() => {
            let certificate = h.certificate();
            (
                Some(certificate.is_some_and(|c| c.proven_optimal)),
                certificate
                    .map(|c| c.dual_bound.raw())
                    .filter(|b| b.is_finite()),
            )
        }
        _ => (None, None),
    };
    SingleRunResult {
        run_index,
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
        proven_optimal,
        dual_bound,
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

/// One instance file a config's `[[instances]]` entry matched.
struct ExpandedInstance<'a> {
    path: String,
    initial_solution: Option<&'a [usize]>,
    problem: ProblemKind,
}

fn expand_instance_paths(config: &BenchmarkConfig) -> Result<Vec<ExpandedInstance<'_>>, OptError> {
    let mut instance_paths: Vec<ExpandedInstance> = Vec::new();

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
        // Every matched file would start from the one solution.
        if inst.initial_solution.is_some() && expanded.len() > 1 {
            return Err(OptError::Config(format!(
                "'initial_solution' names one instance, but '{}' matched {} files",
                inst.path,
                expanded.len()
            )));
        }

        for path in expanded {
            let Some(path_str) = path.to_str() else {
                return Err(OptError::Config(format!(
                    "instance path '{}' is not valid UTF-8",
                    path.display()
                )));
            };
            instance_paths.push(ExpandedInstance {
                path: path_str.to_string(),
                initial_solution: inst.initial_solution.as_deref(),
                problem: inst.problem.clone(),
            });
        }
    }

    Ok(instance_paths)
}

impl Benchmark {
    /// The `best_objective` a run ending on `solution` would report, the
    /// penalty included where the problem's objective carries one.
    /// `solution` is written as a report's `solution` is, and `path` names
    /// one instance file, not a glob.
    pub fn evaluate_solution(
        problem: &ProblemKind,
        path: &str,
        solution: &[usize],
    ) -> Result<f64, OptError> {
        struct Score<'a> {
            path: &'a str,
            solution: &'a [usize],
        }
        impl ProblemVisitor for Score<'_> {
            type Output = Result<f64, OptError>;
            fn visit<P>(self) -> Result<f64, OptError>
            where
                P: ConfigurableProblem,
                P::Solution: BenchmarkSolution + Distance + Evaluate,
            {
                let prob = P::load_instance(self.path)?;
                Ok(prob.decode_indices(self.solution)?.best_objective_f64())
            }
        }
        with_problem(problem, Score { path, solution })
    }

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
        // Per-run seeds are derived from (master, instance, heuristic, run)
        // positions, so the report is independent of scheduling order.
        let results: Vec<InstanceHeuristicResult> = instance_paths
            .par_iter()
            .enumerate()
            .map(|(instance_idx, inst)| {
                with_problem(
                    &inst.problem,
                    InstanceVisitor {
                        config: &config,
                        instance_idx,
                        instance_path: &inst.path,
                        initial_solution: inst.initial_solution,
                        problem_kind: &inst.problem,
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
