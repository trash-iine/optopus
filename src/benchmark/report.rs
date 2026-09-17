//! Output report types and summary statistics.

use serde::Serialize;

use super::config::HeuristicConfig;
use crate::error::OptError;

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
    /// Anytime trajectory: `(elapsed_secs, objective)` per strict improvement
    /// of the incumbent, monotone in both time and objective.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub trajectory: Vec<(f64, f64)>,
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
    pub problem: crate::benchmark::config::ProblemKind,
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

pub(crate) fn compute_summary(runs: &[SingleRunResult], minimize: bool) -> Summary {
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
    let min = objectives.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = objectives.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let (best, worst) = if minimize { (min, max) } else { (max, min) };
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
    let accepted_vals: Vec<f64> = successful
        .iter()
        .filter_map(|r| r.n_accepted.map(|v| v as f64))
        .collect();
    let rejected_vals: Vec<f64> = successful
        .iter()
        .filter_map(|r| r.n_rejected.map(|v| v as f64))
        .collect();
    let avg_acceptance_rate = if accepted_vals.len() == n && rejected_vals.len() == n {
        let rates: Vec<f64> = accepted_vals
            .iter()
            .zip(rejected_vals.iter())
            .map(|(&a, &r)| {
                let total = a + r;
                if total == 0.0 { 0.0 } else { a / total }
            })
            .collect();
        Some(rates.iter().sum::<f64>() / n as f64)
    } else {
        None
    };
    let avg_n_accepted = avg_opt(accepted_vals);
    let avg_n_rejected = avg_opt(rejected_vals);
    let avg_n_best_updates = avg_opt(
        successful
            .iter()
            .filter_map(|r| r.n_best_updates.map(|v| v as f64))
            .collect(),
    );

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

impl BenchmarkReport {
    /// Serializes the report as TOML into
    /// `<out_dir>/<config_stem>_<timestamp>.toml` (stem and timestamp taken
    /// from the report itself), creating the directory if needed.
    /// Returns the path of the written file.
    pub fn write_to_dir(
        &self,
        out_dir: impl AsRef<std::path::Path>,
    ) -> Result<std::path::PathBuf, OptError> {
        let out_dir = out_dir.as_ref();
        let toml_str = toml::to_string(self)?;
        std::fs::create_dir_all(out_dir).map_err(|e| OptError::FileLoad {
            path: out_dir.display().to_string(),
            line: 0,
            detail: format!("failed to create output directory: {e}"),
        })?;
        let config_stem = std::path::Path::new(&self.config_file)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("benchmark");
        let output_file = out_dir.join(format!("{config_stem}_{}.toml", self.timestamp));
        std::fs::write(&output_file, toml_str).map_err(|e| OptError::FileLoad {
            path: output_file.display().to_string(),
            line: 0,
            detail: format!("failed to write result file: {e}"),
        })?;
        Ok(output_file)
    }
}

#[cfg(test)]
mod report_tests {
    use super::*;

    /// A run with everything set, so a test can knock out one field at a time.
    fn run(best_objective: f64, status: &str) -> SingleRunResult {
        SingleRunResult {
            run_index: 0,
            status: status.to_string(),
            best_objective,
            best_iteration: 1,
            time_to_best_secs: 0.5,
            total_time_secs: 1.0,
            initial_objective: Some(0.0),
            improvement: Some(best_objective),
            n_accepted: Some(3),
            n_rejected: Some(1),
            n_best_updates: Some(2),
            seed: None,
            solution: vec![],
            trajectory: vec![],
        }
    }

    /// `best` and `worst` are the two ends of the same range, and which end is
    /// which is the only thing `minimize` decides. Getting it backwards would
    /// not fail any run, only report the worst result of every minimizing
    /// benchmark as its best.
    #[test]
    fn minimize_decides_which_end_of_the_range_is_best() {
        let runs = [
            run(1.0, "success"),
            run(5.0, "success"),
            run(3.0, "success"),
        ];

        let maximized = compute_summary(&runs, false);
        assert_eq!(maximized.best_objective, 5.0);
        assert_eq!(maximized.worst_objective, 1.0);

        let minimized = compute_summary(&runs, true);
        assert_eq!(minimized.best_objective, 1.0);
        assert_eq!(minimized.worst_objective, 5.0);

        // The direction changes nothing else.
        assert_eq!(maximized.avg_objective, 3.0);
        assert_eq!(minimized.avg_objective, 3.0);

        // Population standard deviation of 1, 5, 3 about 3: sqrt(8/3).
        assert!((minimized.std_objective - (8.0f64 / 3.0).sqrt()).abs() < 1e-12);
    }

    /// Every problem's direction is the one the summary is read with, so the
    /// table is checked through the summary rather than against a copy of
    /// itself.
    #[test]
    fn every_problem_kind_summarises_in_its_own_direction() {
        use crate::benchmark::config::ProblemKind;

        let runs = [run(1.0, "success"), run(5.0, "success")];
        for kind in ProblemKind::ALL {
            let summary = compute_summary(&runs, kind.minimize());
            let (best, worst) = match kind {
                ProblemKind::MaxCut | ProblemKind::Sat => (5.0, 1.0),
                _ => (1.0, 5.0),
            };
            assert_eq!(summary.best_objective, best, "{kind:?}");
            assert_eq!(summary.worst_objective, worst, "{kind:?}");
        }
    }

    /// Failed runs are dropped before anything is averaged, and a combination
    /// where every run failed has no statistics rather than zeroed ones --
    /// a zero would read as a legitimate result in the report.
    #[test]
    fn failed_runs_are_excluded_and_an_all_failed_summary_is_not_a_number() {
        let mixed = [
            run(1.0, "success"),
            run(100.0, "error: out of memory"),
            run(3.0, "success"),
        ];
        let summary = compute_summary(&mixed, false);
        assert_eq!(summary.num_successful_runs, 2);
        assert_eq!(summary.best_objective, 3.0);
        assert_eq!(summary.avg_objective, 2.0);

        let none = [run(1.0, "error: out of memory")];
        let summary = compute_summary(&none, false);
        assert_eq!(summary.num_successful_runs, 0);
        assert!(summary.best_objective.is_nan());
        assert!(summary.avg_objective.is_nan());
        assert!(summary.std_objective.is_nan());
        assert!(summary.avg_total_time_secs.is_nan());
        assert!(summary.avg_improvement.is_none());
        assert!(summary.avg_acceptance_rate.is_none());
    }

    /// The optional averages are all-or-nothing: a field that only some runs
    /// carry is reported for none of them, because an average over a subset
    /// would silently be an average over a different denominator.
    #[test]
    fn an_optional_field_is_averaged_only_when_every_run_carries_it() {
        let mut runs = [run(1.0, "success"), run(3.0, "success")];
        let summary = compute_summary(&runs, false);
        assert_eq!(summary.avg_n_accepted, Some(3.0));
        assert_eq!(summary.avg_n_best_updates, Some(2.0));
        assert_eq!(summary.avg_acceptance_rate, Some(0.75)); // 3 / (3 + 1)

        runs[1].n_accepted = None;
        let summary = compute_summary(&runs, false);
        assert_eq!(summary.avg_n_accepted, None);
        assert_eq!(summary.avg_acceptance_rate, None, "the rate needs both");
        assert_eq!(summary.avg_n_rejected, Some(1.0), "the other is unaffected");
    }

    /// A run that applied no move at all divides by zero; the rate is reported
    /// as no acceptances rather than as a NaN that spreads through the average.
    #[test]
    fn a_run_that_moved_nothing_has_an_acceptance_rate_of_zero() {
        let mut runs = [run(1.0, "success"), run(3.0, "success")];
        runs[0].n_accepted = Some(0);
        runs[0].n_rejected = Some(0);
        let summary = compute_summary(&runs, false);
        assert_eq!(summary.avg_acceptance_rate, Some(0.375)); // (0 + 0.75) / 2
    }

    #[test]
    fn write_to_dir_creates_timestamped_file() {
        let report = BenchmarkReport {
            timestamp: "19700101_000000".to_string(),
            config_file: "configs/my_bench.toml".to_string(),
            results: vec![],
        };
        let mut dir = std::env::temp_dir();
        dir.push(format!("optopus_report_test_{}", std::process::id()));
        let path = report.write_to_dir(&dir).expect("report writes");
        assert_eq!(
            path.file_name().and_then(|s| s.to_str()),
            Some("my_bench_19700101_000000.toml")
        );
        let written = std::fs::read_to_string(&path).unwrap();
        assert!(written.contains("19700101_000000"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
