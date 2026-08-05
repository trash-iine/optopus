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
