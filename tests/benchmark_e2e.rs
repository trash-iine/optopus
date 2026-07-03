//! End-to-end tests for the benchmark pipeline: TOML config →
//! `Benchmark::run_from_config` → report, including seed reproducibility
//! (which must hold even though runs execute in parallel under rayon).

use optopus::benchmark::{Benchmark, BenchmarkConfig, BenchmarkReport};

/// Writes `content` to a unique temp file and returns its path.
fn write_temp_file(tag: &str, content: &str) -> std::path::PathBuf {
    use std::io::Write;
    let mut path = std::env::temp_dir();
    path.push(format!(
        "optopus_e2e_{tag}_{}_{}.txt",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(content.as_bytes()).unwrap();
    path
}

fn run_benchmark(config_toml: &str) -> BenchmarkReport {
    let config: BenchmarkConfig = toml::from_str(config_toml).expect("config parses");
    Benchmark::run_from_config(config, "benchmark_e2e").expect("benchmark runs")
}

#[test]
fn run_from_config_executes_and_reports_all_runs() {
    // 4-cycle MaxCut instance; optimal cut value is 4.
    let instance = write_temp_file("cycle", "4 4\n1 2 1\n2 3 1\n3 4 1\n4 1 1\n");
    let config_toml = format!(
        r#"
num_runs = 3
seed = 42

[[instances]]
path = "{}"
problem = "MaxCut"

[[heuristics]]
kind = "LocalSearch"
neighbor = "Flip"

[heuristics.stop_condition]
max_iteration = 1000
"#,
        instance.display()
    );

    let report = run_benchmark(&config_toml);
    let _ = std::fs::remove_file(&instance);

    assert_eq!(report.results.len(), 1);
    let result = &report.results[0];
    assert_eq!(result.runs.len(), 3);
    for run in &result.runs {
        assert_eq!(run.status, "success");
        assert!(run.seed.is_some(), "per-run seed must be reported");
        // `solution` encodes the vertices on one side of the cut (0-indexed).
        assert!(run.solution.iter().all(|&v| v < 4));
    }
    assert_eq!(result.summary.num_successful_runs, 3);
    // Greedy local search always reaches the optimum on a 4-cycle.
    assert_eq!(result.summary.best_objective, 4.0);
}

#[test]
fn run_from_config_is_bit_identical_across_reruns_with_seed() {
    let instance = write_temp_file("repro", "4 4\n1 2 1\n2 3 1\n3 4 1\n4 1 1\n");
    // SimulatedAnnealing consumes the RNG on every step, so any seeding
    // mistake (including rayon scheduling nondeterminism) would show up here.
    let config_toml = format!(
        r#"
num_runs = 4
seed = 123

[[instances]]
path = "{}"
problem = "MaxCut"

[[heuristics]]
kind = "SimulatedAnnealing"
neighbor = "Flip"
initial_temperature = 1.0
cooling_rate = 0.99

[heuristics.stop_condition]
max_iteration = 500
"#,
        instance.display()
    );

    let first = run_benchmark(&config_toml);
    let second = run_benchmark(&config_toml);
    let _ = std::fs::remove_file(&instance);

    let first_runs = &first.results[0].runs;
    let second_runs = &second.results[0].runs;
    assert_eq!(first_runs.len(), second_runs.len());
    for (a, b) in first_runs.iter().zip(second_runs) {
        assert_eq!(a.run_index, b.run_index);
        assert_eq!(a.seed, b.seed, "derived per-run seed must be stable");
        assert_eq!(a.best_objective, b.best_objective);
        assert_eq!(a.best_iteration, b.best_iteration);
        assert_eq!(a.solution, b.solution, "run {} diverged", a.run_index);
    }
}

#[test]
fn run_from_config_rejects_empty_glob() {
    let config_toml = r#"
num_runs = 1

[[instances]]
path = "/nonexistent/optopus_e2e/*.txt"
problem = "MaxCut"

[[heuristics]]
kind = "LocalSearch"
neighbor = "Flip"

[heuristics.stop_condition]
max_iteration = 10
"#;
    let config: BenchmarkConfig = toml::from_str(config_toml).expect("config parses");
    let result = Benchmark::run_from_config(config, "benchmark_e2e");
    assert!(result.is_err(), "empty glob must be rejected");
}
