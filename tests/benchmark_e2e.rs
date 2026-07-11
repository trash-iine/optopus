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
fn runs_record_a_monotone_anytime_trajectory() {
    let instance = write_temp_file("trajectory", "4 4\n1 2 1\n2 3 1\n3 4 1\n4 1 1\n");
    // Iterated composition exercises the sub-run trajectory merge path.
    let config_toml = format!(
        r#"
num_runs = 2
seed = 7

[[instances]]
path = "{}"
problem = "MaxCut"

[[heuristics]]
kind = "Iterated"

[heuristics.stop_condition]
max_iteration = 200

[[heuristics.steps]]
kind = "LocalSearch"
neighbor = "Flip"

[heuristics.steps.stop_condition]
max_iteration = 50

[[heuristics.steps]]
kind = "SimulatedAnnealing"
neighbor = "Flip"
initial_temperature = 10.0
cooling_rate = 1.0

[heuristics.steps.stop_condition]
max_iteration = 5
"#,
        instance.display()
    );

    let report = run_benchmark(&config_toml);
    let _ = std::fs::remove_file(&instance);

    for run in &report.results[0].runs {
        assert_eq!(run.status, "success");
        assert!(
            !run.trajectory.is_empty(),
            "an improving run must record trajectory points"
        );
        // MaxCut maximizes: objectives strictly increase, times never decrease.
        for pair in run.trajectory.windows(2) {
            assert!(pair[1].1 > pair[0].1, "objective must strictly improve");
            assert!(pair[1].0 >= pair[0].0, "elapsed time must be monotone");
        }
        let &(last_elapsed, last_objective) = run.trajectory.last().unwrap();
        assert_eq!(
            last_objective, run.best_objective,
            "trajectory must end at the final best objective"
        );
        assert!(last_elapsed <= run.total_time_secs + 1e-6);
        assert!(
            (run.time_to_best_secs - last_elapsed).abs() < 1e-9,
            "time_to_best must come from the trajectory"
        );
    }
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

/// Runs `heuristic_toml` twice on a fixed instance with a fixed seed and asserts
/// bit-identical per-run results.
fn assert_reproducible(tag: &str, heuristic_toml: &str) {
    // A slightly larger instance so tabu/perturbation phases actually engage.
    let instance = write_temp_file(
        tag,
        "6 9\n1 2 1\n2 3 1\n3 4 1\n4 5 1\n5 6 1\n6 1 1\n1 4 1\n2 5 1\n3 6 1\n",
    );
    let config_toml = format!(
        r#"
num_runs = 3
seed = 777

[[instances]]
path = "{}"
problem = "MaxCut"

{}
"#,
        instance.display(),
        heuristic_toml
    );

    let first = run_benchmark(&config_toml);
    let second = run_benchmark(&config_toml);
    let _ = std::fs::remove_file(&instance);

    let first_runs = &first.results[0].runs;
    let second_runs = &second.results[0].runs;
    assert_eq!(first_runs.len(), second_runs.len());
    for (a, b) in first_runs.iter().zip(second_runs) {
        assert_eq!(a.status, "success");
        assert_eq!(
            a.best_objective, b.best_objective,
            "{tag} objective diverged"
        );
        assert_eq!(
            a.best_iteration, b.best_iteration,
            "{tag} iteration diverged"
        );
        assert_eq!(a.solution, b.solution, "{tag} run {} diverged", a.run_index);
    }
}

/// TabuSearch samples the tabu tenure from the RNG on every applied move, so
/// this locks in the seeded-tenure fix (previously the thread RNG was used and
/// runs were not reproducible).
#[test]
fn tabu_search_is_bit_identical_across_reruns_with_seed() {
    assert_reproducible(
        "repro_ts",
        r#"
[[heuristics]]
kind = "TabuSearch"
neighbor = "Flip"
tabu_tenure = [2, 5]

[heuristics.stop_condition]
max_iteration = 300
"#,
    );
}

/// BLS consumes the RNG for tenure sampling, perturbation-type selection, and
/// random flips; all three previously used the thread RNG.
#[test]
fn breakout_local_search_is_bit_identical_across_reruns_with_seed() {
    assert_reproducible(
        "repro_bls",
        r#"
[[heuristics]]
kind = "BreakoutLocalSearch"
tabu_tenure = [2, 5]
t = 100
l0 = 3
p0 = 0.8
q = 0.5

[heuristics.stop_condition]
max_iteration = 300
"#,
    );
}

/// The plateau-cluster perturbation adds RNG consumption for cluster seeding
/// and the strong fallback; runs with `plateau_prob` set must be seed-stable.
#[test]
fn breakout_local_search_with_plateau_is_bit_identical_across_reruns_with_seed() {
    assert_reproducible(
        "repro_bls_plateau",
        r#"
[[heuristics]]
kind = "BreakoutLocalSearch"
tabu_tenure = [2, 5]
t = 100
l0 = 3
p0 = 0.8
q = 0.5
plateau_prob = 0.4

[heuristics.stop_condition]
max_iteration = 300
"#,
    );
}

/// Population Annealing consumes the RNG for population init, Metropolis
/// sweeps, cluster-move independent-set selection, and resampling; all must be
/// seed-stable.
#[test]
fn population_annealing_is_bit_identical_across_reruns_with_seed() {
    assert_reproducible(
        "repro_pa",
        r#"
[[heuristics]]
kind = "PopulationAnnealingForMaxCut"
population_size = 12
initial_beta = 0.1
delta_beta = 0.05
sweeps_per_step = 5
reset_period = 20
cluster_moves = true

[heuristics.stop_condition]
max_iteration = 300
"#,
    );
}

/// The RL perturbation controller consumes the RNG for tabu tenures, bandit
/// action sampling, and strong-perturbation flips; all must be seed-stable.
#[test]
fn rl_breakout_local_search_is_bit_identical_across_reruns_with_seed() {
    assert_reproducible(
        "repro_rl_bls",
        r#"
[[heuristics]]
kind = "RlBreakoutLocalSearch"
tabu_tenure = [2, 5]
t = 100
l0 = 3
learning_rate = 0.1
exploration = 0.05

[heuristics.stop_condition]
max_iteration = 300
"#,
    );
}

/// RlSearch consumes the RNG for reservoir sampling (`max_candidates`) and
/// softmax move sampling; this locks in the sampled-before-evaluation path.
#[test]
fn rl_search_is_bit_identical_across_reruns_with_seed() {
    assert_reproducible(
        "repro_rl",
        r#"
[[heuristics]]
kind = "RlSearch"
neighbor = "Flip"
learning_rate = 0.05
max_candidates = 4

[heuristics.stop_condition]
max_iteration = 300
"#,
    );
}

/// Meta-heuristic composition forks the RNG per sub-run; this covers the
/// clone_for_new_run path together with the tabu tenure fix.
#[test]
fn iterated_tabu_search_is_bit_identical_across_reruns_with_seed() {
    assert_reproducible(
        "repro_ils",
        r#"
[[heuristics]]
kind = "Iterated"

[heuristics.stop_condition]
max_iteration = 300

[[heuristics.steps]]
kind = "TabuSearch"
neighbor = "Flip"
tabu_tenure = [2, 5]

[heuristics.steps.stop_condition]
max_iteration = 50

[[heuristics.steps]]
kind = "SimulatedAnnealing"
neighbor = "Flip"
initial_temperature = 10.0
cooling_rate = 1.0

[heuristics.steps.stop_condition]
max_iteration = 10
"#,
    );
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
