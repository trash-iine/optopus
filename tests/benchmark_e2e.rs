//! End-to-end tests for the benchmark pipeline: TOML config →
//! `Benchmark::run_from_config` → report, including seed reproducibility
//! (which must hold even though runs execute in parallel under rayon).

use optopus::benchmark::{Benchmark, BenchmarkConfig, BenchmarkReport};
use optopus::prelude::{Graph, MaxCut};

/// Writes `content` to a temp file tagged with `tag` and returns its path.
///
/// The tag is what makes the path unique: every caller below passes a
/// different one, so no two concurrent tests build the same path. A wall-clock
/// suffix would not do that job — the clock is quantized (1 us on macOS), so
/// two tests entering here in the same microsecond would still collide and
/// read each other's file.
fn write_temp_file(tag: &str, content: &str) -> std::path::PathBuf {
    use std::io::Write;
    let mut path = std::env::temp_dir();
    path.push(format!("optopus_e2e_{tag}_{}.txt", std::process::id()));
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

    // Not `== 4.0`: the 4-cycle has a plateau. Its three-versus-one splits cut
    // 2 and every flip out of them gains exactly zero, so a strictly improving
    // descent stops there — measured, a single run reaches the optimal 4 about
    // three times in four, which makes best-of-three a ~1.5% coin flip. (The
    // config's `seed` cannot pin that down here: per-run seeds are derived
    // from the instance *path*, and this instance is a temp file named after
    // the process and the clock.) What every run does owe us is that it
    // stopped at a local optimum, and that the objective the report carries is
    // the cut the solution it carries actually makes.
    let cycle = MaxCut::new(Graph::from_edges([
        (0, 1, 1.0),
        (1, 2, 1.0),
        (2, 3, 1.0),
        (3, 0, 1.0),
    ]));
    for run in &result.runs {
        let mut x = vec![false; 4];
        for &v in &run.solution {
            x[v] = true;
        }
        assert_eq!(
            f64::from(cycle.calculate_cut_size(&x)),
            run.best_objective,
            "the reported objective must be the cut the reported solution makes"
        );
        for v in 0..4 {
            assert!(
                cycle.calculate_gain(&x, v) <= 0.0,
                "flipping {v} still improves, so the run stopped short of a local optimum"
            );
        }
    }
    assert!(result.summary.best_objective >= 2.0);
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
        // One clock per run: both are measured from `SearchState::start_time`,
        // so this holds exactly rather than within a tolerance.
        assert!(
            last_elapsed <= run.total_time_secs,
            "a trajectory point cannot postdate the run: {last_elapsed} > {}",
            run.total_time_secs
        );
        assert!(
            (run.time_to_best_secs - last_elapsed).abs() < 1e-9,
            "time_to_best must come from the trajectory"
        );
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
        assert_eq!(a.run_index, b.run_index);
        assert_eq!(a.seed, b.seed, "{tag} derived per-run seed must be stable");
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

/// BeamSearch draws nothing from the RNG after the seeded start solution, so
/// this pins that the start solution and the neighborhood order stay put.
#[test]
fn beam_search_is_bit_identical_across_reruns_with_seed() {
    assert_reproducible(
        "repro_bs",
        r#"
[[heuristics]]
kind = "BeamSearch"
neighbor = "Flip"
beam_width = 3

[heuristics.stop_condition]
max_iteration = 20
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

/// Population Annealing consumes the RNG for population init, resampling and
/// Metropolis sweeps; all must be seed-stable.
#[test]
fn population_annealing_is_bit_identical_across_reruns_with_seed() {
    assert_reproducible(
        "repro_pa",
        r#"
[[heuristics]]
kind = "PopulationAnnealing"
neighbor = "Flip"
population_size = 12
initial_beta = 0.1
delta_beta = 0.05
sweeps_per_step = 5
reset_period = 20

[heuristics.stop_condition]
max_iteration = 300
"#,
    );
}

/// ReinforcementLearningSearch consumes the RNG for reservoir sampling (`max_candidates`) and
/// softmax move sampling; this locks in the sampled-before-evaluation path.
#[test]
fn rl_search_is_bit_identical_across_reruns_with_seed() {
    assert_reproducible(
        "repro_rl",
        r#"
[[heuristics]]
kind = "ReinforcementLearningSearch"
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

/// VNS forks the RNG across three sub-heuristics (search + two shakes) and
/// exercises the incumbent-restore path on failed cycles.
#[test]
fn variable_neighborhood_search_is_bit_identical_across_reruns_with_seed() {
    assert_reproducible(
        "repro_vns",
        r#"
[[heuristics]]
kind = "VariableNeighborhoodSearch"

[heuristics.stop_condition]
max_iteration = 300

[[heuristics.steps]]
kind = "TabuSearch"
neighbor = "Flip"
tabu_tenure = [2, 5]

[heuristics.steps.stop_condition]
max_iteration = 50

[[heuristics.steps]]
kind = "RandomWalk"
neighbor = "Flip"

[heuristics.steps.stop_condition]
max_iteration = 5

[[heuristics.steps]]
kind = "RandomWalk"
neighbor = "Flip"

[heuristics.steps.stop_condition]
max_iteration = 15
"#,
    );
}

/// A CVRPLIB-format instance small enough for a fast test: a depot plus 8
/// customers on a ring, capacity 3.
fn write_temp_cvrp(tag: &str) -> std::path::PathBuf {
    let mut body = String::from(
        "NAME : e2e\nTYPE : CVRP\nDIMENSION : 9\nEDGE_WEIGHT_TYPE : EUC_2D\n\
         CAPACITY : 3\nNODE_COORD_SECTION\n1 0 0\n",
    );
    for i in 0..8 {
        let theta = 2.0 * std::f64::consts::PI * i as f64 / 8.0;
        body.push_str(&format!(
            "{} {} {}\n",
            i + 2,
            (10.0 * theta.cos()).round() as i64,
            (10.0 * theta.sin()).round() as i64
        ));
    }
    body.push_str("DEMAND_SECTION\n1 0\n");
    for i in 0..8 {
        body.push_str(&format!("{} 1\n", i + 2));
    }
    body.push_str("DEPOT_SECTION\n1\n-1\nEOF\n");
    write_temp_file(tag, &body)
}

/// HGS threads every decision — initial tours, parent tournaments, crossover
/// cut points, local-search move order, repair coin flips — through `state.rng`.
#[test]
fn hybrid_genetic_search_is_bit_identical_across_reruns_with_seed() {
    let instance = write_temp_cvrp("repro_hgs");
    let config_toml = format!(
        r#"
num_runs = 2
seed = 4242

[[instances]]
path = "{}"
problem = "Vrp"

[[heuristics]]
kind = "HybridGeneticSearch"
min_population_size = 6
generation_size = 8
granularity = 4
target_feasible = 0.2

[heuristics.stop_condition]
max_iteration = 200
"#,
        instance.display()
    );

    let first = assert_reruns_are_bit_identical(&config_toml);
    let _ = std::fs::remove_file(&instance);
    for run in &first.results[0].runs {
        // The encoding is `0, route…, 0, route…`: every customer exactly once.
        let mut customers: Vec<usize> = run.solution.iter().copied().filter(|&c| c != 0).collect();
        customers.sort_unstable();
        assert_eq!(customers, (1..=8).collect::<Vec<_>>());
    }
}

/// Runs `config_toml` twice and asserts the per-run results are bit-identical,
/// then hands back the first report for any further check.
fn assert_reruns_are_bit_identical(config_toml: &str) -> BenchmarkReport {
    let first = run_benchmark(config_toml);
    let second = run_benchmark(config_toml);
    for (a, b) in first.results[0].runs.iter().zip(&second.results[0].runs) {
        assert_eq!(a.status, "success");
        assert_eq!(a.seed, b.seed, "derived per-run seed must be stable");
        assert_eq!(a.best_objective, b.best_objective, "objective diverged");
        assert_eq!(a.best_iteration, b.best_iteration, "iteration diverged");
        assert_eq!(a.solution, b.solution, "run {} diverged", a.run_index);
    }
    first
}

/// Writes a 24-city TSPLIB instance, two rings of twelve.
fn write_temp_tsp(tag: &str) -> std::path::PathBuf {
    let mut body = String::from(
        "NAME : e2e\nTYPE : TSP\nDIMENSION : 24\nEDGE_WEIGHT_TYPE : EUC_2D\n\
         NODE_COORD_SECTION\n",
    );
    for i in 0..24 {
        let theta = 2.0 * std::f64::consts::PI * (i % 12) as f64 / 12.0;
        let radius = if i < 12 { 10.0 } else { 25.0 };
        body.push_str(&format!(
            "{} {} {}\n",
            i + 1,
            (radius * theta.cos()).round() as i64,
            (radius * theta.sin()).round() as i64
        ));
    }
    body.push_str("EOF\n");
    write_temp_file(tag, &body)
}

/// Ruin-and-recreate is registered for TSP through `alns_for_tsp`, so the
/// benchmark has to build it from a config and replay it bit for bit under a
/// seed, the anchored descent's shuffles included.
#[test]
fn alns_on_tsp_is_bit_identical_across_reruns_with_seed() {
    let instance = write_temp_tsp("alns_tsp");
    let config_toml = format!(
        r#"
num_runs = 2
seed = 4242

[[instances]]
path = "{}"
problem = "Tsp"

[[heuristics]]
kind = "AdaptiveLargeNeighborhoodSearch"

[heuristics.stop_condition]
max_iteration = 300
"#,
        instance.display()
    );
    let first = assert_reruns_are_bit_identical(&config_toml);
    let _ = std::fs::remove_file(&instance);
    assert_eq!(first.results[0].runs.len(), 2);
    for run in &first.results[0].runs {
        // The reported tour visits every city exactly once.
        let mut cities = run.solution.clone();
        cities.sort_unstable();
        assert_eq!(cities, (0..24).collect::<Vec<_>>());
    }
}

/// `HybridGeneticSearch` is registered only for `Vrp`; every other problem must
/// reject it when the config is validated, before any run starts, rather than
/// silently ignoring the config.
#[test]
fn hybrid_genetic_search_is_rejected_for_non_vrp_problems() {
    let instance = write_temp_file("hgs_reject", "4 4\n1 2 1\n2 3 1\n3 4 1\n4 1 1\n");
    let config_toml = format!(
        r#"
num_runs = 1

[[instances]]
path = "{}"
problem = "MaxCut"

[[heuristics]]
kind = "HybridGeneticSearch"

[heuristics.stop_condition]
max_iteration = 10
"#,
        instance.display()
    );
    let config: BenchmarkConfig = toml::from_str(&config_toml).expect("config parses");
    let result = Benchmark::run_from_config(config, "benchmark_e2e");
    let _ = std::fs::remove_file(&instance);
    let Err(err) = result else {
        panic!("an unsupported heuristic is a config error");
    };
    let msg = err.to_string();
    assert!(
        msg.contains("not supported for MaxCut") && msg.contains("HybridGeneticSearch"),
        "expected a config error naming the heuristic, got {msg:?}"
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

const SIX_VERTICES: &str = "6 9\n1 2 1\n2 3 1\n3 4 1\n4 5 1\n5 6 1\n6 1 1\n1 4 1\n2 5 1\n3 6 1\n";

/// A seeded tabu search with one `[[instances]]` entry per path.
fn seeded_config(paths: &[&str]) -> String {
    let instances: String = paths
        .iter()
        .map(|path| format!("[[instances]]\npath = \"{path}\"\nproblem = \"MaxCut\"\n\n"))
        .collect();
    format!(
        r#"
num_runs = 3
seed = 99

{instances}[[heuristics]]
kind = "TabuSearch"
neighbor = "Flip"
tabu_tenure = [1, 3]

[heuristics.stop_condition]
max_iteration = 300
"#
    )
}

/// The seeds come from positions in the config, not from the path, so the same
/// instance read from another path, as a server writing each request to a new
/// temp file does, repeats its seeded runs.
#[test]
fn a_seeded_run_repeats_when_the_instance_moves() {
    let a = write_temp_file("seed_path_a", SIX_VERTICES);
    let b = write_temp_file("seed_path_b", SIX_VERTICES);
    let first = run_benchmark(&seeded_config(&[&a.display().to_string()]));
    let second = run_benchmark(&seeded_config(&[&b.display().to_string()]));
    let _ = std::fs::remove_file(&a);
    let _ = std::fs::remove_file(&b);

    for (x, y) in first.results[0].runs.iter().zip(&second.results[0].runs) {
        assert_eq!(x.status, "success");
        assert_eq!(x.seed, y.seed, "the seed must not depend on the path");
        assert_eq!(x.best_objective, y.best_objective);
        assert_eq!(x.best_iteration, y.best_iteration);
        assert_eq!(x.solution, y.solution);
    }
}

/// Instances in one config draw from their own streams, even when they are the
/// same file.
#[test]
fn instances_in_one_config_get_their_own_seeds() {
    let path = write_temp_file("seed_position", SIX_VERTICES);
    let path_str = path.display().to_string();
    let report = run_benchmark(&seeded_config(&[&path_str, &path_str]));
    let _ = std::fs::remove_file(&path);

    let seeds = |i: usize| {
        report.results[i]
            .runs
            .iter()
            .map(|r| r.seed.expect("a seeded config reports each run's seed"))
            .collect::<Vec<_>>()
    };
    assert_eq!(report.results.len(), 2);
    assert!(
        seeds(0).iter().all(|s| !seeds(1).contains(s)),
        "{:?} and {:?} must not share a seed",
        seeds(0),
        seeds(1)
    );
}

/// A heterogeneous-fleet instance goes through the same pipeline as a CVRPLIB
/// one, the TOML format being chosen by the file extension. This pins the
/// whole path: `ProblemKind::Vrp` with a `.toml` path, `Vrp::load_file`, a
/// generic heuristic over a `Relocate` neighborhood, the report's
/// depot-separated solution encoding, and bit-identical reruns under a fixed
/// seed.
#[test]
fn a_fleet_vrp_instance_runs_from_toml_and_is_reproducible() {
    let config_toml = r#"
num_runs = 2
seed = 2024

[[instances]]
path = "data/instances/vrp/demo_fleet.toml"
problem = "Vrp"

[[heuristics]]
kind = "LocalSearch"
neighbor = "Relocate"

[heuristics.stop_condition]
max_iteration = 2000
"#;

    let report = assert_reruns_are_bit_identical(config_toml);
    let runs = &report.results[0].runs;
    assert_eq!(runs.len(), 2);
    for run in runs {
        // The encoding is depot-separated: `0, route…, 0, route…, 0`, so every
        // customer 1..=6 appears exactly once and the rest are depots.
        let mut customers: Vec<usize> = run.solution.iter().copied().filter(|&c| c != 0).collect();
        customers.sort_unstable();
        assert_eq!(customers, (1..=6).collect::<Vec<_>>(), "{:?}", run.solution);
    }
}

/// Runs a seeded `BranchAndBound` with a short local search per node on
/// `instance`, `extra` adding fields to the heuristic, and returns its runs.
fn branch_and_bound_runs(
    tag: &str,
    problem: &str,
    instance: &str,
    neighbor: &str,
    extra: &str,
) -> Vec<optopus::benchmark::SingleRunResult> {
    let path = write_temp_file(tag, instance);
    let config_toml = format!(
        r#"
num_runs = 2
seed = 5

[[instances]]
path = "{}"
problem = "{problem}"

[[heuristics]]
kind = "BranchAndBound"
{extra}
[heuristics.stop_condition]
max_iteration = 10_000

[[heuristics.steps]]
kind = "LocalSearch"
neighbor = "{neighbor}"
[heuristics.steps.stop_condition]
max_iteration = 100
"#,
        path.display()
    );
    let report = run_benchmark(&config_toml);
    let _ = std::fs::remove_file(&path);
    report.results.into_iter().next().unwrap().runs
}

/// `BranchAndBound` runs from a config on every problem it bounds, proves the
/// optimum of a small instance, and reports a dual bound on the objective's
/// own scale: equal to `best_objective` once that is proven.
#[test]
fn branch_and_bound_proves_small_instances_optimal() {
    let cases = [
        // A 5-cycle cuts at most 4 edges.
        (
            "bnb_max_cut",
            "MaxCut",
            "5 5\n1 2 1\n2 3 1\n3 4 1\n4 5 1\n5 1 1\n",
            "Flip",
            "",
            4.0,
        ),
        (
            "bnb_max_cut_eigen",
            "MaxCut",
            "5 5\n1 2 1\n2 3 1\n3 4 1\n4 5 1\n5 1 1\n",
            "Flip",
            "relaxation = \"Eigenvalue\"\neigen_max_iterations = 50\n",
            4.0,
        ),
        // Minimizes -x1 - x2 - x3 + 2 x1 x2: two of the three set, -2.
        (
            "bnb_qubo",
            "Qubo",
            "3 4\n1 1 -1\n2 2 -1\n3 3 -1\n1 2 2\n",
            "Flip",
            "",
            -2.0,
        ),
        // (x1 or x2) and (not x1 or x2) and (not x2 or x3): all three hold.
        (
            "bnb_sat",
            "Sat",
            "p cnf 3 3\n1 2 0\n-1 2 0\n-2 3 0\n",
            "Flip",
            "",
            3.0,
        ),
        // A path of four vertices is covered by its two inner vertices.
        (
            "bnb_vertex_cover",
            "VertexCover",
            "4 3\n1 2\n2 3\n3 4\n",
            "Flip",
            "",
            2.0,
        ),
    ];
    for (tag, problem, instance, neighbor, extra, optimum) in cases {
        for run in branch_and_bound_runs(tag, problem, instance, neighbor, extra) {
            assert_eq!(run.status, "success", "{tag}");
            assert_eq!(run.best_objective, optimum, "{tag}");
            assert_eq!(run.proven_optimal, Some(true), "{tag}");
            assert_eq!(run.dual_bound, Some(optimum), "{tag}");
        }
    }
}

/// Only `BranchAndBound` carries a certificate; other kinds leave both keys
/// out of the report.
#[test]
fn a_heuristic_that_bounds_nothing_reports_no_certificate() {
    let instance = write_temp_file("no_certificate", "3 3\n1 2 1\n1 3 1\n2 3 1\n");
    let report = run_benchmark(&seeded_config(&[&instance.display().to_string()]));
    let _ = std::fs::remove_file(&instance);
    for run in &report.results[0].runs {
        assert_eq!(run.proven_optimal, None);
        assert_eq!(run.dual_bound, None);
    }
    let toml = toml::to_string(&report).unwrap();
    assert!(!toml.contains("proven_optimal") && !toml.contains("dual_bound"));
}

/// A `BranchAndBound` config the factory cannot build fails at startup, with
/// a message naming what is wrong.
#[test]
fn branch_and_bound_configs_that_cannot_run_are_rejected() {
    let triangle = "3 3\n1 2 1\n1 3 1\n2 3 1\n";
    let tour = "NAME: t\nTYPE: TSP\nDIMENSION: 3\nEDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n\
                1 0 0\n2 1 0\n3 0 1\nEOF\n";
    let leaf = "[[heuristics.steps]]\nkind = \"LocalSearch\"\nneighbor = \"Flip\"\n";
    let cases = [
        (
            "bnb_bad_eigen_qubo",
            "Qubo",
            triangle,
            "relaxation = \"Eigenvalue\"\n",
            leaf,
            "relaxation 'Eigenvalue'",
        ),
        (
            "bnb_bad_tsp",
            "Tsp",
            tour,
            "",
            "[[heuristics.steps]]\nkind = \"LocalSearch\"\nneighbor = \"TwoOpt\"\n",
            "not supported for Tsp",
        ),
        (
            "bnb_bad_steps",
            "MaxCut",
            triangle,
            "",
            &format!("{leaf}{leaf}"),
            "exactly 1 step",
        ),
        (
            "bnb_bad_limit",
            "MaxCut",
            triangle,
            "eigen_certify_limit = 10\n",
            leaf,
            "only to relaxation = 'Eigenvalue'",
        ),
        (
            "bnb_bad_zero",
            "MaxCut",
            triangle,
            "relaxation = \"Eigenvalue\"\neigen_lanczos_steps = 0\n",
            leaf,
            "at least 1",
        ),
    ];
    for (tag, problem, instance, extra, steps, needle) in cases {
        let path = write_temp_file(tag, instance);
        let config_toml = format!(
            "num_runs = 1\n\n[[instances]]\npath = \"{}\"\nproblem = \"{problem}\"\n\n\
             [[heuristics]]\nkind = \"BranchAndBound\"\n{extra}\n{steps}",
            path.display()
        );
        let config: BenchmarkConfig = toml::from_str(&config_toml).expect("config parses");
        let result = Benchmark::run_from_config(config, "benchmark_e2e");
        let _ = std::fs::remove_file(&path);
        let Err(err) = result else {
            panic!("{tag}: must be rejected");
        };
        let msg = err.to_string();
        assert!(msg.contains(needle), "{tag}: {msg}");
    }
}
