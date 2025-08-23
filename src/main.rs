use optopus::{
    heuristic::{BreakoutLocalSearchForMaxCut, Heuristic, StopCondition},
    problem::max_cut::MaxCut,
    search_state::SearchState,
};
use serde::Serialize;

#[derive(Serialize)]
struct BenchmarkResult {
    status: String,
    instance: String,
    best_objective: f64,
    best_iteration: u64,
    time_taken: f64,
    solution: Vec<usize>,
}

#[derive(Serialize)]
struct Benchmark {
    results: Vec<BenchmarkResult>,
}

fn main() {
    tracing_subscriber::fmt::init();

    let mut results = Vec::new();

    let files = glob::glob("data/max_cut/G*").unwrap();
    for file in files {
        tracing::info!("Processing file: {:?}", file);
        let mc = MaxCut::load_from_file(file.as_ref().unwrap().to_str().unwrap()).unwrap();
        let mut state = SearchState::new(&mc, rand::rng());
        let sc = StopCondition::new(Some(200000), None, None);
        let bls = BreakoutLocalSearchForMaxCut::new(
            (3, (mc.len() / 10) as u64),
            sc,
            1000,
            (mc.len() / 100) as u64,
            0.8,
            0.5,
        );
        let status = bls.run(&mut state);
        tracing::info!("Best objective: {}", state.best_objective);
        results.push(BenchmarkResult {
            status: {
                match status {
                    Ok(_) => "success".to_string(),
                    Err(e) => format!("error: {}", e),
                }
            },
            instance: file
                .unwrap()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string(),
            best_objective: state.best_objective.into(),
            best_iteration: state.best_iteration,
            time_taken: state.best_time.elapsed().as_secs_f64(),
            solution: state
                .best_solution
                .cut
                .iter()
                .filter(|(_, &v)| v)
                .map(|(&i, _)| i)
                .collect(),
        });
    }

    // Serialize results to TOML
    let toml_str = toml::to_string(&Benchmark { results }).unwrap();
    let output_file = chrono::Local::now()
        .format("result/%Y%m%d_%H%M%S.toml")
        .to_string();
    std::fs::write(output_file, toml_str).unwrap();
}
