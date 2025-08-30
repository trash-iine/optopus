use optopus::{
    heuristic::{
        BreakoutLocalSearchForMaxCut, Heuristic, ParallelHeuristic, StopCondition, TabuSearch,
    },
    problem::{max_cut::MaxCut, MaxCutFlipNeighbor, MaxCutSwapNeighbor},
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
        let instance_number = file
            .as_ref()
            .unwrap()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .replace("G", "")
            .parse::<u64>()
            .unwrap();
        if instance_number > 40 {
            continue;
        }

        tracing::info!("Processing file: {:?}", file);
        let mc = MaxCut::load_from_file(file.as_ref().unwrap().to_str().unwrap()).unwrap();
        let mut state = SearchState::new(&mc, rand::rng());
        // let sc = StopCondition::new(Some(200000), None, None);
        let sc = StopCondition::new(Some(30000), None, None);
        let bls = BreakoutLocalSearchForMaxCut::new(
            (3, (mc.len() / 10) as u64),
            sc,
            1000,
            (mc.len() / 100) as u64,
            0.8,
            0.5,
        );
        let ts = TabuSearch::<MaxCutFlipNeighbor>::new(
            // StopCondition::new(Some(200000), None, None),
            StopCondition::new(Some(1000000), None, None),
            (3, (mc.len() / 100) as u64),
            None,
        );
        // let status = bls.run(&mut state);
        let start = std::time::Instant::now();
        // let status = ts.run(&mut state);
        let status = bls.run(&mut state);
        let end = std::time::Instant::now();
        tracing::info!(
            "Best objective: {} ({:?})",
            state.best_objective,
            end - start
        );
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
        break;
    }

    // Serialize results to TOML
    let toml_str = toml::to_string(&Benchmark { results }).unwrap();
    let output_file = chrono::Local::now()
        .format("result/%Y%m%d_%H%M%S.toml")
        .to_string();
    std::fs::write(output_file, toml_str).unwrap();
}
