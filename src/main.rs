use optopus::{
    benchmark::{
        Benchmark, BenchmarkSetting, BreakoutLocalSearchSetting, MaxCutBenchmarkSetting,
        MaxCutHeuristicSetting,
    },
    heuristic::StopCondition,
    problem::max_cut::MaxCut,
};

fn main() {
    tracing_subscriber::fmt::init();

    let mut benchmark = Benchmark::new();

    let files = glob::glob("data/max_cut/G1*").unwrap();
    for file in files {
        let path = file.unwrap();
        let instance_number = path
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

        let instance_path = path.to_str().unwrap().to_string();
        tracing::info!("Processing file: {}", instance_path);

        let mc = MaxCut::load_from_file(&instance_path).unwrap();
        let n = mc.len();

        let setting = BenchmarkSetting::MaxCut(MaxCutBenchmarkSetting {
            instance_path,
            heuristic: MaxCutHeuristicSetting::BreakoutLocalSearch(BreakoutLocalSearchSetting {
                tabu_tenure: (3, (n / 10) as u64),
                t: 1000,
                l0: (n / 100) as u64,
                p0: 0.8,
                q: 0.5,
            }),
            stop_condition: StopCondition::new(Some(10_000_000), None, None),
        });

        benchmark.run(setting);
    }

    #[derive(serde::Serialize)]
    struct ResultWrapper {
        results: Vec<optopus::benchmark::BenchmarkResult>,
    }
    let toml_str = toml::to_string(&ResultWrapper { results: benchmark.results }).unwrap();
    let output_file = chrono::Local::now()
        .format("result/%Y%m%d_%H%M%S.toml")
        .to_string();
    std::fs::write(output_file, toml_str).unwrap();
}
