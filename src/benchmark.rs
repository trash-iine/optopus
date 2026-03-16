use crate::{
    heuristic::{
        BreakoutLocalSearchForMaxCut, Heuristic, LocalSearch, SimulatedAnnealing, StopCondition,
        TabuSearch,
    },
    problem::{max_cut::MaxCut, MaxCutFlipNeighbor},
    search_state::SearchState,
};
use serde::Serialize;

#[derive(Clone, Serialize)]
pub struct SimulatedAnnealingSetting {
    pub initial_temperature: f64,
    pub cooling_rate: f64,
}

#[derive(Clone, Serialize)]
pub struct TabuSearchSetting {
    pub tabu_tenure: (u64, u64),
}

#[derive(Clone, Serialize)]
pub struct BreakoutLocalSearchSetting {
    pub tabu_tenure: (u64, u64),
    pub t: u64,
    pub l0: u64,
    pub p0: f64,
    pub q: f64,
}

#[derive(Clone, Serialize)]
pub enum HeuristicSetting {
    SimulatedAnnealing(SimulatedAnnealingSetting),
    LocalSearch,
    TabuSearch(TabuSearchSetting),
    BreakoutLocalSearch(BreakoutLocalSearchSetting),
}

impl HeuristicSetting {
    pub fn build(&self, cond: StopCondition) -> Box<dyn Heuristic<MaxCut>> {
        match self {
            HeuristicSetting::SimulatedAnnealing(s) => {
                Box::new(SimulatedAnnealing::<MaxCutFlipNeighbor>::new(
                    cond,
                    s.initial_temperature,
                    s.cooling_rate,
                ))
            }
            HeuristicSetting::LocalSearch => {
                Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(cond))
            }
            HeuristicSetting::TabuSearch(s) => Box::new(TabuSearch::<MaxCutFlipNeighbor>::new(
                cond,
                s.tabu_tenure,
                None,
            )),
            HeuristicSetting::BreakoutLocalSearch(s) => Box::new(
                BreakoutLocalSearchForMaxCut::new(s.tabu_tenure, cond, s.t, s.l0, s.p0, s.q),
            ),
        }
    }
}

#[derive(Clone, Serialize)]
pub enum InstanceType {
    MaxCut,
}

/// 1回の実行に必要な設定（再現実験のための情報をすべて含む）
#[derive(Clone, Serialize)]
pub struct BenchmarkSetting {
    pub instance_path: String,
    pub instance_type: InstanceType,
    pub heuristic: HeuristicSetting,
    pub stop_condition: StopCondition,
}

/// 1回の実行結果（再現に必要な設定 + 評価値）
#[derive(Serialize)]
pub struct BenchmarkResult {
    /// 再現実験に必要な設定（インスタンス・ヒューリスティック・停止条件）
    pub setting: BenchmarkSetting,
    /// 実行ステータス ("success" または "error: ...")
    pub status: String,
    /// 最良解の目的関数値
    pub best_objective: f64,
    /// 最良解が得られたイテレーション
    pub best_iteration: u64,
    /// 最良解が得られるまでの経過時間（秒）
    pub time_to_best_secs: f64,
    /// 実行全体の経過時間（秒）
    pub total_time_secs: f64,
    /// 最良解のソリューション（カット側に属する頂点のリスト）
    pub solution: Vec<usize>,
}

pub struct Benchmark {
    pub results: Vec<BenchmarkResult>,
}

impl Benchmark {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    pub fn run(&mut self, setting: BenchmarkSetting) {
        let result = match setting.instance_type {
            InstanceType::MaxCut => run_max_cut(setting),
        };
        self.results.push(result);
    }
}

fn run_max_cut(setting: BenchmarkSetting) -> BenchmarkResult {
    let mc = match MaxCut::load_from_file(&setting.instance_path) {
        Ok(mc) => mc,
        Err(e) => {
            return BenchmarkResult {
                setting,
                status: format!("error loading instance: {}", e),
                best_objective: 0.0,
                best_iteration: 0,
                time_to_best_secs: 0.0,
                total_time_secs: 0.0,
                solution: Vec::new(),
            };
        }
    };

    let mut state = SearchState::new(&mc, rand::rng());
    let heuristic = setting.heuristic.build(setting.stop_condition.clone());

    let start = std::time::Instant::now();
    let status = heuristic.run(&mut state);
    let total_time = start.elapsed();

    BenchmarkResult {
        setting,
        status: match status {
            Ok(_) => "success".to_string(),
            Err(e) => format!("error: {}", e),
        },
        best_objective: state.best_solution.objective as f64,
        best_iteration: state.best_iteration,
        time_to_best_secs: (state.best_time - state.start_time).as_secs_f64(),
        total_time_secs: total_time.as_secs_f64(),
        solution: state
            .best_solution
            .cut
            .iter()
            .filter(|&(_, &v)| v)
            .map(|(&i, _)| i)
            .collect(),
    }
}
