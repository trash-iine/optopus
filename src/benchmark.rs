use crate::{
    heuristic::{
        BreakoutLocalSearchForMaxCut, Heuristic, LocalSearch, SimulatedAnnealing, StopCondition,
        TabuSearch,
    },
    problem::{
        max_cut::MaxCut,
        qubo::Qubo,
        sat::Sat,
        tsp::TspWithCoordinates,
        MaxCutFlipNeighbor, MaxCutSwapNeighbor,
        QuboFlipNeighbour, QuboSwapNeighbour,
        sat::{SatFlipNeighbor, SatSwapNeighbor},
        tsp::{TspTwoOptNeighbor, TspRelocateNeighbor},
    },
    search_state::SearchState,
};
use serde::Serialize;

// ---------------------------------------------------------------------------
// 共通アルゴリズムパラメータ
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize)]
pub struct SimulatedAnnealingSetting {
    pub initial_temperature: f64,
    pub cooling_rate: f64,
}

#[derive(Clone, Serialize)]
pub struct TabuSearchSetting {
    pub tabu_tenure: (u64, u64),
}

/// BreakoutLocalSearch は MaxCut 専用
#[derive(Clone, Serialize)]
pub struct BreakoutLocalSearchSetting {
    pub tabu_tenure: (u64, u64),
    pub t: u64,
    pub l0: u64,
    pub p0: f64,
    pub q: f64,
}

// ---------------------------------------------------------------------------
// MaxCut
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize)]
pub enum MaxCutNeighborKind {
    Flip,
    Swap,
}

#[derive(Clone, Serialize)]
pub enum MaxCutHeuristicSetting {
    LocalSearch(MaxCutNeighborKind),
    TabuSearch(TabuSearchSetting, MaxCutNeighborKind),
    SimulatedAnnealing(SimulatedAnnealingSetting, MaxCutNeighborKind),
    BreakoutLocalSearch(BreakoutLocalSearchSetting),
}

impl MaxCutHeuristicSetting {
    fn build(&self, cond: StopCondition) -> Box<dyn Heuristic<MaxCut>> {
        match self {
            Self::LocalSearch(MaxCutNeighborKind::Flip) => {
                Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(cond))
            }
            Self::LocalSearch(MaxCutNeighborKind::Swap) => {
                Box::new(LocalSearch::<MaxCutSwapNeighbor>::new(cond))
            }
            Self::TabuSearch(s, MaxCutNeighborKind::Flip) => Box::new(
                TabuSearch::<MaxCutFlipNeighbor>::new(cond, s.tabu_tenure, None),
            ),
            Self::TabuSearch(s, MaxCutNeighborKind::Swap) => Box::new(
                TabuSearch::<MaxCutSwapNeighbor>::new(cond, s.tabu_tenure, None),
            ),
            Self::SimulatedAnnealing(s, MaxCutNeighborKind::Flip) => {
                Box::new(SimulatedAnnealing::<MaxCutFlipNeighbor>::new(
                    cond,
                    s.initial_temperature,
                    s.cooling_rate,
                ))
            }
            Self::SimulatedAnnealing(s, MaxCutNeighborKind::Swap) => {
                Box::new(SimulatedAnnealing::<MaxCutSwapNeighbor>::new(
                    cond,
                    s.initial_temperature,
                    s.cooling_rate,
                ))
            }
            Self::BreakoutLocalSearch(s) => Box::new(BreakoutLocalSearchForMaxCut::new(
                s.tabu_tenure,
                cond,
                s.t,
                s.l0,
                s.p0,
                s.q,
            )),
        }
    }
}

#[derive(Clone, Serialize)]
pub struct MaxCutBenchmarkSetting {
    pub instance_path: String,
    pub heuristic: MaxCutHeuristicSetting,
    pub stop_condition: StopCondition,
}

// ---------------------------------------------------------------------------
// QUBO
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize)]
pub enum QuboNeighborKind {
    Flip,
    Swap,
}

#[derive(Clone, Serialize)]
pub enum QuboHeuristicSetting {
    LocalSearch(QuboNeighborKind),
    TabuSearch(TabuSearchSetting, QuboNeighborKind),
    SimulatedAnnealing(SimulatedAnnealingSetting, QuboNeighborKind),
}

impl QuboHeuristicSetting {
    fn build(&self, cond: StopCondition) -> Box<dyn Heuristic<Qubo>> {
        match self {
            Self::LocalSearch(QuboNeighborKind::Flip) => {
                Box::new(LocalSearch::<QuboFlipNeighbour>::new(cond))
            }
            Self::LocalSearch(QuboNeighborKind::Swap) => {
                Box::new(LocalSearch::<QuboSwapNeighbour>::new(cond))
            }
            Self::TabuSearch(s, QuboNeighborKind::Flip) => Box::new(
                TabuSearch::<QuboFlipNeighbour>::new(cond, s.tabu_tenure, None),
            ),
            Self::TabuSearch(s, QuboNeighborKind::Swap) => Box::new(
                TabuSearch::<QuboSwapNeighbour>::new(cond, s.tabu_tenure, None),
            ),
            Self::SimulatedAnnealing(s, QuboNeighborKind::Flip) => {
                Box::new(SimulatedAnnealing::<QuboFlipNeighbour>::new(
                    cond,
                    s.initial_temperature,
                    s.cooling_rate,
                ))
            }
            Self::SimulatedAnnealing(s, QuboNeighborKind::Swap) => {
                Box::new(SimulatedAnnealing::<QuboSwapNeighbour>::new(
                    cond,
                    s.initial_temperature,
                    s.cooling_rate,
                ))
            }
        }
    }
}

#[derive(Clone, Serialize)]
pub struct QuboBenchmarkSetting {
    pub instance_path: String,
    pub heuristic: QuboHeuristicSetting,
    pub stop_condition: StopCondition,
}

// ---------------------------------------------------------------------------
// SAT
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize)]
pub enum SatNeighborKind {
    Flip,
    Swap,
}

#[derive(Clone, Serialize)]
pub enum SatHeuristicSetting {
    LocalSearch(SatNeighborKind),
    TabuSearch(TabuSearchSetting, SatNeighborKind),
    SimulatedAnnealing(SimulatedAnnealingSetting, SatNeighborKind),
}

impl SatHeuristicSetting {
    fn build(&self, cond: StopCondition) -> Box<dyn Heuristic<Sat>> {
        match self {
            Self::LocalSearch(SatNeighborKind::Flip) => {
                Box::new(LocalSearch::<SatFlipNeighbor>::new(cond))
            }
            Self::LocalSearch(SatNeighborKind::Swap) => {
                Box::new(LocalSearch::<SatSwapNeighbor>::new(cond))
            }
            Self::TabuSearch(s, SatNeighborKind::Flip) => Box::new(
                TabuSearch::<SatFlipNeighbor>::new(cond, s.tabu_tenure, None),
            ),
            Self::TabuSearch(s, SatNeighborKind::Swap) => Box::new(
                TabuSearch::<SatSwapNeighbor>::new(cond, s.tabu_tenure, None),
            ),
            Self::SimulatedAnnealing(s, SatNeighborKind::Flip) => {
                Box::new(SimulatedAnnealing::<SatFlipNeighbor>::new(
                    cond,
                    s.initial_temperature,
                    s.cooling_rate,
                ))
            }
            Self::SimulatedAnnealing(s, SatNeighborKind::Swap) => {
                Box::new(SimulatedAnnealing::<SatSwapNeighbor>::new(
                    cond,
                    s.initial_temperature,
                    s.cooling_rate,
                ))
            }
        }
    }
}

#[derive(Clone, Serialize)]
pub struct SatBenchmarkSetting {
    pub instance_path: String,
    pub heuristic: SatHeuristicSetting,
    pub stop_condition: StopCondition,
}

// ---------------------------------------------------------------------------
// TSP
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize)]
pub enum TspNeighborKind {
    TwoOpt,
    Relocate,
}

#[derive(Clone, Serialize)]
pub enum TspHeuristicSetting {
    LocalSearch(TspNeighborKind),
    TabuSearch(TabuSearchSetting, TspNeighborKind),
    SimulatedAnnealing(SimulatedAnnealingSetting, TspNeighborKind),
}

impl TspHeuristicSetting {
    fn build(&self, cond: StopCondition) -> Box<dyn Heuristic<TspWithCoordinates>> {
        match self {
            Self::LocalSearch(TspNeighborKind::TwoOpt) => {
                Box::new(LocalSearch::<TspTwoOptNeighbor>::new(cond))
            }
            Self::LocalSearch(TspNeighborKind::Relocate) => {
                Box::new(LocalSearch::<TspRelocateNeighbor>::new(cond))
            }
            Self::TabuSearch(s, TspNeighborKind::TwoOpt) => Box::new(
                TabuSearch::<TspTwoOptNeighbor>::new(cond, s.tabu_tenure, None),
            ),
            Self::TabuSearch(s, TspNeighborKind::Relocate) => Box::new(
                TabuSearch::<TspRelocateNeighbor>::new(cond, s.tabu_tenure, None),
            ),
            Self::SimulatedAnnealing(s, TspNeighborKind::TwoOpt) => {
                Box::new(SimulatedAnnealing::<TspTwoOptNeighbor>::new(
                    cond,
                    s.initial_temperature,
                    s.cooling_rate,
                ))
            }
            Self::SimulatedAnnealing(s, TspNeighborKind::Relocate) => {
                Box::new(SimulatedAnnealing::<TspRelocateNeighbor>::new(
                    cond,
                    s.initial_temperature,
                    s.cooling_rate,
                ))
            }
        }
    }
}

#[derive(Clone, Serialize)]
pub struct TspBenchmarkSetting {
    pub instance_path: String,
    pub heuristic: TspHeuristicSetting,
    pub stop_condition: StopCondition,
}

// ---------------------------------------------------------------------------
// Master BenchmarkSetting
// ---------------------------------------------------------------------------

/// 1回の実行設定（再現実験に必要な全情報を含む）
#[derive(Clone, Serialize)]
pub enum BenchmarkSetting {
    MaxCut(MaxCutBenchmarkSetting),
    Qubo(QuboBenchmarkSetting),
    Sat(SatBenchmarkSetting),
    Tsp(TspBenchmarkSetting),
}

// ---------------------------------------------------------------------------
// BenchmarkResult
// ---------------------------------------------------------------------------

/// 1回の実行結果（設定 + 評価値）
#[derive(Serialize)]
pub struct BenchmarkResult {
    /// 再現実験に必要な設定（インスタンス・ヒューリスティック・停止条件）
    pub setting: BenchmarkSetting,
    /// 実行ステータス ("success" または "error: ...")
    pub status: String,
    /// 最良解の目的関数値 (MaxCut/SAT は最大化、QUBO/TSP は最小化)
    pub best_objective: f64,
    /// 最良解が得られたイテレーション
    pub best_iteration: u64,
    /// 最良解が得られるまでの経過時間（秒）
    pub time_to_best_secs: f64,
    /// 実行全体の経過時間（秒）
    pub total_time_secs: f64,
    /// 最良解のソリューション
    /// - MaxCut: カット側の頂点インデックス (0-indexed)
    /// - QUBO: 1 に割り当てられた変数インデックス (0-indexed)
    /// - SAT: true に割り当てられた変数インデックス (0-indexed)
    /// - TSP: 訪問順のインデックス列 (0-indexed)
    pub solution: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

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
        let result = match &setting {
            BenchmarkSetting::MaxCut(s) => run_max_cut(s),
            BenchmarkSetting::Qubo(s) => run_qubo(s),
            BenchmarkSetting::Sat(s) => run_sat(s),
            BenchmarkSetting::Tsp(s) => run_tsp(s),
        };
        // setting を result に移動させるため、run_* は BenchmarkSetting を内部で再構築する
        self.results.push(BenchmarkResult {
            setting,
            ..result
        });
    }
}

// ---------------------------------------------------------------------------
// Per-problem run functions
// ---------------------------------------------------------------------------

/// 共通のエラー結果を生成するヘルパー
fn error_result(e: impl std::fmt::Display) -> BenchmarkResult {
    BenchmarkResult {
        setting: BenchmarkSetting::MaxCut(MaxCutBenchmarkSetting {
            instance_path: String::new(),
            heuristic: MaxCutHeuristicSetting::LocalSearch(MaxCutNeighborKind::Flip),
            stop_condition: StopCondition::new(None, None, None),
        }),
        status: format!("error loading instance: {}", e),
        best_objective: 0.0,
        best_iteration: 0,
        time_to_best_secs: 0.0,
        total_time_secs: 0.0,
        solution: Vec::new(),
    }
}

fn run_max_cut(s: &MaxCutBenchmarkSetting) -> BenchmarkResult {
    let mc = match MaxCut::load_from_file(&s.instance_path) {
        Ok(v) => v,
        Err(e) => return error_result(e),
    };
    let mut state = SearchState::new(&mc);
    let mut heuristic = s.heuristic.build(s.stop_condition.clone());

    let start = std::time::Instant::now();
    let status = heuristic.run(&mut state);
    let total_time = start.elapsed();

    BenchmarkResult {
        setting: BenchmarkSetting::MaxCut(s.clone()), // 後で上書きされる
        status: status_str(status),
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

fn run_qubo(s: &QuboBenchmarkSetting) -> BenchmarkResult {
    let qubo = match Qubo::load_file_as_max_cut(&s.instance_path) {
        Ok(v) => v,
        Err(e) => return error_result(e),
    };
    let mut state = SearchState::new(&qubo);
    let mut heuristic = s.heuristic.build(s.stop_condition.clone());

    let start = std::time::Instant::now();
    let status = heuristic.run(&mut state);
    let total_time = start.elapsed();

    BenchmarkResult {
        setting: BenchmarkSetting::Qubo(s.clone()),
        status: status_str(status),
        best_objective: state.best_solution.objective as f64,
        best_iteration: state.best_iteration,
        time_to_best_secs: (state.best_time - state.start_time).as_secs_f64(),
        total_time_secs: total_time.as_secs_f64(),
        solution: state
            .best_solution
            .x
            .iter()
            .filter(|&(_, &v)| v)
            .map(|(&i, _)| i)
            .collect(),
    }
}

fn run_sat(s: &SatBenchmarkSetting) -> BenchmarkResult {
    let sat = match Sat::load_file(&s.instance_path) {
        Ok(v) => v,
        Err(e) => return error_result(e),
    };
    let mut state = SearchState::new(&sat);
    let mut heuristic = s.heuristic.build(s.stop_condition.clone());

    let start = std::time::Instant::now();
    let status = heuristic.run(&mut state);
    let total_time = start.elapsed();

    BenchmarkResult {
        setting: BenchmarkSetting::Sat(s.clone()),
        status: status_str(status),
        best_objective: state.best_solution.n_satisfied as f64,
        best_iteration: state.best_iteration,
        time_to_best_secs: (state.best_time - state.start_time).as_secs_f64(),
        total_time_secs: total_time.as_secs_f64(),
        solution: (0..sat.n_vars())
            .filter(|&i| state.best_solution.x[i])
            .collect(),
    }
}

fn run_tsp(s: &TspBenchmarkSetting) -> BenchmarkResult {
    let tsp = match TspWithCoordinates::load_file(&s.instance_path) {
        Ok(v) => v,
        Err(e) => return error_result(e),
    };
    let mut state = SearchState::new(&tsp);
    let mut heuristic = s.heuristic.build(s.stop_condition.clone());

    let start = std::time::Instant::now();
    let status = heuristic.run(&mut state);
    let total_time = start.elapsed();

    BenchmarkResult {
        setting: BenchmarkSetting::Tsp(s.clone()),
        status: status_str(status),
        best_objective: state.best_solution.objective,
        best_iteration: state.best_iteration,
        time_to_best_secs: (state.best_time - state.start_time).as_secs_f64(),
        total_time_secs: total_time.as_secs_f64(),
        solution: state.best_solution.tour.clone(),
    }
}

fn status_str(r: Result<(), crate::error::OptError>) -> String {
    match r {
        Ok(_) => "success".to_string(),
        Err(e) => format!("error: {}", e),
    }
}
