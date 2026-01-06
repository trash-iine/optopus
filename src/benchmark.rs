use crate::{
    heuristic::{
        BreakoutLocalSearchForMaxCut, Heuristic, LocalSearch, ParallelHeuristic,
        SimulatedAnnealing, StopCondition, TabuSearch,
    },
    problem::{max_cut::MaxCut, MaxCutFlipNeighbor, MaxCutSwapNeighbor},
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
pub enum HeuristicSetting {
    SimulatedAnnealing(SimulatedAnnealingSetting),
    LocalSearch,
    TabuSearch(TabuSearchSetting),
}

impl HeuristicSetting {
    pub fn build(&self, cond: StopCondition) -> Box<dyn Heuristic<MaxCut>> {
        match self {
            HeuristicSetting::SimulatedAnnealing(setting) => {
                Box::new(SimulatedAnnealing::<MaxCutFlipNeighbor>::new(
                    cond,
                    setting.initial_temperature,
                    setting.cooling_rate,
                ))
            }
            HeuristicSetting::LocalSearch => Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(cond)),
            HeuristicSetting::TabuSearch(tabu_search_setting) => todo!(),
        }
    }
}

#[derive(Serialize)]
pub struct BenchmarkSetting {
    pub instance: String,
    pub heuristics: Vec<HeuristicSetting>,
    pub stop_condition: StopCondition,
}

#[derive(Serialize)]
pub struct BenchmarkResult {
    status: String,
    objective: f64,
    iteration: u64,
    time_taken: f64,
    solution: Vec<usize>,
    heuristic: HeuristicSetting,
}

#[derive(Serialize)]
pub struct Benchmark {
    pub instance: String,
    pub results: Vec<BenchmarkResult>,
    pub setting: BenchmarkSetting,
}
