mod beam_search;
mod local_search;
mod random_walk;
mod sequential;
mod simulated_annealing;
mod specific;
mod tabu_search;

pub use beam_search::BeamSearch;
pub use local_search::LocalSearch;
pub use random_walk::RandomWalk;
pub use sequential::Sequential;
pub use simulated_annealing::{BangBangSimulatedAnnealing, SimulatedAnnealing};
pub use specific::BreakoutLocalSearchForMaxCut;
pub use tabu_search::TabuSearch;

use crate::error::OptError;
use crate::search_state::{ProblemTrait, SearchState};
use serde::Serialize;

/// ヒューリスティックアルゴリズムの共通インターフェース。
///
/// `is_done` で終了条件を判定し、`run_once` で1ステップ実行します。
/// `run` は終了まで `run_once` を繰り返すデフォルト実装を持ちます。
pub trait Heuristic<Problem: ProblemTrait> {
    fn clear(&mut self) {}
    fn is_done<'a>(&self, state: &SearchState<'a, Problem>) -> bool;
    fn run_once<'a>(
        &mut self,
        state: &mut SearchState<'a, Problem>,
    ) -> Result<(), OptError>;
    fn run<'a>(
        &mut self,
        state: &mut SearchState<'a, Problem>,
    ) -> Result<(), OptError> {
        self.clear();
        while !self.is_done(state) {
            self.run_once(state)?;
        }

        return Ok(());
    }
}

pub trait ParallelHeuristic<Problem: ProblemTrait>: Heuristic<Problem> {
    fn run_once_par<'a>(
        &mut self,
        state: &mut SearchState<'a, Problem>,
    ) -> Result<(), OptError> {
        self.run_once(state)
    }
    fn run_par<'a>(
        &mut self,
        state: &mut SearchState<'a, Problem>,
    ) -> Result<(), OptError> {
        self.clear();
        while !self.is_done(state) {
            self.run_once_par(state)?;
        }

        return Ok(());
    }
}

/// ヒューリスティックの終了条件。
///
/// 反復回数・実行時間・改善なし反復回数のいずれか（複数も可）で停止させます。
///
/// # Example
///
/// ```
/// use optopus::heuristic::StopCondition;
/// use std::time::Duration;
///
/// // 反復回数のみ
/// let sc = StopCondition::iterations(1_000_000);
///
/// // 実行時間のみ
/// let sc = StopCondition::duration(Duration::from_secs(30));
///
/// // 組み合わせ（チェーン）
/// let sc = StopCondition::iterations(1_000_000)
///     .with_duration(Duration::from_secs(30));
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct StopCondition {
    pub max_iteration: Option<u64>,
    pub max_duration: Option<std::time::Duration>,
    pub max_failed_update: Option<u64>,
}

impl StopCondition {
    /// 全条件を直接指定して作成します。
    pub fn new(
        max_iteration: Option<u64>,
        max_duration: Option<std::time::Duration>,
        max_failed_update: Option<u64>,
    ) -> Self {
        Self {
            max_iteration,
            max_duration,
            max_failed_update,
        }
    }

    /// 反復回数だけを条件にした `StopCondition` を作成します。
    pub fn iterations(n: u64) -> Self {
        Self {
            max_iteration: Some(n),
            max_duration: None,
            max_failed_update: None,
        }
    }

    /// 実行時間だけを条件にした `StopCondition` を作成します。
    pub fn duration(d: std::time::Duration) -> Self {
        Self {
            max_iteration: None,
            max_duration: Some(d),
            max_failed_update: None,
        }
    }

    /// 改善なし反復回数だけを条件にした `StopCondition` を作成します。
    pub fn failed_updates(n: u64) -> Self {
        Self {
            max_iteration: None,
            max_duration: None,
            max_failed_update: Some(n),
        }
    }

    /// 最大反復回数を追加（チェーン用）。
    pub fn with_iterations(mut self, n: u64) -> Self {
        self.max_iteration = Some(n);
        self
    }

    /// 最大実行時間を追加（チェーン用）。
    pub fn with_duration(mut self, d: std::time::Duration) -> Self {
        self.max_duration = Some(d);
        self
    }

    /// 最大改善なし反復回数を追加（チェーン用）。
    pub fn with_failed_updates(mut self, n: u64) -> Self {
        self.max_failed_update = Some(n);
        self
    }

    pub fn is_done<'a, Problem: ProblemTrait>(&self, state: &SearchState<'a, Problem>) -> bool {
        if let Some(max_iter) = self.max_iteration {
            if state.iteration - state.start_iteration >= max_iter {
                return true;
            }
        }
        if let Some(max_duration) = self.max_duration {
            if state.duration() >= max_duration {
                return true;
            }
        }
        if let Some(max_failed_update) = self.max_failed_update {
            if state.iteration - state.best_iteration >= max_failed_update {
                return true;
            }
        }
        return false;
    }
}
