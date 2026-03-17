/// よく使う型・トレイトをまとめてインポートするためのモジュール。
///
/// ```rust
/// use optopus::prelude::*;
/// ```
///
/// で以下のすべてが使えるようになります。

// エラー型
pub use crate::error::OptError;

// 探索状態
pub use crate::search_state::{SearchState, SearchStateCloneType};

// ヒューリスティック
pub use crate::heuristic::{
    BangBangSimulatedAnnealing, BreakoutLocalSearchForMaxCut, Heuristic, LocalSearch,
    ParallelHeuristic, RandomWalk, Sequential, SimulatedAnnealing, StopCondition, TabuSearch,
};

// トレイト
pub use crate::search_state::{
    EnabledTabu, Evaluable, MoveToNeigbor, ProblemTrait, Rankable,
};

// 問題型と近傍型
pub use crate::problem::{
    // MaxCut
    MaxCut,
    MaxCutFlipNeighbor,
    MaxCutSwapNeighbor,
    // SAT
    Sat,
    SatFlipNeighbor,
    SatSolution,
    SatSwapNeighbor,
    // QUBO
    Qubo,
    QuboFlipNeighbour,
    QuboSolution,
    QuboSwapNeighbour,
    // TSP
    TspRelocateNeighbor,
    TspSolution,
    TspTour,
    TspTwoOptNeighbor,
    TspWithCoordinates,
    // Formula
    Constraint,
    ConstraintRel,
    Expr,
    FormulaProblem,
    FormulaSolution,
    FormulaFlipNeighbor,
    FormulaSwapNeighbor,
    OptDirection,
};
