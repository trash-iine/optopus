//! Max Cut 問題のサンプル。
//!
//! LocalSearch と TabuSearch を同じ問題インスタンスに適用して結果を比較します。
//!
//! 実行方法:
//! ```
//! cargo run --example max_cut
//! ```

use optopus::prelude::*;

fn main() {
    // 小さな手動グラフを作成（ファイル不要）
    let mc = MaxCut::new(Graph::from_edges([
        (0, 1, 1.0),
        (0, 2, 1.0),
        (0, 3, 1.0),
        (1, 2, 1.0),
        (1, 4, 1.0),
        (2, 5, 1.0),
        (3, 4, 1.0),
        (3, 5, 1.0),
        (4, 5, 1.0),
    ]));

    // --- Local Search ---
    let mut state = SearchState::new(&mc);
    let mut ls = LocalSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(100_000));
    ls.run(&mut state).unwrap();
    println!(
        "[LocalSearch]  best objective = {:.1} (iter {})",
        state.best_solution.objective, state.best_iteration
    );

    // --- Tabu Search ---
    let mut state = SearchState::new(&mc);
    let mut ts = TabuSearch::<MaxCutFlipNeighbor>::new(
        StopCondition::iterations(100_000),
        (3, 7),
        None,
    );
    ts.run(&mut state).unwrap();
    println!(
        "[TabuSearch]   best objective = {:.1} (iter {})",
        state.best_solution.objective, state.best_iteration
    );
}
