//! Beam Search サンプル。
//!
//! MaxCut 問題に対して BeamSearch・LocalSearch・TabuSearch を適用して結果を比較します。
//!
//! 実行方法:
//! ```
//! cargo run --example beam_search
//! ```

use optopus::prelude::*;

fn main() {
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

    let sc = StopCondition::iterations(100);

    // --- Beam Search (beam_width = 3) ---
    let mut state = SearchState::new(&mc);
    let mut bs = BeamSearch::<MaxCut, MaxCutFlipNeighbor>::new(sc.clone(), 3);
    bs.run(&mut state).unwrap();
    println!(
        "[BeamSearch w=3]  best = {:.1} (iter {})",
        state.best_solution.objective, state.best_iteration
    );

    // --- Beam Search (beam_width = 1 = LocalSearch に相当) ---
    let mut state = SearchState::new(&mc);
    let mut bs1 = BeamSearch::<MaxCut, MaxCutFlipNeighbor>::new(sc.clone(), 1);
    bs1.run(&mut state).unwrap();
    println!(
        "[BeamSearch w=1]  best = {:.1} (iter {})",
        state.best_solution.objective, state.best_iteration
    );

    // --- LocalSearch (比較用) ---
    let mut state = SearchState::new(&mc);
    let mut ls = LocalSearch::<MaxCutFlipNeighbor>::new(sc.clone());
    ls.run(&mut state).unwrap();
    println!(
        "[LocalSearch]     best = {:.1} (iter {})",
        state.best_solution.objective, state.best_iteration
    );
}
