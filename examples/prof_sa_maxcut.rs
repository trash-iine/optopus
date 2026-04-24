//! プロファイリング: SimulatedAnnealing × MaxCut × Flip + Swap (Tier 1)
//!
//! ホットパス (Flip):
//!   - iter().choose() が reservoir sampling で O(n) 全消費
//!   - boltzmann_accept: f64::exp + rng.random() (悪化時のみ)
//!   - MaxCutSolution.gain: Vec<f32> インデックスアクセス
//!
//! ホットパス (Swap):
//!   - iter() が全 (i,j) ペアを Vec として collect → O(n²) alloc
//!   - その後 choose() が O(n²) 全消費 — Flip と比較してスケーリングを確認
//!
//! 実行方法:
//! ```
//! cargo build --profile profiling --example prof_sa_maxcut
//! samply record ./target/profiling/examples/prof_sa_maxcut
//! ```

use std::time::Duration;

use optopus::prelude::*;

fn main() {
    let mc = MaxCut::new(Graph::load_from_file("data/max_cut/G22").unwrap());

    // --- SA × Flip: O(n) iter().choose() ---
    let mut state = SearchState::new(&mc);
    SimulatedAnnealing::<MaxCutFlipNeighbor>::new(
        StopCondition::duration(Duration::from_secs(3)),
        100.0,
        0.9999,
    )
    .run(&mut state)
    .unwrap();

    // --- SA × Swap: O(n²) iter().choose() ---
    let mut state2 = SearchState::new(&mc);
    SimulatedAnnealing::<MaxCutSwapNeighbor>::new(
        StopCondition::duration(Duration::from_secs(3)),
        100.0,
        0.9999,
    )
    .run(&mut state2)
    .unwrap();
}
