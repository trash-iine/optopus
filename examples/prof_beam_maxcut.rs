//! プロファイリング: BeamSearch × MaxCut × Flip (Tier 1)
//!
//! ホットパス:
//!   - beam_width × n 回の MaxCutSolution::clone()
//!     (Vec<bool> cut + Vec<f32> gain の memcpy — n に線形)
//!   - 各 candidate に対する apply_to_solution (O(degree))
//!   - select_nth_unstable_by による beam 絞り込み
//!
//! アロケーション解析には cargo-instruments が有効:
//! ```
//! cargo instruments --profile profiling --example prof_beam_maxcut -t Allocations
//! ```
//!
//! 実行方法 (samply):
//! ```
//! cargo build --profile profiling --example prof_beam_maxcut
//! samply record ./target/profiling/examples/prof_beam_maxcut
//! ```

use std::time::Duration;

use optopus::prelude::*;

fn main() {
    // G22: n=2000 — beam_width × n = 5 × 2000 clones/iter でアロケーション圧を最大化
    let mc = MaxCut::new(Graph::load_from_file("data/max_cut/G22").unwrap());
    let mut state = SearchState::new(&mc);
    BeamSearch::<MaxCut, MaxCutFlipNeighbor>::new(
        StopCondition::duration(Duration::from_secs(5)),
        5,
    )
    .run(&mut state)
    .unwrap();
}
