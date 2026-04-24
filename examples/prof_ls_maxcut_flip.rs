//! プロファイリング: LocalSearch × MaxCut × Flip (Tier 1 基準ベースライン)
//!
//! ホットパス:
//!   - MaxCutFlipNeighbor::iter (Graph::vertices の Vec iteration)
//!   - MaxCutSolution.gain: Vec<f32> の読み取り → filter → max_by
//!   - apply_to_solution: 隣接頂点のみ gain[] を更新 (O(degree))
//!
//! LocalSearch は局所最適で停止するため Restart で固定時間走らせて
//! サンプル密度を確保する。
//!
//! 実行方法 (samply):
//! ```
//! cargo build --profile profiling --example prof_ls_maxcut_flip
//! samply record ./target/profiling/examples/prof_ls_maxcut_flip
//! ```

use std::time::Duration;

use optopus::prelude::*;

fn main() {
    let mc = MaxCut::new(Graph::load_from_file("data/max_cut/G22").unwrap());
    let mut state = SearchState::new(&mc);

    Restart::new(
        StopCondition::duration(Duration::from_secs(5)),
        Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
            StopCondition::failed_updates(1),
        )),
        StopCondition::failed_updates(1),
    )
    .run(&mut state)
    .unwrap();
}
