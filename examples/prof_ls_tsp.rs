//! プロファイリング: LocalSearch × TSP × TwoOpt + Relocate (Tier 1)
//!
//! ホットパス (TwoOpt):
//!   - O(n²) iter: normalize_edge_pair ((Edge,Edge)) をキーにした
//!     `sol.gain: HashMap<(Edge,Edge), f64>` の lookup
//!   - apply_to_solution: update_gains_for_removed/added_edge が
//!     セグメント長 (j - i + 1) に線形
//!
//! ホットパス (Relocate):
//!   - iter() は O(n²) 列挙を Vec に eager collect (Arc は不使用)
//!   - 各 (pos, ins) ペアで prob.distance() = f64::sqrt × 6 回
//!
//! LocalSearch は局所最適で停止するため Restart で固定時間走らせる。
//!
//! 実行方法:
//! ```
//! cargo build --profile profiling --example prof_ls_tsp
//! samply record ./target/profiling/examples/prof_ls_tsp
//! ```

use std::time::Duration;

use optopus::prelude::*;

fn main() {
    let tsp = TspWithCoordinates::load_file("data/tsp/berlin52.tsp").unwrap();

    // --- TwoOpt ---
    let mut state = SearchState::new(&tsp);
    Restart::new(
        StopCondition::duration(Duration::from_secs(3)),
        Box::new(LocalSearch::<TspTwoOptNeighbor>::new(
            StopCondition::failed_updates(1),
        )),
        StopCondition::failed_updates(1),
    )
    .run(&mut state)
    .unwrap();

    // --- Relocate ---
    let mut state2 = SearchState::new(&tsp);
    Restart::new(
        StopCondition::duration(Duration::from_secs(3)),
        Box::new(LocalSearch::<TspRelocateNeighbor>::new(
            StopCondition::failed_updates(1),
        )),
        StopCondition::failed_updates(1),
    )
    .run(&mut state2)
    .unwrap();
}
