//! プロファイリング: Lin-Kernighan (Helsgaun) × TSP (Tier 1, LKH ベースライン)
//!
//! ホットパス:
//!   - ensure_candidates(): 初回のみ O(n²) の全ペア距離計算 + sort (候補リスト構築)
//!   - find_lk_move(): depth-bounded DFS — tour 上で succ/pred を辿りながら
//!     improving なエッジ交換シーケンスを探索
//!   - prob.distance(): f64::sqrt (座標から毎回再計算)
//!   - apply_to_solution(): tour の部分反転 + gain HashMap の差分更新
//!
//! LKH は局所最適で `no_improvement = true` になり自己終了するため、
//! Restart でラップして固定時間走らせる (berlin52 は非常に速く収束するため必要)。
//!
//! 実行方法:
//! ```
//! cargo build --profile profiling --example prof_lkh_tsp
//! samply record ./target/profiling/examples/prof_lkh_tsp
//! ```

use std::time::Duration;

use optopus::prelude::*;

fn main() {
    let tsp = TspWithCoordinates::load_file("data/tsp/berlin52.tsp").unwrap();
    let mut state = SearchState::new(&tsp);

    Restart::new(
        StopCondition::duration(Duration::from_secs(5)),
        Box::new(LinKernighanHelsgottForTsp::new(
            StopCondition::iterations(u64::MAX),
            5,
            5,
        )),
        StopCondition::failed_updates(1),
    )
    .run(&mut state)
    .unwrap();
}
