//! プロファイリング: LocalSearch × SAT × Flip + Swap (Tier 2)
//!
//! ホットパス (Flip):
//!   - apply_to_solution: Vec<usize> (affected 変数) alloc + sort_unstable + dedup
//!   - calc_gain() を affected 変数ごとに呼び出し (節走査)
//!
//! ホットパス (Swap):
//!   - iter() 全体を Vec<SatSwapNeighbor> に eager collect
//!   - HashSet<(usize,usize)> (seen ペア去重) を毎回 alloc
//!   - calc_gain_with_virtual_flip を節ペアごとに呼び出し
//!
//! LocalSearch は局所最適で停止するため Restart で固定時間走らせる。
//!
//! 実行方法:
//! ```
//! cargo build --profile profiling --example prof_ls_sat
//! samply record ./target/profiling/examples/prof_ls_sat
//! ```

use std::time::Duration;

use optopus::prelude::*;

fn main() {
    // 3-SAT インスタンスを構築: n=500 変数, ~2000 節 (密度 ~4.0)
    // clause_ratio ~4.0 は位相遷移付近で gain 更新が頻繁に発生する
    let n_vars = 500usize;
    let n_clauses = 2000usize;
    let mut sat = Sat::new(n_vars);
    for k in 0..n_clauses {
        let a = ((k * 7 + 1) % n_vars) as i64 + 1;
        let b = ((k * 13 + 3) % n_vars) as i64 + 1;
        let c = ((k * 19 + 5) % n_vars) as i64 + 1;
        let lit_a = if k % 3 == 0 { -a } else { a };
        let lit_b = if k % 5 == 0 { -b } else { b };
        sat.add_clause([lit_a, lit_b, c]);
    }

    // --- Flip ---
    let mut state = SearchState::new(&sat);
    Restart::new(
        StopCondition::duration(Duration::from_secs(3)),
        Box::new(LocalSearch::<SatFlipNeighbor>::new(
            StopCondition::failed_updates(1),
        )),
        StopCondition::failed_updates(1),
    )
    .run(&mut state)
    .unwrap();

    // --- Swap ---
    let mut state2 = SearchState::new(&sat);
    Restart::new(
        StopCondition::duration(Duration::from_secs(3)),
        Box::new(LocalSearch::<SatSwapNeighbor>::new(
            StopCondition::failed_updates(1),
        )),
        StopCondition::failed_updates(1),
    )
    .run(&mut state2)
    .unwrap();
}
