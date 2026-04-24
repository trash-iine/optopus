//! プロファイリング: LocalSearch × FormulaProblem × Flip + Swap (Tier 2)
//!
//! ホットパス (Flip):
//!   - apply_to_solution: interaction_neighbors[i] のみ gain 再計算
//!   - calc_gain_fast: eval_poly_delta の多項式評価
//!
//! ホットパス (Swap):
//!   - iter(): sol.x.clone() + sol.constraint_vals.clone() + Vec<Neighbor>
//!     を全 O(n²) ペアで eager 構築
//!   - 各 i ごとに delta_i: Vec<Value> を制約数に線形で構築
//!
//! LocalSearch は局所最適で停止するため Restart で固定時間走らせる。
//!
//! 実行方法:
//! ```
//! cargo build --profile profiling --example prof_ls_formula
//! samply record ./target/profiling/examples/prof_ls_formula
//! ```

use std::time::Duration;

use optopus::prelude::*;

fn main() {
    // n=60 変数の MaxCut 相当 formula: sum_{i<j} x[i]*x[j]
    // 全ペアに交差項があるため interaction_neighbors が密になり gain 再計算コストが高い
    let n = 60usize;
    let objective = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| Expr::Var(i) * Expr::Var(j)))
        .reduce(|acc, e| acc + e)
        .unwrap();

    let prob = FormulaProblem::new(n, objective, OptDirection::Maximize, vec![]);

    // --- Flip ---
    let mut state = SearchState::new(&prob);
    Restart::new(
        StopCondition::duration(Duration::from_secs(3)),
        Box::new(LocalSearch::<FormulaFlipNeighbor>::new(
            StopCondition::failed_updates(1),
        )),
        StopCondition::failed_updates(1),
    )
    .run(&mut state)
    .unwrap();

    // --- Swap ---
    let mut state2 = SearchState::new(&prob);
    Restart::new(
        StopCondition::duration(Duration::from_secs(3)),
        Box::new(LocalSearch::<FormulaSwapNeighbor>::new(
            StopCondition::failed_updates(1),
        )),
        StopCondition::failed_updates(1),
    )
    .run(&mut state2)
    .unwrap();
}
