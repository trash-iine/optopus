//! プロファイリング: TabuSearch × MaxCut × Flip (Tier 1)
//!
//! ホットパス:
//!   - filter_best() が vec![r] で毎回 Vec 再確保
//!   - is_move_enabled() が TabuMap (HashMap<usize, u64>)::get × n
//!   - add_to_tabu_map() の HashMap::insert
//!   - MaxCutSolution.gain: Vec<f32> を全頂点走査
//!
//! 実行方法:
//! ```
//! cargo build --profile profiling --example prof_ts_maxcut_flip
//! samply record ./target/profiling/examples/prof_ts_maxcut_flip
//! ```

use std::time::Duration;

use optopus::prelude::*;

fn main() {
    let mc = MaxCut::new(Graph::load_from_file("data/max_cut/G22").unwrap());
    let mut state = SearchState::new(&mc);
    TabuSearch::<MaxCutFlipNeighbor>::new(
        StopCondition::duration(Duration::from_secs(5)),
        (3, 10),
        None,
    )
    .run(&mut state)
    .unwrap();
}
