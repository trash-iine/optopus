//! プロファイリング: GeneticAlgorithm × MaxCut (Tier 2)
//!
//! ホットパス:
//!   - mutation.run(): LocalSearch sub-run (inner heuristic の全コスト)
//!   - insert_into_population(): O(population_size) worst-member scan
//!   - clone_for_new_run() + update_state(): SearchState clone コスト
//!   - tournament_select(): 4 回の random index アクセス
//!
//! 実行方法:
//! ```
//! cargo build --profile profiling --example prof_ga_maxcut
//! samply record ./target/profiling/examples/prof_ga_maxcut
//! ```

use std::time::Duration;

use optopus::prelude::*;
use optopus::problem::MaxCutUniformCrossover;

fn main() {
    let mc = MaxCut::new(Graph::load_from_file("data/max_cut/G22").unwrap());
    let mut state = SearchState::new(&mc);

    GeneticAlgorithm::new(
        StopCondition::duration(Duration::from_secs(5)),
        20,
        MaxCutUniformCrossover,
        Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
            StopCondition::failed_updates(1),
        )),
    )
    .run(&mut state)
    .unwrap();
}
