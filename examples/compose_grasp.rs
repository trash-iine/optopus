//! GRASP composed from `Iterated`, a local search and
//! `GreedyRandomizedConstruction`.
//!
//! GRASP repeats two phases until the budget is spent. A randomized greedy
//! construction builds a solution from nothing, each element drawn from the
//! ones whose cheapest placement is within `alpha` of the cheapest of all, and
//! a local search takes it to a local optimum. `Iterated` already alternates a
//! search with a second step and keeps the best of every cycle, so with the
//! construction as that second step it is GRASP, and nothing else is written.
//! The construction builds one solution per run whatever its stop condition,
//! so it is given an empty one.
//!
//! The construction asks only for `Ruinable`, so the same composition runs on
//! a flow shop and on bin packing. On the bin packing the local search is
//! itself a composition, a variable neighborhood descent over relocations and
//! exchanges. Each is compared with multi-start descent from random solutions
//! on the same budget.
//!
//! Run with:
//! ```
//! cargo run --release --example compose_grasp
//! ```

use std::time::Duration;

use optopus::prelude::*;

fn main() {
    let budget = StopCondition::duration(Duration::from_secs(2));

    let fs = FlowShop::load_file("data/instances/flow_shop/taillard/ta051.txt").unwrap();
    println!(
        "{}: {} jobs, {} machines",
        fs.name, fs.n_jobs, fs.n_machines
    );
    let descent = || -> Box<dyn Heuristic<FlowShop>> {
        Box::new(LocalSearch::<FlowShopInsertNeighbor>::new(
            StopCondition::iterations(u64::MAX),
        ))
    };
    let mut state = SearchState::new_with_seed(&fs, 42);
    Restart::<FlowShop>::new(budget.clone(), descent(), StopCondition::failed_updates(0))
        .run(&mut state)
        .unwrap();
    println!("  multi-start: makespan {}", state.best_solution.objective);
    let mut state = SearchState::new_with_seed(&fs, 42);
    Iterated::<FlowShop>::new(
        budget.clone(),
        descent(),
        Box::new(GreedyRandomizedConstruction::<FlowShop>::new(
            StopCondition::new(None, None, None),
            0.2,
        )),
    )
    .run(&mut state)
    .unwrap();
    println!("  GRASP:       makespan {}", state.best_solution.objective);

    let bp = BinPacking::load_file("data/instances/bin_packing/falkenauer/u250_00.txt").unwrap();
    println!(
        "{}: {} items, at least {} bins",
        bp.name,
        bp.num_items(),
        bp.lower_bound()
    );
    // Relocations until none helps, then exchanges, and again, until two
    // passes in a row find nothing.
    let vnd = || -> Box<dyn Heuristic<BinPacking>> {
        let until_stuck = || StopCondition::iterations(u64::MAX);
        Box::new(Sequential::<BinPacking>::new(
            StopCondition::failed_updates(2),
            vec![
                Box::new(LocalSearch::<BinPackingRelocateNeighbor>::new(until_stuck())),
                Box::new(LocalSearch::<BinPackingSwapNeighbor>::new(until_stuck())),
            ],
        ))
    };
    let mut state = SearchState::new_with_seed(&bp, 42);
    Restart::<BinPacking>::new(budget.clone(), vnd(), StopCondition::failed_updates(0))
        .run(&mut state)
        .unwrap();
    println!("  multi-start: {} bins", state.best_solution.num_bins());
    let mut state = SearchState::new_with_seed(&bp, 42);
    Iterated::<BinPacking>::new(
        budget.clone(),
        vnd(),
        Box::new(GreedyRandomizedConstruction::<BinPacking>::new(
            StopCondition::new(None, None, None),
            0.2,
        )),
    )
    .run(&mut state)
    .unwrap();
    println!("  GRASP:       {} bins", state.best_solution.num_bins());
}
