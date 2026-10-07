//! Variable Neighborhood Descent composed from `Sequential` and two
//! `LocalSearch` runs over different moves.
//!
//! A schedule that no swap improves may still be improved by moving one
//! operation elsewhere, and the other way round. `Sequential` runs the two
//! descents in turn and starts over from the first once the list is done.
//!
//! What makes it stop at the right place is the outer budget,
//! `failed_updates(2)`. A descent that finds no improving move still charges
//! one iteration for the pass that showed it, so two passes in a row without a
//! new best mean both neighborhoods are exhausted, and that is exactly when a
//! variable neighborhood descent ends.
//!
//! Run with:
//! ```
//! cargo run --release --example compose_vnd_pipeline
//! ```

use optopus::prelude::*;

fn main() {
    let jssp = JobShopScheduling::load_file("data/instances/jssp/orlib/ft10.txt").unwrap();
    let descent = StopCondition::iterations(100_000);

    let mut state = SearchState::new_with_seed(&jssp, 42);
    LocalSearch::<JobShopSwapNeighbor>::new(descent.clone())
        .run(&mut state)
        .unwrap();
    report("swap only", &state);

    let mut state = SearchState::new_with_seed(&jssp, 42);
    Sequential::<JobShopScheduling>::new(
        StopCondition::failed_updates(2),
        vec![
            Box::new(LocalSearch::<JobShopSwapNeighbor>::new(descent.clone())),
            Box::new(LocalSearch::<JobShopRelocateNeighbor>::new(descent)),
        ],
    )
    .run(&mut state)
    .unwrap();
    report("VND", &state);
}

fn report(name: &str, state: &SearchState<JobShopScheduling>) {
    println!(
        "{name:>10}: makespan = {} (after {} iterations)",
        state.best_solution.objective, state.iteration
    );
}
