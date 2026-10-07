//! The permutation flow shop written as an `IntegerProblem`, next to the
//! built-in `FlowShop` it could have been.
//!
//! The variables are a permutation of the jobs and the objective is
//! `FlowShop::makespan` of that order, a closure and nothing else.
//! Simulated annealing on `IntSwapNeighbor` searches it, pricing every
//! candidate with a full O(nm) schedule.
//!
//! The built-in problem prices an insertion at every place of the sequence in
//! O(nm) together, with Taillard's heads and tails, and that is what
//! Iterated Greedy needs. Both get the same time here, and the gap between them
//! is what the dedicated problem buys.
//!
//! Run with:
//! ```
//! cargo run --release --example integer_flow_shop
//! ```

use std::time::Duration;

use optopus::prelude::*;

fn main() {
    let fs = FlowShop::load_file("data/instances/flow_shop/taillard/ta051.txt").unwrap();
    let budget = StopCondition::duration(Duration::from_secs(2));
    println!(
        "{}: {} jobs, {} machines",
        fs.name, fs.n_jobs, fs.n_machines
    );
    println!("NEH = {}", fs.neh().objective);

    let fs_ref = &fs;
    let prob = IntegerProblem::minimize(IntVars::permutation(fs.n_jobs), move |v: &[i64]| {
        let order: Vec<usize> = v.iter().map(|&j| j as usize).collect();
        f64::from(fs_ref.makespan(&order))
    });
    let mut state = SearchState::new_with_seed(&prob, 42);
    SimulatedAnnealing::<IntSwapNeighbor>::new(budget.clone(), 20.0, 0.99995)
        .run(&mut state)
        .unwrap();
    println!(
        "IntegerProblem + SA       makespan = {} after {} iterations",
        state.best_solution.evaluate().minimized(),
        state.iteration
    );

    let mut state = SearchState::new_with_seed(&fs, 42);
    iterated_greedy_for_flow_shop(budget, 4, 0.4)
        .run(&mut state)
        .unwrap();
    println!(
        "FlowShop + IteratedGreedy makespan = {} after {} iterations",
        state.best_solution.objective, state.iteration
    );
}
