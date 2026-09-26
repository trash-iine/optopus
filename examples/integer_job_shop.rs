//! Job shop scheduling written as an `IntegerProblem` over a permutation.
//!
//! The variables are a permutation of `0..jobs * machines`, and value `v` is
//! read as job `v / machines`, so every job appears once per machine. That
//! sequence of jobs is the operation order `JobShopScheduling::decode` turns
//! into a schedule, and its makespan is the objective. `IntSwapNeighbor`
//! exchanges two operations in the order.
//!
//! Run with:
//! ```
//! cargo run --example integer_job_shop
//! ```

use optopus::prelude::*;

fn main() {
    let jssp = JobShopScheduling::load_file("data/instances/jssp/ft06.txt").unwrap();
    let m = jssp.n_machines;

    let as_jobs = |v: &[i64]| -> Vec<usize> { v.iter().map(|&p| p as usize / m).collect() };
    let prob = IntegerProblem::minimize(IntVars::permutation(jssp.n_jobs * m), |v: &[i64]| {
        let (makespan, _) = jssp.decode(&as_jobs(v)).unwrap();
        makespan as f64
    });

    let mut state = SearchState::new_with_seed(&prob, 42);
    SimulatedAnnealing::<IntSwapNeighbor>::new(StopCondition::iterations(100_000), 5.0, 0.99995)
        .run(&mut state)
        .unwrap();
    println!("operations = {:?}", as_jobs(state.best_solution.values()));
    println!("makespan = {:?}", state.best_solution.evaluate());
}
