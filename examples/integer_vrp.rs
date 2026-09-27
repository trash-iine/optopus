//! Capacitated vehicle routing written as an `IntegerProblem` over a
//! permutation.
//!
//! The variables are a permutation of the customers, the order a single
//! vehicle would visit them in. `split_giant_tour` cuts that order into routes
//! at the best places, and the objective is the distance of those routes plus
//! the penalty for any load over capacity. `IntReverseNeighbor` reverses a
//! stretch of the order.
//!
//! Run with:
//! ```
//! cargo run --example integer_vrp
//! ```

use optopus::prelude::*;
use optopus::problem::vrp::split_giant_tour;

fn main() {
    let vrp = Vrp::load_file("data/instances/vrp/demo16.vrp").unwrap();

    // Customers are numbered from 1, the depot being 0.
    let routes = |order: &[i64]| {
        let giant: Vec<usize> = order.iter().map(|&c| c as usize + 1).collect();
        split_giant_tour(&vrp, &giant, vrp.penalty_weight())
    };
    let prob = IntegerProblem::minimize(IntVars::permutation(vrp.get_n()), |order: &[i64]| {
        vrp.solution_from_routes(routes(order)).objective
    });

    let mut state = SearchState::new_with_seed(&prob, 42);
    SimulatedAnnealing::<IntReverseNeighbor>::new(StopCondition::iterations(20_000), 10.0, 0.9998)
        .run(&mut state)
        .unwrap();
    println!("routes = {:?}", routes(state.best_solution.values()));
    println!("objective = {:?}", state.best_solution.evaluate());
}
