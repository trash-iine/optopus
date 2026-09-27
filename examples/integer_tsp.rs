//! TSP written as an `IntegerProblem` over a permutation.
//!
//! Variable `p` is the city visited `p`th, and the objective is the length of
//! the closed tour. `IntReverseNeighbor` reverses a stretch of the tour, which
//! is a 2-opt move.
//!
//! Run with:
//! ```
//! cargo run --example integer_tsp
//! ```

use optopus::prelude::*;

fn main() {
    let tsp = Tsp::load_file("data/instances/tsp/eil51.tsp").unwrap();
    let n = tsp.get_n();

    let prob = IntegerProblem::minimize(IntVars::permutation(n), |tour: &[i64]| {
        (0..n)
            .map(|p| tsp.distance(tour[p] as usize, tour[(p + 1) % n] as usize))
            .sum()
    });

    let mut state = SearchState::new_with_seed(&prob, 42);
    SimulatedAnnealing::<IntReverseNeighbor>::new(
        StopCondition::iterations(200_000),
        10.0,
        0.99997,
    )
    .run(&mut state)
    .unwrap();
    println!("tour = {:?}", state.best_solution.values());
    println!("length = {:?}", state.best_solution.evaluate());
}
