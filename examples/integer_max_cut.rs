//! MaxCut written as an `IntegerProblem`.
//!
//! One binary variable per vertex says which side of the cut it is on, and the
//! objective adds up the weight of every edge whose two ends differ. There is
//! no move type and no trait impl, and `IntChangeNeighbor` flips one vertex.
//!
//! Run with:
//! ```
//! cargo run --example integer_max_cut
//! ```

use optopus::prelude::*;

fn main() {
    let graph = Graph::erdos_renyi(100, 0.1, &mut seeded_rng(1));

    let vars: IntVars = (0..graph.len()).map(|_| IntVar::binary()).collect();
    let prob = IntegerProblem::maximize(vars, |x: &[i64]| {
        graph
            .edges()
            .filter(|&(i, j, _)| x[i] != x[j])
            .map(|(_, _, w)| w as f64)
            .sum()
    });

    let mut state = SearchState::new_with_seed(&prob, 42);
    SimulatedAnnealing::<IntChangeNeighbor>::new(StopCondition::iterations(20_000), 2.0, 0.9995)
        .run(&mut state)
        .unwrap();
    println!("sides = {:?}", state.best_solution.values());
    println!("cut = {:?}", state.best_solution.evaluate());
}
