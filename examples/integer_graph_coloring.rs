//! Graph coloring written as an `IntegerProblem`.
//!
//! One variable per vertex holds its color, one of `0..=k-1`. The objective is
//! the number of colors in use plus a penalty for every edge whose two ends
//! share a color. The penalty exceeds the number of vertices, so the best
//! solution is a proper coloring. `IntChangeNeighbor` recolors one vertex.
//!
//! Run with:
//! ```
//! cargo run --example integer_graph_coloring
//! ```

use optopus::prelude::*;

fn main() {
    let graph = Graph::erdos_renyi(50, 0.2, &mut seeded_rng(1));
    let n = graph.len();
    let k = (0..n).map(|v| graph.degree(v)).max().unwrap() as i64 + 1;

    let vars: IntVars = (0..n).map(|_| IntVar::new(0, k - 1)).collect();
    let prob = IntegerProblem::minimize(vars, |color: &[i64]| {
        let mut used = vec![false; k as usize];
        color.iter().for_each(|&c| used[c as usize] = true);
        let colors = used.iter().filter(|&&u| u).count();
        let conflicts = graph
            .edges()
            .filter(|&(i, j, _)| color[i] == color[j])
            .count();
        (colors + (n + 1) * conflicts) as f64
    });

    let mut state = SearchState::new_with_seed(&prob, 42);
    SimulatedAnnealing::<IntChangeNeighbor>::new(StopCondition::iterations(200_000), 2.0, 0.99997)
        .run(&mut state)
        .unwrap();
    println!("colors = {:?}", state.best_solution.values());
    println!("objective = {:?}", state.best_solution.evaluate());
}
