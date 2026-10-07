//! Maximum independent set written as a QUBO.
//!
//! One binary variable per vertex says whether it is in the set. The energy
//! rewards every chosen vertex with `-1` and charges `2` for every edge with
//! both ends chosen. Dropping one end of such an edge removes the charge and
//! loses one reward, a net gain of at least one, so every minimum is an
//! independent set and its energy is minus its size. The penalty only has to
//! exceed the reward, any value above `1` would do.
//!
//! Nothing problem-specific is written here. The QUBO is a list of entries,
//! and the moves and the heuristics are the ones every QUBO gets.
//!
//! Run with:
//! ```
//! cargo run --release --example qubo_max_independent_set
//! ```

use optopus::prelude::*;

const PENALTY: i32 = 2;

fn main() {
    let graph = Graph::erdos_renyi(300, 0.05, &mut seeded_rng(11));
    let n = graph.num_vertices();
    let edges: Vec<(usize, usize)> = graph.edges().map(|(u, v, _)| (u, v)).collect();

    let qubo = Qubo::from_entries(
        (0..n)
            .map(|v| (v, v, -1))
            .chain(edges.iter().map(|&(u, v)| (u, v, PENALTY))),
    );

    let mut state = SearchState::new_with_seed(&qubo, 42);
    TabuSearch::<QuboFlipNeighbor>::new(StopCondition::iterations(50_000), (5, 20))
        .run(&mut state)
        .unwrap();

    let x = &state.best_solution.x;
    let size = (0..n).filter(|&v| x[v]).count();
    let violated = edges.iter().filter(|&&(u, v)| x[u] && x[v]).count();
    println!("independent set size = {size}");
    println!("energy = {}", state.best_solution.objective);
    println!("edges with both ends chosen = {violated}");
}
