//! A small MaxCut solved to proven optimality by `BranchAndBound`, with
//! Breakout Local Search searching each node.
//!
//! The fixed vertices of a node are folded into one reference vertex, so the
//! node is itself a MaxCut and any MaxCut heuristic searches it. The bound is
//! the folding's constant plus the positive weights left, which is weak, so
//! the instance is kept to a few dozen vertices.
//!
//! Run with:
//! ```
//! cargo run --release --example max_cut_branch_and_bound
//! ```

use optopus::prelude::*;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .map_or(24, |a| a.parse().expect("the number of vertices"));
    let prob = MaxCut::new(Graph::erdos_renyi(n, 0.3, &mut seeded_rng(1)));

    let mut bnb = BranchAndBound::new(
        StopCondition::new(None, None, None),
        Box::new(bls_for_max_cut(
            StopCondition::iterations(200),
            (3, 10),
            20,
            5,
            0.8,
            0.5,
        )),
        BinaryRelaxation,
    );
    let mut state = SearchState::new_with_seed(&prob, 42);
    bnb.run(&mut state).unwrap();

    println!("vertices = {n}, edges = {}", prob.graph.num_edges());
    println!("cut = {}", state.best_solution.objective);
    println!(
        "proven optimal = {}",
        bnb.is_proven_optimal(&state.best_solution)
    );
    println!("iterations = {}", state.iteration);
    println!("elapsed = {:.3}s", state.duration().as_secs_f64());
}
