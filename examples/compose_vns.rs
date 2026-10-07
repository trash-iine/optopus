//! Variable Neighborhood Search composed from `VariableNeighborhoodSearch`,
//! `LocalSearch` and three `RandomWalk` shakes.
//!
//! The shakes are the same random recoloring at three strengths, 2, 5 and 12
//! random moves. VNS shakes with the gentlest, descends, and keeps the result
//! only if it improved. On a failure it restores the incumbent and moves to
//! the next stronger shake, and on a success it drops back to the gentlest.
//!
//! The graph is drawn from a fixed seed, so no instance file is needed.
//!
//! Run with:
//! ```
//! cargo run --release --example compose_vns
//! ```

use optopus::prelude::*;

fn main() {
    let gc = GraphColoring::new(Graph::erdos_renyi(120, 0.1, &mut seeded_rng(3)));
    let budget = 30_000;
    let shake = |moves| -> Box<dyn Heuristic<GraphColoring>> {
        Box::new(RandomWalk::<GraphColoringRecolorNeighbor>::new(
            StopCondition::iterations(moves),
        ))
    };

    let mut state = SearchState::new_with_seed(&gc, 42);
    LocalSearch::<GraphColoringRecolorNeighbor>::new(StopCondition::iterations(budget))
        .run(&mut state)
        .unwrap();
    report("LocalSearch", &state);

    let mut state = SearchState::new_with_seed(&gc, 42);
    VariableNeighborhoodSearch::<GraphColoring>::new(
        StopCondition::iterations(budget),
        Box::new(LocalSearch::<GraphColoringRecolorNeighbor>::new(
            StopCondition::iterations(budget),
        )),
        vec![shake(2), shake(5), shake(12)],
    )
    .run(&mut state)
    .unwrap();
    report("VNS", &state);
}

fn report(name: &str, state: &SearchState<GraphColoring>) {
    let best = &state.best_solution;
    println!(
        "{name:>12}: colors = {}, conflicts = {} (best at iteration {} of {})",
        best.colors_used, best.conflicts, state.best_iteration, state.iteration
    );
}
