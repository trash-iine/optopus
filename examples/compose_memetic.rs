//! A memetic algorithm composed from `GeneticAlgorithm` and `TabuSearch`.
//!
//! `GeneticAlgorithm` takes its mutation step as any heuristic. Handing it a
//! short tabu search turns every offspring into a local optimum before it
//! enters the population, which is the hybrid evolutionary algorithm of
//! Galinier and Hao for graph coloring, a crossover for exploration and tabu
//! search for intensification.
//!
//! The graph is drawn from a fixed seed, so no instance file is needed.
//!
//! Run with:
//! ```
//! cargo run --release --example compose_memetic
//! ```

use optopus::prelude::*;
use optopus::problem::GraphColoringUniformCrossover;

fn main() {
    let gc = GraphColoring::new(Graph::erdos_renyi(120, 0.1, &mut seeded_rng(3)));
    let budget = 60_000;

    let mut state = SearchState::new_with_seed(&gc, 42);
    TabuSearch::<GraphColoringRecolorNeighbor>::new(StopCondition::iterations(budget), (5, 15))
        .run(&mut state)
        .unwrap();
    report("TabuSearch", &state);

    let mut state = SearchState::new_with_seed(&gc, 42);
    GeneticAlgorithm::new(
        StopCondition::iterations(budget),
        10,
        GraphColoringUniformCrossover,
        Box::new(TabuSearch::<GraphColoringRecolorNeighbor>::new(
            StopCondition::failed_updates(300),
            (5, 15),
        )),
        ParentSelection::Tournament,
    )
    .run(&mut state)
    .unwrap();
    report("memetic", &state);
}

fn report(name: &str, state: &SearchState<GraphColoring>) {
    let best = &state.best_solution;
    println!(
        "{name:>12}: colors = {}, conflicts = {} (best at iteration {} of {})",
        best.colors_used, best.conflicts, state.best_iteration, state.iteration
    );
}
