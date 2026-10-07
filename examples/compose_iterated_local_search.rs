//! Iterated Local Search composed from `Iterated`, `LocalSearch` and
//! `RandomWalk`.
//!
//! `LocalSearch` alone stops at the first local optimum it reaches. `Iterated`
//! turns it into ILS by alternating it with a perturbation, here a few random
//! flips, so the search leaves the basin it stalled in and descends again. The
//! global best survives every cycle, so nothing found is ever lost. No new
//! heuristic is written, the three parts already exist.
//!
//! The instance is a sparse random graph of G-set size, generated from a fixed
//! seed so no instance file is needed.
//!
//! Run with:
//! ```
//! cargo run --release --example compose_iterated_local_search
//! ```

use optopus::prelude::*;

fn main() {
    let mc = MaxCut::new(Graph::erdos_renyi(800, 0.01, &mut seeded_rng(1)));
    let budget = 200_000;

    // A single descent from a random cut.
    let mut state = SearchState::new_with_seed(&mc, 42);
    LocalSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(budget))
        .run(&mut state)
        .unwrap();
    report("LocalSearch", &state);

    // ILS: descend, kick with 10 random flips, descend again, and so on.
    let mut state = SearchState::new_with_seed(&mc, 42);
    Iterated::<MaxCut>::new(
        StopCondition::iterations(budget),
        Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
            StopCondition::iterations(budget),
        )),
        Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(
            StopCondition::iterations(10),
        )),
    )
    .run(&mut state)
    .unwrap();
    report("Iterated LS", &state);
}

fn report(name: &str, state: &SearchState<MaxCut>) {
    println!(
        "{name:>12}: cut = {:.0} (best at iteration {} of {})",
        state.best_solution.objective, state.best_iteration, state.iteration
    );
}
