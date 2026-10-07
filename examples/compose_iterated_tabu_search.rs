//! Iterated Tabu Search composed from `Iterated`, `TabuSearch` and
//! `RandomWalk`.
//!
//! Tabu search already escapes local optima on its own, but on a long run it
//! circles the same region. Iterated Tabu Search (Misevicius) restarts it
//! from a perturbed copy of the incumbent each time it stops improving, which
//! is `Iterated` with tabu search as the search phase. The inner tabu search
//! stops after a stretch without improvement, so each cycle is one tabu walk
//! plus one kick.
//!
//! Run with:
//! ```
//! cargo run --release --example compose_iterated_tabu_search
//! ```

use optopus::prelude::*;

fn main() {
    let mc = MaxCut::new(Graph::erdos_renyi(2_000, 0.003, &mut seeded_rng(1)));
    let budget = 100_000;

    let mut state = SearchState::new_with_seed(&mc, 42);
    TabuSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(budget), (10, 30))
        .run(&mut state)
        .unwrap();
    report("TabuSearch", &state);

    let mut state = SearchState::new_with_seed(&mc, 42);
    Iterated::<MaxCut>::new(
        StopCondition::iterations(budget),
        Box::new(TabuSearch::<MaxCutFlipNeighbor>::new(
            StopCondition::failed_updates(2_000),
            (10, 30),
        )),
        Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(
            StopCondition::iterations(25),
        )),
    )
    .run(&mut state)
    .unwrap();
    report("Iterated TS", &state);
}

fn report(name: &str, state: &SearchState<MaxCut>) {
    println!(
        "{name:>12}: cut = {:.0} (best at iteration {} of {})",
        state.best_solution.objective, state.best_iteration, state.iteration
    );
}
