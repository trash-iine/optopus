//! Multi-start local search composed from `Restart` and `LocalSearch`.
//!
//! `Restart` runs its inner heuristic and then, once the restart condition
//! holds, replaces the current solution with a fresh random one while the
//! global best survives. With `failed_updates(0)` the condition holds after
//! every descent, so each cycle is one 2-opt descent from a new random tour,
//! the classic multi-start scheme.
//!
//! The cities are drawn from a fixed seed, so no instance file is needed.
//!
//! Run with:
//! ```
//! cargo run --release --example compose_multi_start
//! ```

use optopus::prelude::*;
use rand::Rng;

fn main() {
    let mut rng = seeded_rng(7);
    let cities: Vec<(f64, f64)> = (0..150)
        .map(|_| (rng.random_range(0.0..1000.0), rng.random_range(0.0..1000.0)))
        .collect();
    let tsp = Tsp::new("random150".to_string(), cities);
    let budget = 20_000;

    let mut state = SearchState::new_with_seed(&tsp, 42);
    LocalSearch::<TspTwoOptNeighbor>::new(StopCondition::iterations(budget))
        .run(&mut state)
        .unwrap();
    report("one descent", &state);

    let mut state = SearchState::new_with_seed(&tsp, 42);
    Restart::<Tsp>::new(
        StopCondition::iterations(budget),
        Box::new(LocalSearch::<TspTwoOptNeighbor>::new(
            StopCondition::iterations(budget),
        )),
        StopCondition::failed_updates(0),
    )
    .run(&mut state)
    .unwrap();
    report("multi-start", &state);
}

fn report(name: &str, state: &SearchState<Tsp>) {
    println!(
        "{name:>12}: length = {:.1} (best at iteration {} of {})",
        state.best_solution.objective, state.best_iteration, state.iteration
    );
}
