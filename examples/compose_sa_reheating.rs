//! Simulated annealing with reheating, and with restarts, composed from
//! `Sequential` and `Restart`.
//!
//! A single annealing run freezes once its temperature is low, and the rest of
//! its budget is spent in place. `Heuristic::run` resets the temperature, so
//! running the same annealer again from where it stopped reheats it.
//! `Sequential` with one step does exactly that, since it repeats its list
//! until the outer budget is spent and each step continues from the current
//! solution. `Restart` is the contrast, it starts each annealing run from a
//! fresh random schedule instead.
//!
//! Run with:
//! ```
//! cargo run --release --example compose_sa_reheating
//! ```

use optopus::prelude::*;

fn main() {
    let jssp = JobShopScheduling::load_file("data/instances/jssp/orlib/ft10.txt").unwrap();
    let budget = 400_000;
    let annealer = |stop| SimulatedAnnealing::<JobShopSwapNeighbor>::new(stop, 20.0, 0.9999);
    let one_run = || annealer(StopCondition::iterations(50_000));

    let mut state = SearchState::new_with_seed(&jssp, 42);
    annealer(StopCondition::iterations(budget))
        .run(&mut state)
        .unwrap();
    report("one SA", &state);

    let mut state = SearchState::new_with_seed(&jssp, 42);
    Sequential::<JobShopScheduling>::new(
        StopCondition::iterations(budget),
        vec![Box::new(one_run())],
    )
    .run(&mut state)
    .unwrap();
    report("reheated SA", &state);

    let mut state = SearchState::new_with_seed(&jssp, 42);
    Restart::<JobShopScheduling>::new(
        StopCondition::iterations(budget),
        Box::new(one_run()),
        StopCondition::failed_updates(0),
    )
    .run(&mut state)
    .unwrap();
    report("restarted SA", &state);
}

fn report(name: &str, state: &SearchState<JobShopScheduling>) {
    println!(
        "{name:>12}: makespan = {} (best at iteration {} of {})",
        state.best_solution.objective, state.best_iteration, state.iteration
    );
}
