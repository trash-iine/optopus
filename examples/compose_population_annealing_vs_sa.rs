//! Population annealing and simulated annealing on the same instance and the
//! same time budget.
//!
//! Both anneal with the same Metropolis move. Simulated annealing cools one
//! solution. Population annealing cools a population of replicas and, at
//! every temperature step, resamples it toward the low-energy replicas, which
//! is how it gets past barriers that a single cooling run freezes behind.
//! `PopulationAnnealing` is parameterized by the problem rather than by one
//! neighbor in a config, but in Rust it is called like any other heuristic.
//!
//! The time budget makes the two comparable, since one population annealing
//! step sweeps every replica and is far more work than one annealing move.
//!
//! Run with:
//! ```
//! cargo run --release --example compose_population_annealing_vs_sa
//! ```

use std::time::{Duration, Instant};

use optopus::heuristic::PopulationAnnealing;
use optopus::prelude::*;

fn main() {
    let mc = MaxCut::new(Graph::erdos_renyi(2_000, 0.003, &mut seeded_rng(1)));
    let budget = StopCondition::duration(Duration::from_secs(2));

    let start = Instant::now();
    let mut state = SearchState::new_with_seed(&mc, 42);
    SimulatedAnnealing::<MaxCutFlipNeighbor>::new(budget.clone(), 2.0, 0.999999)
        .run(&mut state)
        .unwrap();
    report("SimulatedAnnealing", &state, start);

    let start = Instant::now();
    let mut state = SearchState::new_with_seed(&mc, 42);
    PopulationAnnealing::<MaxCut, MaxCutFlipNeighbor>::new(budget, 50, 0.1, 0.02, 10, Some(400))
        .run(&mut state)
        .unwrap();
    report("PopulationAnnealing", &state, start);
}

fn report(name: &str, state: &SearchState<MaxCut>, start: Instant) {
    println!(
        "{name:>19}: cut = {:.0} (found after {:.2}s)",
        state.best_solution.objective,
        state.best_time.duration_since(start).as_secs_f64()
    );
}
