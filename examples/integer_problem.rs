//! Example of a problem built with `IntegerProblem` alone.
//!
//! Bounded knapsack, take up to `limit_i` copies of item `i` to maximize the
//! total value without exceeding the capacity. There is no move type and no
//! trait impl here, the problem is the variables and a closure, and
//! `IntChangeNeighbor` comes with it.
//!
//! Run with:
//! ```
//! cargo run --example integer_problem
//! ```

use optopus::prelude::*;

/// Every unit of weight over the capacity costs more than any item is worth,
/// so the optimum is feasible.
const PENALTY: f64 = 100.0;

fn main() {
    // (value, weight, how many copies there are)
    let items = [
        (60.0, 10.0, 3),
        (100.0, 20.0, 2),
        (120.0, 30.0, 2),
        (30.0, 5.0, 5),
        (45.0, 9.0, 4),
    ];
    let capacity = 100.0;

    let vars: IntVars = items
        .iter()
        .map(|&(_, _, limit)| IntVar::new(0, limit))
        .collect();
    let weight = move |counts: &[i64]| -> f64 {
        counts
            .iter()
            .zip(&items)
            .map(|(&c, &(_, w, _))| c as f64 * w)
            .sum()
    };
    // No delta is given, so every candidate is priced by the whole objective.
    // `with_delta` takes one when every evaluation counts.
    let prob = IntegerProblem::maximize(vars, move |counts: &[i64]| {
        let value: f64 = counts
            .iter()
            .zip(&items)
            .map(|(&c, &(v, _, _))| c as f64 * v)
            .sum();
        value - PENALTY * (weight(counts) - capacity).max(0.0)
    });

    let mut state = SearchState::new_with_seed(&prob, 42);
    LocalSearch::<IntChangeNeighbor>::new(StopCondition::iterations(1_000))
        .run(&mut state)
        .unwrap();
    report(
        "LocalSearch",
        weight(state.best_solution.values()),
        &state.best_solution,
    );

    let mut state = SearchState::new_with_seed(&prob, 42);
    SimulatedAnnealing::<IntChangeNeighbor>::new(StopCondition::iterations(50_000), 50.0, 0.9999)
        .run(&mut state)
        .unwrap();
    report(
        "SimulatedAnnealing",
        weight(state.best_solution.values()),
        &state.best_solution,
    );
}

fn report(name: &str, weight: f64, sol: &IntSolution) {
    println!(
        "{name:>18}: counts = {:?}, weight = {weight}, objective = {:?}",
        sol.values(),
        sol.evaluate()
    );
}
