//! Example of a problem written with `IntegerProblem` alone.
//!
//! Bounded knapsack, take up to `limit_i` copies of item `i` to maximize the
//! total value without exceeding the capacity. There is no move type here,
//! `IntChangeNeighbor` comes with `IntegerProblem`.
//!
//! Run with:
//! ```
//! cargo run --example integer_problem
//! ```

use optopus::prelude::*;

struct BoundedKnapsack {
    vars: IntVars,
    values: Vec<f64>,
    weights: Vec<f64>,
    capacity: f64,
}

impl BoundedKnapsack {
    /// Every unit of weight over the capacity costs more than any item is
    /// worth, so the optimum is feasible.
    const PENALTY: f64 = 100.0;

    fn new(items: &[(f64, f64, i64)], capacity: f64) -> Self {
        Self {
            vars: items
                .iter()
                .map(|&(_, _, limit)| IntVar::new(0, limit))
                .collect(),
            values: items.iter().map(|&(v, _, _)| v).collect(),
            weights: items.iter().map(|&(_, w, _)| w).collect(),
            capacity,
        }
    }

    fn weight(&self, counts: &[i64]) -> f64 {
        counts
            .iter()
            .zip(&self.weights)
            .map(|(&c, w)| c as f64 * w)
            .sum()
    }
}

impl IntegerProblem for BoundedKnapsack {
    fn variables(&self) -> &IntVars {
        &self.vars
    }

    fn objective(&self, counts: &[i64]) -> Evaluable<f64> {
        let value: f64 = counts
            .iter()
            .zip(&self.values)
            .map(|(&c, v)| c as f64 * v)
            .sum();
        let overweight = (self.weight(counts) - self.capacity).max(0.0);
        Evaluable::Maximize(value - Self::PENALTY * overweight)
    }

    // `delta` is left to its default, which evaluates the whole objective.
    // Override it when every evaluation counts.
}

fn main() {
    // (value, weight, how many copies there are)
    let items = [
        (60.0, 10.0, 3),
        (100.0, 20.0, 2),
        (120.0, 30.0, 2),
        (30.0, 5.0, 5),
        (45.0, 9.0, 4),
    ];
    let prob = BoundedKnapsack::new(&items, 100.0);

    let mut state = SearchState::new_with_seed(&prob, 42);
    LocalSearch::<IntChangeNeighbor>::new(StopCondition::iterations(1_000))
        .run(&mut state)
        .unwrap();
    report("LocalSearch", &prob, &state.best_solution);

    let mut state = SearchState::new_with_seed(&prob, 42);
    SimulatedAnnealing::<IntChangeNeighbor>::new(StopCondition::iterations(50_000), 50.0, 0.9999)
        .run(&mut state)
        .unwrap();
    report("SimulatedAnnealing", &prob, &state.best_solution);
}

fn report(name: &str, prob: &BoundedKnapsack, sol: &IntSolution) {
    println!(
        "{name:>18}: counts = {:?}, weight = {}, objective = {:?}",
        sol.values(),
        prob.weight(sol.values()),
        sol.evaluate()
    );
}
