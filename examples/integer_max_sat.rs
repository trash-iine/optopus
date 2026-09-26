//! MaxSAT written as an `IntegerProblem`.
//!
//! One binary variable per boolean variable, and the objective counts the
//! clauses that hold. A literal `l` asks variable `|l| - 1` to be `1` when `l`
//! is positive and `0` when it is negative.
//!
//! Run with:
//! ```
//! cargo run --example integer_max_sat
//! ```

use optopus::prelude::*;

fn main() {
    let sat = Sat::load_file("data/instances/sat/sample.cnf").unwrap();

    let vars: IntVars = (0..sat.n_vars()).map(|_| IntVar::binary()).collect();
    let prob = IntegerProblem::maximize(vars, |x: &[i64]| {
        let holds = |l: i64| (x[l.unsigned_abs() as usize - 1] == 1) == (l > 0);
        sat.all_clauses()
            .filter(|clause| clause.iter().any(|&l| holds(l)))
            .count() as f64
    });

    let mut state = SearchState::new_with_seed(&prob, 42);
    TabuSearch::<IntChangeNeighbor>::new(StopCondition::iterations(200), (2, 5))
        .run(&mut state)
        .unwrap();
    println!("x = {:?}", state.best_solution.values());
    println!(
        "satisfied = {:?} of {}",
        state.best_solution.evaluate(),
        sat.n_clauses()
    );
}
