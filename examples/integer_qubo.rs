//! QUBO written as a `FormulaProblem`.
//!
//! One binary variable per index, and the energy `Σ Q[i][j] x_i x_j` written
//! as an `Expr`. A `FormulaProblem` works out what every flip changes from the
//! expression, so nothing but the formula is written here.
//!
//! Run with:
//! ```
//! cargo run --example integer_qubo
//! ```

use optopus::prelude::*;

fn main() {
    let qubo = Qubo::load_file("data/instances/qubo/bqp/bqp100_1.txt").unwrap();

    let vars: IntVars = (0..qubo.len()).map(|_| IntVar::binary()).collect();
    let energy = qubo.entries().fold(Expr::Const(0.0), |sum, (i, j, q)| {
        sum + q as f64 * Expr::Var(i) * Expr::Var(j)
    });
    let prob = FormulaProblem::minimize(vars, energy);

    let mut state = SearchState::new_with_seed(&prob, 42);
    SimulatedAnnealing::<IntChangeNeighbor>::new(StopCondition::iterations(100_000), 100.0, 0.9999)
        .run(&mut state)
        .unwrap();
    println!("x = {:?}", state.best_solution.values());
    println!("energy = {:?}", state.best_solution.evaluate());
}
