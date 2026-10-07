//! Weighted set cover written as a `FormulaProblem`.
//!
//! Every element of a universe has to be covered by at least one chosen set,
//! and the chosen sets should cost as little as possible. One binary variable
//! per set says whether it is chosen. The objective is the total cost, and
//! one constraint per element asks for at least one of the sets that hold
//! it. A broken constraint costs more than the dearest set, so covering an
//! element is always worth paying for, and the best solution covers
//! everything.
//!
//! `IntChangeNeighbor` flips one set in or out, and tabu search walks on it.
//! The problem works out what a flip changes from the formula, so a flip is
//! priced by the constraints of the elements that set holds, not the whole
//! universe.
//!
//! Run with:
//! ```
//! cargo run --release --example integer_set_cover
//! ```

use optopus::prelude::*;
use rand::Rng;

fn main() {
    let (elements, sets) = (200, 80);
    let mut rng = seeded_rng(4);
    // Each set covers about 5% of the universe at a cost between 1 and 20,
    // and every element is given to one more set so a cover exists.
    let mut covers: Vec<Vec<usize>> = (0..sets)
        .map(|_| (0..elements).filter(|_| rng.random_bool(0.05)).collect())
        .collect();
    for e in 0..elements {
        covers[rng.random_range(0..sets)].push(e);
    }
    let cost: Vec<f64> = (0..sets).map(|_| rng.random_range(1..=20) as f64).collect();
    let penalty = cost.iter().copied().fold(0.0, f64::max) + 1.0;

    // The cost to pay, and one constraint per element asking that at least one
    // of the sets holding it is chosen.
    let vars: IntVars = (0..sets).map(|_| IntVar::binary()).collect();
    let paid = (0..sets).fold(Expr::Const(0.0), |sum, s| sum + Expr::Var(s) * cost[s]);
    let prob = (0..elements).fold(FormulaProblem::minimize(vars, paid), |prob, e| {
        let holders = (0..sets)
            .filter(|&s| covers[s].contains(&e))
            .fold(Expr::Const(0.0), |sum, s| sum + Expr::Var(s));
        prob.with_constraint(Constraint::Comparison {
            lhs: holders,
            rel: ConstraintRel::Ge,
            rhs: Expr::Const(1.0),
            penalty_weight: penalty,
        })
    });

    let mut state = SearchState::new_with_seed(&prob, 42);
    TabuSearch::<IntChangeNeighbor>::new(StopCondition::iterations(5_000), (3, 10))
        .run(&mut state)
        .unwrap();

    let x = state.best_solution.values();
    let chosen: Vec<usize> = (0..sets).filter(|&s| x[s] == 1).collect();
    println!("chosen sets = {chosen:?}");
    println!(
        "cost = {}, uncovered elements = {}",
        prob.eval_objective(x),
        prob.eval_penalty(x) / penalty
    );
}
