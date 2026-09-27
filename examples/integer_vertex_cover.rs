//! Vertex cover written as a `FormulaProblem`.
//!
//! One binary variable per vertex says whether it is in the cover. The
//! objective counts the chosen vertices, and one constraint per edge asks for
//! at least one of its ends. A broken constraint costs more than the vertex
//! that would mend it, so the best solution is a cover.
//!
//! Run with:
//! ```
//! cargo run --example integer_vertex_cover
//! ```

use optopus::prelude::*;

fn main() {
    let graph = Graph::erdos_renyi(100, 0.05, &mut seeded_rng(1));

    let vars: IntVars = (0..graph.len()).map(|_| IntVar::binary()).collect();
    let size = (0..graph.len()).fold(Expr::Const(0.0), |sum, i| sum + Expr::Var(i));
    let prob = graph
        .edges()
        .fold(FormulaProblem::minimize(vars, size), |prob, (i, j, _)| {
            prob.with_constraint(Constraint::Comparison {
                lhs: Expr::Var(i) + Expr::Var(j),
                rel: ConstraintRel::Ge,
                rhs: Expr::Const(1.0),
                penalty_weight: 2.0,
            })
        });

    let mut state = SearchState::new_with_seed(&prob, 42);
    SimulatedAnnealing::<IntChangeNeighbor>::new(StopCondition::iterations(100_000), 1.0, 0.9999)
        .run(&mut state)
        .unwrap();
    let cover = state.best_solution.values();
    println!("cover = {cover:?}");
    println!("size = {}", prob.eval_objective(cover));
    println!("uncovered edges = {}", prob.eval_penalty(cover) / 2.0);
}
