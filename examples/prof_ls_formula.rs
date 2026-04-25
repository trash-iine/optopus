//! Profiling: LocalSearch × FormulaProblem × Flip + Swap (Tier 2)
//!
//! Hot paths (Flip):
//!   - apply_to_solution: refresh gain only for interaction_neighbors[i]
//!   - calc_gain_fast: polynomial evaluation via eval_poly_delta
//!
//! Hot paths (Swap):
//!   - iter(): eagerly builds sol.x.clone() + sol.constraint_vals.clone()
//!     + Vec<Neighbor> across all O(n²) pairs
//!   - For each i, builds delta_i: Vec<Value> linear in the constraint count
//!
//! LocalSearch halts at a local optimum, so wrap in Restart to run for a fixed time.
//!
//! How to run:
//! ```
//! cargo build --profile profiling --example prof_ls_formula
//! samply record ./target/profiling/examples/prof_ls_formula
//! ```

use std::time::Duration;

use optopus::prelude::*;

fn main() {
    // n=60 MaxCut-equivalent formula: sum_{i<j} x[i]*x[j].
    // Every pair has a cross term, so interaction_neighbors is dense and
    // gain recomputation is expensive.
    let n = 60usize;
    let objective = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| Expr::Var(i) * Expr::Var(j)))
        .reduce(|acc, e| acc + e)
        .unwrap();

    let prob = FormulaProblem::new(n, objective, OptDirection::Maximize, vec![]);

    // --- Flip ---
    let mut state = SearchState::new(&prob);
    Restart::new(
        StopCondition::duration(Duration::from_secs(3)),
        Box::new(LocalSearch::<FormulaFlipNeighbor>::new(
            StopCondition::failed_updates(1),
        )),
        StopCondition::failed_updates(1),
    )
    .run(&mut state)
    .unwrap();

    // --- Swap ---
    let mut state2 = SearchState::new(&prob);
    Restart::new(
        StopCondition::duration(Duration::from_secs(3)),
        Box::new(LocalSearch::<FormulaSwapNeighbor>::new(
            StopCondition::failed_updates(1),
        )),
        StopCondition::failed_updates(1),
    )
    .run(&mut state2)
    .unwrap();
}
