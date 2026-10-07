//! A multidimensional knapsack solved by `BranchAndBound` over a
//! `FormulaProblem`.
//!
//! Each item has a value and a weight in each of several dimensions, and each
//! dimension has its own capacity. One binary variable per item says whether
//! it is packed. The objective is the packed value, and one constraint per
//! dimension keeps the load within capacity, charging the overweight at twice
//! the best value per unit of weight, so no overweight packing can beat a
//! feasible one.
//!
//! The relaxation is the point of this example. Dropping every dimension but
//! one leaves an ordinary knapsack, whose fractional fill bounds the packed
//! value from above, and so does the bound of every other dimension. The
//! smallest of them is the bound handed to `BranchAndBound`. This is the same
//! Dantzig bound as in `integer_branch_and_bound.rs`, taken once per
//! dimension.
//!
//! Run with:
//! ```
//! cargo run --release --example integer_multi_knapsack
//! ```

use optopus::prelude::*;
use rand::Rng;

fn main() {
    let (n, dims) = (50, 3);
    let mut rng = seeded_rng(2);
    let value: Vec<f64> = (0..n).map(|_| rng.random_range(10..100) as f64).collect();
    let weight: Vec<Vec<f64>> = (0..dims)
        .map(|_| (0..n).map(|_| rng.random_range(10..100) as f64).collect())
        .collect();
    let capacity: Vec<f64> = weight.iter().map(|w| w.iter().sum::<f64>() / 2.0).collect();
    let penalty = 2.0
        * weight
            .iter()
            .flat_map(|w| value.iter().zip(w).map(|(v, wi)| v / wi))
            .fold(0.0, f64::max);

    let vars: IntVars = (0..n).map(|_| IntVar::binary()).collect();
    let packed =
        |coef: &[f64]| (0..n).fold(Expr::Const(0.0), |sum, i| sum + Expr::Var(i) * coef[i]);
    let prob = (0..dims).fold(FormulaProblem::maximize(vars, packed(&value)), |prob, d| {
        prob.with_constraint(Constraint::Comparison {
            lhs: packed(&weight[d]),
            rel: ConstraintRel::Le,
            rhs: Expr::Const(capacity[d]),
            penalty_weight: penalty,
        })
    });

    // Each dimension's items by value per unit of its weight, best first.
    // The order is the instance's, so it is sorted once, not per node.
    let order: Vec<Vec<usize>> = weight
        .iter()
        .map(|w| {
            let mut items: Vec<usize> = (0..n).collect();
            items.sort_by(|&a, &b| (value[b] / w[b]).total_cmp(&(value[a] / w[a])));
            items
        })
        .collect();

    // The Dantzig bound of the single knapsack in dimension `d`. The items
    // forced in, then the free ones in order until the capacity is reached,
    // the last of them in part.
    let dantzig = |vars: &IntVars, d: usize| -> f64 {
        let (w_d, cap) = (&weight[d], capacity[d]);
        let forced = (0..n).filter(|&i| vars[i].lower() == 1);
        let (mut v, w) = forced.fold((0.0, 0.0), |(v, w), i| (v + value[i], w + w_d[i]));
        if w > cap {
            return v - penalty * (w - cap);
        }
        let mut room = cap - w;
        for &i in order[d]
            .iter()
            .filter(|&&i| vars[i].lower() == 0 && vars[i].upper() == 1)
        {
            if w_d[i] <= room {
                v += value[i];
                room -= w_d[i];
            } else {
                v += value[i] * room / w_d[i];
                break;
            }
        }
        v
    };
    let bound = |_: &_, vars: &IntVars| {
        Evaluable::Maximize(
            (0..dims)
                .map(|d| dantzig(vars, d))
                .fold(f64::INFINITY, f64::min),
        )
    };

    let mut bnb = BranchAndBound::new(
        StopCondition::new(None, None, None),
        Box::new(LocalSearch::<IntChangeNeighbor>::new(
            StopCondition::iterations(100),
        )),
        bound,
    );
    let mut state = SearchState::new_with_seed(&prob, 42);
    bnb.run(&mut state).unwrap();

    let x = state.best_solution.values();
    let chosen: Vec<usize> = (0..n).filter(|&i| x[i] == 1).collect();
    println!("packed items = {chosen:?}");
    for d in 0..dims {
        let load: f64 = chosen.iter().map(|&i| weight[d][i]).sum();
        println!("dimension {d}: weight = {load} of {}", capacity[d]);
    }
    println!("value = {}", prob.eval_objective(x));
    println!(
        "proven optimal = {}",
        bnb.is_proven_optimal(&state.best_solution)
    );
    println!("elapsed = {:.3}s", state.duration().as_secs_f64());
}
