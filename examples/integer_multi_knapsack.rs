//! A multidimensional knapsack solved by `BranchAndBound` over an
//! `IntegerProblem`.
//!
//! Each item has a value and a weight in each of several dimensions, and each
//! dimension has its own capacity. One binary variable per item says whether
//! it is packed. The objective is the packed value, less the overweight in
//! every dimension at twice the best value per unit of weight, so no
//! overweight packing can beat a feasible one.
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
        * (0..n)
            .flat_map(|i| weight.iter().map(move |w| (i, w[i])))
            .map(|(i, wi)| value[i] / wi)
            .fold(0.0, f64::max);
    let (value, weight, capacity) = (&value, &weight, &capacity);

    let vars: IntVars = (0..n).map(|_| IntVar::binary()).collect();
    let prob = IntegerProblem::maximize(vars, |x: &[i64]| {
        let packed = |w: &[f64]| x.iter().zip(w).map(|(&xi, wi)| xi as f64 * wi).sum::<f64>();
        let v = packed(value);
        let over: f64 = (0..dims)
            .map(|d| (packed(&weight[d]) - capacity[d]).max(0.0))
            .sum();
        v - penalty * over
    });

    // The Dantzig bound of the single knapsack in dimension `d`.
    let dantzig = |vars: &IntVars, d: usize| -> f64 {
        let (w_d, cap) = (&weight[d], capacity[d]);
        let (mut v, mut w) = (0.0, 0.0);
        let mut free = vec![];
        for (i, range) in vars.iter().enumerate() {
            if range.lower() == 1 {
                v += value[i];
                w += w_d[i];
            } else if range.upper() == 1 {
                free.push(i);
            }
        }
        if w > cap {
            return v - penalty * (w - cap);
        }
        free.sort_by(|&a, &b| (value[b] / w_d[b]).total_cmp(&(value[a] / w_d[a])));
        let mut room = cap - w;
        for i in free {
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
    let packed: Vec<usize> = (0..n).filter(|&i| x[i] == 1).collect();
    println!("packed items = {packed:?}");
    for d in 0..dims {
        let load: f64 = packed.iter().map(|&i| weight[d][i]).sum();
        println!("dimension {d}: weight = {load} of {}", capacity[d]);
    }
    println!("value = {}", -state.best_solution.evaluate().minimized());
    println!(
        "proven optimal = {}",
        bnb.is_proven_optimal(&state.best_solution)
    );
    println!("elapsed = {:.3}s", state.duration().as_secs_f64());
}
