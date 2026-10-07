//! Weighted set cover written as an `IntegerProblem`.
//!
//! Every element of a universe has to be covered by at least one chosen set,
//! and the chosen sets should cost as little as possible. One binary variable
//! per set says whether it is chosen. The objective is the total cost plus a
//! penalty for every element left uncovered. The penalty is larger than the
//! dearest set, so covering an element is always worth paying for, and the
//! best solution covers everything.
//!
//! `IntChangeNeighbor` flips one set in or out, and tabu search walks on it.
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
        let s = rng.random_range(0..sets);
        if !covers[s].contains(&e) {
            covers[s].push(e);
        }
    }
    let cost: Vec<f64> = (0..sets).map(|_| rng.random_range(1..=20) as f64).collect();
    let penalty = cost.iter().cloned().fold(0.0, f64::max) + 1.0;
    let (covers, cost) = (&covers, &cost);

    let uncovered = move |x: &[i64]| -> usize {
        let mut covered = vec![false; elements];
        for (s, _) in x.iter().enumerate().filter(|&(_, &xs)| xs == 1) {
            for &e in &covers[s] {
                covered[e] = true;
            }
        }
        covered.iter().filter(|&&c| !c).count()
    };
    let vars: IntVars = (0..sets).map(|_| IntVar::binary()).collect();
    let prob = IntegerProblem::minimize(vars, move |x: &[i64]| {
        let paid: f64 = x.iter().zip(cost).map(|(&xs, c)| xs as f64 * c).sum();
        paid + penalty * uncovered(x) as f64
    });

    let mut state = SearchState::new_with_seed(&prob, 42);
    TabuSearch::<IntChangeNeighbor>::new(StopCondition::iterations(5_000), (3, 10))
        .run(&mut state)
        .unwrap();

    let x = state.best_solution.values();
    let chosen: Vec<usize> = (0..sets).filter(|&s| x[s] == 1).collect();
    let paid: f64 = chosen.iter().map(|&s| cost[s]).sum();
    println!("chosen sets = {chosen:?}");
    println!("cost = {paid}, uncovered elements = {}", uncovered(x));
}
