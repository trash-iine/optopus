//! A 0-1 knapsack solved to proven optimality by `BranchAndBound`.
//!
//! One binary variable per item says whether it is packed. The objective is
//! the packed value, less the overweight at twice the best value per unit of
//! weight, which no overweight packing can make up for. The inner heuristic
//! finds good packings inside each node, and the relaxation, the fractional
//! fill of Dantzig, rules out the nodes that cannot beat them.
//!
//! Run with:
//! ```
//! cargo run --release --example integer_branch_and_bound
//! ```

use optopus::prelude::*;
use rand::Rng;

fn main() {
    let mut rng = seeded_rng(1);
    let n = 40;
    let value: Vec<f64> = (0..n).map(|_| rng.random_range(10..100) as f64).collect();
    let weight: Vec<f64> = (0..n).map(|_| rng.random_range(10..100) as f64).collect();
    let capacity = weight.iter().sum::<f64>() / 2.0;
    let penalty = 2.0
        * value
            .iter()
            .zip(&weight)
            .map(|(v, w)| v / w)
            .fold(0.0, f64::max);
    let (value, weight) = (&value, &weight);

    let vars: IntVars = (0..n).map(|_| IntVar::binary()).collect();
    let prob = IntegerProblem::maximize(vars, |x: &[i64]| {
        let (v, w) = x.iter().enumerate().fold((0.0, 0.0), |(v, w), (i, &xi)| {
            (v + xi as f64 * value[i], w + xi as f64 * weight[i])
        });
        v - penalty * (w - capacity).max(0.0)
    });

    // The items forced in, then the free ones by value per weight until the
    // capacity is reached, the last of them in part.
    let dantzig = |_: &_, vars: &IntVars| {
        let (mut v, mut w) = (0.0, 0.0);
        let mut free = vec![];
        for (i, range) in vars.iter().enumerate() {
            if range.lower() == 1 {
                v += value[i];
                w += weight[i];
            } else if range.upper() == 1 {
                free.push(i);
            }
        }
        if w > capacity {
            return Evaluable::Maximize(v - penalty * (w - capacity));
        }
        free.sort_by(|&a, &b| (value[b] / weight[b]).total_cmp(&(value[a] / weight[a])));
        let mut room = capacity - w;
        for i in free {
            if weight[i] <= room {
                v += value[i];
                room -= weight[i];
            } else {
                v += value[i] * room / weight[i];
                break;
            }
        }
        Evaluable::Maximize(v)
    };

    let mut bnb = BranchAndBound::new(
        StopCondition::new(None, None, None),
        Box::new(LocalSearch::<IntChangeNeighbor>::new(
            StopCondition::iterations(100),
        )),
        dantzig,
    );
    let mut state = SearchState::new_with_seed(&prob, 42);
    bnb.run(&mut state).unwrap();

    let packed: Vec<usize> = (0..n)
        .filter(|&i| state.best_solution.value(i) == 1)
        .collect();
    let packed_weight: f64 = packed.iter().map(|&i| weight[i]).sum();
    println!("packed items = {packed:?}");
    println!("weight = {packed_weight} of {capacity}");
    println!("value = {}", -state.best_solution.evaluate().minimized());
    println!(
        "proven optimal = {}",
        bnb.is_proven_optimal(&state.best_solution)
    );
    println!("iterations = {}", state.iteration);
    println!("elapsed = {:.3}s", state.duration().as_secs_f64());
}
