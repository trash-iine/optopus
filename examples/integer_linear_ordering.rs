//! The linear ordering problem written as an `IntegerProblem` over a
//! permutation.
//!
//! Given a weight `w[a][b]` for every ordered pair, find the order of the
//! items that maximizes the total weight of the pairs it puts in their own
//! direction, `w[a][b]` whenever `a` comes before `b`. It is the problem
//! behind ranking by pairwise preferences and triangulating input-output
//! tables, and a standard test bed for permutation searches.
//!
//! Variable `p` is the item placed `p`th. `IntSwapNeighbor` exchanges two
//! positions, and `Iterated` runs a descent over it with a few random swaps
//! between descents.
//!
//! Run with:
//! ```
//! cargo run --release --example integer_linear_ordering
//! ```

use optopus::prelude::*;
use rand::Rng;

fn main() {
    let n = 30;
    let mut rng = seeded_rng(9);
    let w: Vec<Vec<f64>> = (0..n)
        .map(|a| {
            (0..n)
                .map(|b| {
                    if a == b {
                        0.0
                    } else {
                        rng.random_range(0..100) as f64
                    }
                })
                .collect()
        })
        .collect();
    let w = &w;

    let forward = move |order: &[i64]| -> f64 {
        let mut total = 0.0;
        for (i, &a) in order.iter().enumerate() {
            for &b in &order[i + 1..] {
                total += w[a as usize][b as usize];
            }
        }
        total
    };
    let prob = IntegerProblem::maximize(IntVars::permutation(n), forward);

    // Every pair counts in one direction or the other, so the sum of the
    // larger of the two weights bounds any order from above.
    let upper: f64 = (0..n)
        .flat_map(|a| (a + 1..n).map(move |b| w[a][b].max(w[b][a])))
        .sum();

    let mut state = SearchState::new_with_seed(&prob, 42);
    Iterated::new(
        StopCondition::iterations(10_000),
        Box::new(LocalSearch::<IntSwapNeighbor>::new(
            StopCondition::iterations(20_000),
        )),
        Box::new(RandomWalk::<IntSwapNeighbor>::new(
            StopCondition::iterations(3),
        )),
    )
    .run(&mut state)
    .unwrap();

    let value = forward(state.best_solution.values());
    println!("order = {:?}", state.best_solution.values());
    println!("forward weight = {value} (at most {upper})");
}
