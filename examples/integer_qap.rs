//! The quadratic assignment problem written as an `IntegerProblem`, next to
//! the built-in `Qap`, and the two tabu searches the built-in one is known
//! for.
//!
//! Variable `i` is the location of facility `i`, so the variables are a
//! permutation, and the objective is `Qap::cost`, an O(n²) closure. The
//! built-in problem prices an exchange of two facilities in O(n), and on it
//!
//! - `TabuSearch` with a tenure drawn at random per move is Taillard's robust
//!   tabu search, since the tabu rule of `QapSwapNeighbor` is his, and
//! - `Iterated` around a shorter tabu search, with a few random exchanges
//!   between runs, is the Iterated Tabu Search of Misevicius.
//!
//! The instance is uniform random like Taillard's `tai*a` family, drawn from
//! a fixed seed so no file is needed. All three get the same time.
//!
//! Run with:
//! ```
//! cargo run --release --example integer_qap
//! ```

use std::time::Duration;

use optopus::prelude::*;
use rand::Rng;

fn main() {
    let n = 30;
    let mut rng = seeded_rng(12);
    let mut matrix = || -> Vec<Vec<i64>> {
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i == j { 0 } else { rng.random_range(0..100) })
                    .collect()
            })
            .collect()
    };
    let (a, b) = (matrix(), matrix());
    let qap = Qap::new("random30", a, b);
    let budget = StopCondition::duration(Duration::from_secs(2));

    let qap_ref = &qap;
    let prob = IntegerProblem::minimize(IntVars::permutation(n), move |v: &[i64]| {
        let p: Vec<usize> = v.iter().map(|&l| l as usize).collect();
        qap_ref.cost(&p) as f64
    });
    let mut state = SearchState::new_with_seed(&prob, 42);
    SimulatedAnnealing::<IntSwapNeighbor>::new(budget.clone(), 2_000.0, 0.99999)
        .run(&mut state)
        .unwrap();
    report(
        "IntegerProblem + SA",
        state.best_solution.evaluate().minimized(),
        state.iteration,
    );

    let tenure = (n as u64 * 9 / 10, n as u64 * 11 / 10);
    let mut state = SearchState::new_with_seed(&qap, 42);
    TabuSearch::<QapSwapNeighbor>::new(budget.clone(), tenure)
        .run(&mut state)
        .unwrap();
    report(
        "robust tabu search",
        state.best_solution.objective as f64,
        state.iteration,
    );

    let mut state = SearchState::new_with_seed(&qap, 42);
    Iterated::<Qap>::new(
        budget,
        Box::new(TabuSearch::<QapSwapNeighbor>::new(
            StopCondition::failed_updates(20 * n as u64),
            tenure,
        )),
        Box::new(RandomWalk::<QapSwapNeighbor>::new(
            StopCondition::iterations(n as u64 / 4),
        )),
    )
    .run(&mut state)
    .unwrap();
    report(
        "iterated tabu search",
        state.best_solution.objective as f64,
        state.iteration,
    );
}

fn report(name: &str, cost: f64, iterations: u64) {
    println!("{name:>21}: cost = {cost} after {iterations} iterations");
}
