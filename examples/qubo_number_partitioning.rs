//! Number partitioning written as a QUBO, searched by simulated annealing,
//! population annealing and parallel tempering on the same budget.
//!
//! Split the numbers `a_i` into two sets whose sums are as close as possible.
//! With `s_i = 2 x_i − 1` the side of number `i`, the squared residue is
//!
//! ```text
//! (Σ a_i s_i)² = Σ_i (4 a_i² − 4 A a_i) x_i + Σ_{i<j} 8 a_i a_j x_i x_j + A²
//! ```
//!
//! where `A = Σ a_i`, so the QUBO is the two sums and the constant is added
//! back afterwards.
//!
//! Number partitioning has a phase transition physicists know in closed form.
//! With `b`-bit numbers it is easy while `b / n` is below about one, where
//! perfect splits are plentiful, and hard above, where the landscape is close
//! to random. The QUBO's energies are 32-bit, and they range over `[-A², 0]`,
//! so `A²` has to fit, which keeps the numbers here at 10 bits and the
//! instance on the easy side. All
//! three find a split with the smallest residue the parity allows, and the
//! example is about how each is called, the two replica methods taking their
//! temperatures as inverse temperatures in the units of the energy.
//!
//! Run with:
//! ```
//! cargo run --release --example qubo_number_partitioning
//! ```

use std::time::Duration;

use optopus::prelude::*;
use rand::Rng;

fn main() {
    let mut rng = seeded_rng(5);
    let a: Vec<i64> = (0..40).map(|_| rng.random_range(1..1000)).collect();
    let total: i64 = a.iter().sum();
    let n = a.len();

    // QUBO coefficients and energies are i32. The energies run from -A² to 0,
    // so A² has to fit as well as every coefficient, and a value that does not
    // stops here rather than wrapping.
    assert!(
        total * total <= i64::from(i32::MAX),
        "A² = {} does not fit the QUBO's i32 energies",
        total * total
    );
    let coef = |v: i64| i32::try_from(v).expect("coefficient fits in i32");
    let mut entries = Vec::new();
    for i in 0..n {
        entries.push((i, i, coef(4 * a[i] * a[i] - 4 * total * a[i])));
        for j in i + 1..n {
            entries.push((i, j, coef(8 * a[i] * a[j])));
        }
    }
    let qubo = Qubo::from_entries(entries);
    let residue = |sol: &QuboSolution| -> i64 {
        let squared = i64::from(sol.objective) + total * total;
        (squared as f64).sqrt().round() as i64
    };
    // The energy changes run to millions, so the inverse temperatures are
    // tiny.
    let budget = StopCondition::duration(Duration::from_secs(1));
    println!("{n} numbers summing to {total}");

    let mut state = SearchState::new_with_seed(&qubo, 42);
    SimulatedAnnealing::<QuboFlipNeighbor>::new(budget.clone(), 1.0e6, 0.999999)
        .run(&mut state)
        .unwrap();
    println!(
        "  SimulatedAnnealing  residue {}",
        residue(&state.best_solution)
    );

    let mut state = SearchState::new_with_seed(&qubo, 42);
    PopulationAnnealing::<Qubo, QuboFlipNeighbor>::new(
        budget.clone(),
        50,
        1.0e-7,
        2.0e-8,
        10,
        Some(400),
    )
    .run(&mut state)
    .unwrap();
    println!(
        "  PopulationAnnealing residue {}",
        residue(&state.best_solution)
    );

    let mut state = SearchState::new_with_seed(&qubo, 42);
    let mut pt = ParallelTempering::<Qubo, QuboFlipNeighbor>::new(budget, 20, 1.0e-7, 1.0e-3, 10);
    pt.run(&mut state).unwrap();
    println!(
        "  ParallelTempering   residue {} (exchanges accepted {:.0}%)",
        residue(&state.best_solution),
        100.0 * pt.exchange_acceptance().unwrap_or(0.0)
    );
}
