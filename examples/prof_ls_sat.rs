//! Profiling: LocalSearch × SAT × Flip + Swap (Tier 2)
//!
//! Hot paths (Flip):
//!   - apply_to_solution: Vec<usize> (affected vars) alloc + sort_unstable + dedup
//!   - calc_gain() called per affected variable (clause scan)
//!
//! Hot paths (Swap):
//!   - iter() eagerly collects every Vec<SatSwapNeighbor>
//!   - HashSet<(usize, usize)> (seen-pair dedup) reallocated each call
//!   - calc_gain_with_virtual_flip called per clause pair
//!
//! LocalSearch halts at a local optimum, so wrap in Restart to run for a fixed time.
//!
//! How to run:
//! ```
//! cargo build --profile profiling --example prof_ls_sat
//! samply record ./target/profiling/examples/prof_ls_sat
//! ```

use std::time::Duration;

use optopus::prelude::*;

fn main() {
    // Build a 3-SAT instance: n=500 vars, ~2000 clauses (density ~4.0).
    // clause_ratio ~4.0 sits near the phase transition where gain updates fire frequently.
    let n_vars = 500usize;
    let n_clauses = 2000usize;
    let mut sat = Sat::new(n_vars);
    for k in 0..n_clauses {
        let a = ((k * 7 + 1) % n_vars) as i64 + 1;
        let b = ((k * 13 + 3) % n_vars) as i64 + 1;
        let c = ((k * 19 + 5) % n_vars) as i64 + 1;
        let lit_a = if k % 3 == 0 { -a } else { a };
        let lit_b = if k % 5 == 0 { -b } else { b };
        sat.add_clause([lit_a, lit_b, c]);
    }

    // --- Flip ---
    let mut state = SearchState::new(&sat);
    Restart::new(
        StopCondition::duration(Duration::from_secs(3)),
        Box::new(LocalSearch::<SatFlipNeighbor>::new(
            StopCondition::failed_updates(1),
        )),
        StopCondition::failed_updates(1),
    )
    .run(&mut state)
    .unwrap();

    // --- Swap ---
    let mut state2 = SearchState::new(&sat);
    Restart::new(
        StopCondition::duration(Duration::from_secs(3)),
        Box::new(LocalSearch::<SatSwapNeighbor>::new(
            StopCondition::failed_updates(1),
        )),
        StopCondition::failed_updates(1),
    )
    .run(&mut state2)
    .unwrap();
}
