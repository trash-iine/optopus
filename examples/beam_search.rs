//! Beam Search example.
//!
//! Applies BeamSearch (two beam widths) and LocalSearch to the same MaxCut
//! instance and compares the results.
//!
//! Run with:
//! ```
//! cargo run --example beam_search
//! ```

use optopus::prelude::*;

fn main() {
    let mc = MaxCut::new(Graph::from_edges([
        (0, 1, 1.0),
        (0, 2, 1.0),
        (0, 3, 1.0),
        (1, 2, 1.0),
        (1, 4, 1.0),
        (2, 5, 1.0),
        (3, 4, 1.0),
        (3, 5, 1.0),
        (4, 5, 1.0),
    ]));

    let sc = StopCondition::iterations(100);

    // --- Beam Search (beam_width = 3) ---
    let mut state = SearchState::new(&mc);
    let mut bs = BeamSearch::<MaxCut, MaxCutFlipNeighbor>::new(sc.clone(), 3);
    bs.run(&mut state).unwrap();
    println!(
        "[BeamSearch w=3]  best = {:.1} (iter {})",
        state.best_solution.objective, state.best_iteration
    );

    // --- Beam Search (beam_width = 1, equivalent to LocalSearch) ---
    let mut state = SearchState::new(&mc);
    let mut bs1 = BeamSearch::<MaxCut, MaxCutFlipNeighbor>::new(sc.clone(), 1);
    bs1.run(&mut state).unwrap();
    println!(
        "[BeamSearch w=1]  best = {:.1} (iter {})",
        state.best_solution.objective, state.best_iteration
    );

    // --- LocalSearch (for comparison) ---
    let mut state = SearchState::new(&mc);
    let mut ls = LocalSearch::<MaxCutFlipNeighbor>::new(sc.clone());
    ls.run(&mut state).unwrap();
    println!(
        "[LocalSearch]     best = {:.1} (iter {})",
        state.best_solution.objective, state.best_iteration
    );
}
