//! Profiling: Hybrid Genetic Search × CVRP (Tier 1, CVRP baseline)
//!
//! Hot paths:
//!   - local_search(): granular descent, one O(1) delta per candidate move; the
//!     dominant cost by far (every offspring is driven to a local optimum)
//!   - split_giant_tour(): O(fleet · n · route length) DP per offspring
//!   - build_neighbor_lists(): one-time O(n²) candidate-list construction
//!   - Subpopulation::push(): O(N·n) broken-pairs row for the newcomer plus an
//!     O(N² log N) re-rank, once per offspring (the matrix itself is cached)
//!   - prob.distance(): O(1) read from the lazily built distance matrix
//!     (nodes ≤ VRP_DIST_MATRIX_MAX_N)
//!
//! Requires the CVRPLIB X set: `bash data/instances/scripts/fetch_cvrp.sh`.
//! Prints the gap to the best-known solution, which is read from the sibling
//! `.sol` file for reporting only — never by the search itself.
//!
//! How to run:
//! ```
//! cargo build --profile profiling --example prof_hgs_vrp
//! samply record ./target/profiling/examples/prof_hgs_vrp
//! ```

use std::time::Duration;

use optopus::prelude::*;

/// Instances to profile, with the seconds of budget each gets.
const INSTANCES: &[(&str, u64)] = &[("X-n101-k25", 30), ("X-n195-k51", 30), ("X-n502-k39", 60)];

/// Reads `Cost <value>` from a CVRPLIB `.sol` file.
fn best_known(path: &str) -> Option<f64> {
    let text = std::fs::read_to_string(path).ok()?;
    text.lines()
        .find_map(|line| line.strip_prefix("Cost "))
        .and_then(|value| value.trim().parse().ok())
}

fn main() {
    for &(name, seconds) in INSTANCES {
        let path = format!("data/instances/vrp/{name}.vrp");
        let Ok(vrp) = Vrp::load_file(&path) else {
            eprintln!("skipping {name}: run data/instances/scripts/fetch_cvrp.sh first");
            continue;
        };

        let mut state = SearchState::new_with_seed(&vrp, 42);
        HybridGeneticSearchForVrp::new(
            StopCondition::duration(Duration::from_secs(seconds)),
            25,
            40,
            20,
            0.2,
            Some(20_000),
        )
        .run(&mut state)
        .unwrap();

        let best = state.best_solution.objective;
        let gap = best_known(&format!("data/instances/vrp/{name}.sol"))
            .map(|bks| format!("{:+.2}% vs BKS {bks}", 100.0 * (best - bks) / bks))
            .unwrap_or_else(|| "BKS unavailable".to_string());
        println!(
            "{name:<12} n={:<5} fleet={:<4} best={best:>10.1}  {gap}  ({} generations in {seconds}s)",
            vrp.get_n(),
            vrp.num_vehicles,
            state.iteration,
        );
    }
}
