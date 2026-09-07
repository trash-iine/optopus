//! Trajectory pins for `Alns<P>`, captured from the VRP-only
//! `AdaptiveLargeNeighborhoodSearchForVrp` it replaced.
//!
//! Before the specific implementation was deleted, both were run from the same
//! seeds on the same instance and compared on the whole trajectory rather than
//! the objective alone: the routes, the objective, both iteration counters and
//! the acceptance tallies all matched, over four seeds and two budgets. The
//! values below are that agreement, frozen. An operator bank that draws in a
//! different order, or a temperature that cools off by one step, moves at least
//! one of these while leaving the objective plausible.

use optopus::heuristic::{Alns, Heuristic, StopCondition};
use optopus::problem::Vrp;
use optopus::problem::vrp::AnchoredRouteDescent;
use optopus::search_state::SearchState;

fn instance() -> Vrp {
    Vrp::load_file("data/instances/vrp/demo16.vrp").expect("the committed fixture must load")
}

/// Everything a run leaves behind that a divergence could show up in.
type Trace = (Vec<Vec<usize>>, f64, u64, u64, u64, u64);

fn run(vrp: &Vrp, seed: u64, iterations: u64) -> Trace {
    let mut state = SearchState::new_with_seed(vrp, seed);
    let mut alns = Alns::<Vrp>::new(StopCondition::iterations(iterations), 0.15, 0.9995)
        .with_local_repair(Box::new(AnchoredRouteDescent::new()));
    alns.run(&mut state).unwrap();
    let best = &state.best_solution;
    (
        best.routes.clone(),
        best.objective,
        state.iteration,
        state.best_iteration,
        state.n_accepted,
        state.n_rejected,
    )
}

/// Prints the pins below. Ignored so it never runs as part of the suite.
#[test]
#[ignore = "capture helper: prints the ALNS pins"]
fn capture_pins() {
    let vrp = instance();
    for seed in [1u64, 7, 42, 2024] {
        for iterations in [200u64, 1_000] {
            let (routes, objective, iter, best_iter, accepted, rejected) =
                run(&vrp, seed, iterations);
            println!(
                "(seed {seed}, {iterations}) objective {objective:?} bits {:#x} iter {iter} best_iter {best_iter} accepted {accepted} rejected {rejected} routes {routes:?}",
                objective.to_bits()
            );
        }
    }
}

#[test]
fn the_alns_trajectories_are_pinned() {
    let vrp = instance();
    for (seed, iterations, objective_bits, iter, best_iter, accepted, rejected) in PINS {
        let (_, objective, got_iter, got_best_iter, got_accepted, got_rejected) =
            run(&vrp, seed, iterations);
        assert_eq!(
            objective.to_bits(),
            objective_bits,
            "objective, seed {seed}, {iterations} iterations"
        );
        assert_eq!(got_iter, iter, "iteration, seed {seed}");
        assert_eq!(got_best_iter, best_iter, "best_iteration, seed {seed}");
        assert_eq!(got_accepted, accepted, "n_accepted, seed {seed}");
        assert_eq!(got_rejected, rejected, "n_rejected, seed {seed}");
    }
}

/// `(seed, iterations, objective bits, iteration, best_iteration, accepted, rejected)`.
const PINS: [(u64, u64, u64, u64, u64, u64, u64); 8] = [
    (1, 200, 0x407a_a000_0000_0000, 200, 69, 200, 0),
    (1, 1_000, 0x407a_a000_0000_0000, 1_000, 69, 1_000, 0),
    (7, 200, 0x407a_a000_0000_0000, 200, 49, 200, 0),
    (7, 1_000, 0x407a_a000_0000_0000, 1_000, 49, 1_000, 0),
    (42, 200, 0x407a_a000_0000_0000, 200, 5, 200, 0),
    (42, 1_000, 0x407a_a000_0000_0000, 1_000, 5, 999, 1),
    (2024, 200, 0x407a_a000_0000_0000, 200, 11, 199, 1),
    (2024, 1_000, 0x407a_a000_0000_0000, 1_000, 11, 999, 1),
];
