//! Trajectory pins for `HybridGeneticSearchForVrp` on the homogeneous CVRP.
//!
//! Captured before the fleet-aware pricing replaced the distance-only route
//! machinery (`RouteState`, `Descent`, `split_giant_tour`), so that the
//! generalization can be checked against the search it replaces rather than
//! against a plausible objective. demo16 is easy enough that every seed
//! reaches the optimum in its first published generation, so the objective
//! alone would pin little; the routes, whose slot order and orientation
//! depend on the initial population, the split and the descent, are pinned
//! too.

use optopus::heuristic::{Heuristic, HybridGeneticSearchForVrp, StopCondition};
use optopus::problem::Vrp;
use optopus::search_state::SearchState;

fn instance() -> Vrp {
    Vrp::load_file("data/instances/vrp/demo16.vrp").expect("the committed fixture must load")
}

/// Everything a run leaves behind that a divergence could show up in.
type Trace = (Vec<Vec<usize>>, f64, u64, u64, u64);

fn run(vrp: &Vrp, seed: u64, generations: u64) -> Trace {
    let mut state = SearchState::new_with_seed(vrp, seed);
    let mut hgs =
        HybridGeneticSearchForVrp::new(StopCondition::iterations(generations), 6, 8, 4, 0.2, None);
    hgs.run(&mut state).unwrap();
    let best = &state.best_solution;
    (
        best.routes.clone(),
        best.objective,
        state.iteration,
        state.best_iteration,
        state.n_best_updates,
    )
}

/// Prints the pins below. Ignored so it never runs as part of the suite.
#[test]
#[ignore = "capture helper: prints the HGS pins"]
fn capture_pins() {
    let vrp = instance();
    for seed in [1u64, 7, 42, 2024] {
        for generations in [50u64, 200] {
            let (routes, objective, iter, best_iter, updates) = run(&vrp, seed, generations);
            println!(
                "(seed {seed}, {generations}) objective {objective:?} bits {:#x} iter {iter} best_iter {best_iter} best_updates {updates} routes {routes:?}",
                objective.to_bits()
            );
        }
    }
}

#[test]
fn the_hgs_trajectories_are_pinned() {
    let vrp = instance();
    for (seed, generations, objective_bits, iter, best_iter, updates, routes) in PINS {
        let (got_routes, objective, got_iter, got_best_iter, got_updates) =
            run(&vrp, seed, generations);
        assert_eq!(
            objective.to_bits(),
            objective_bits,
            "objective, seed {seed}, {generations} generations"
        );
        assert_eq!(got_iter, iter, "iteration, seed {seed}");
        assert_eq!(got_best_iter, best_iter, "best_iteration, seed {seed}");
        assert_eq!(got_updates, updates, "n_best_updates, seed {seed}");
        assert_eq!(
            got_routes, routes,
            "routes, seed {seed}, {generations} generations"
        );
    }
}

/// `(seed, generations, objective bits, iteration, best_iteration, n_best_updates, routes)`.
type Pin = (u64, u64, u64, u64, u64, u64, &'static [&'static [usize]]);

const PINS: [Pin; 8] = [
    (
        1,
        50,
        0x407a_a000_0000_0000,
        50,
        24,
        1,
        &[
            &[12, 9],
            &[10, 11, 7],
            &[13, 1],
            &[2, 15, 14],
            &[3, 4, 6],
            &[5, 8],
        ],
    ),
    (
        1,
        200,
        0x407a_a000_0000_0000,
        200,
        24,
        1,
        &[
            &[12, 9],
            &[10, 11, 7],
            &[13, 1],
            &[2, 15, 14],
            &[3, 4, 6],
            &[5, 8],
        ],
    ),
    (
        7,
        50,
        0x407a_a000_0000_0000,
        50,
        24,
        1,
        &[
            &[3, 4, 6],
            &[9, 12],
            &[10, 11, 7],
            &[2, 15, 14],
            &[5, 8],
            &[13, 1],
        ],
    ),
    (
        7,
        200,
        0x407a_a000_0000_0000,
        200,
        24,
        1,
        &[
            &[3, 4, 6],
            &[9, 12],
            &[10, 11, 7],
            &[2, 15, 14],
            &[5, 8],
            &[13, 1],
        ],
    ),
    (
        42,
        50,
        0x407a_a000_0000_0000,
        50,
        24,
        1,
        &[
            &[5, 8],
            &[14, 15, 2],
            &[13, 1],
            &[3, 4, 6],
            &[7, 11, 10],
            &[12, 9],
        ],
    ),
    (
        42,
        200,
        0x407a_a000_0000_0000,
        200,
        24,
        1,
        &[
            &[5, 8],
            &[14, 15, 2],
            &[13, 1],
            &[3, 4, 6],
            &[7, 11, 10],
            &[12, 9],
        ],
    ),
    (
        2024,
        50,
        0x407a_a000_0000_0000,
        50,
        24,
        1,
        &[
            &[3, 4, 6],
            &[9, 12],
            &[13, 1],
            &[7, 11, 10],
            &[5, 8],
            &[2, 15, 14],
        ],
    ),
    (
        2024,
        200,
        0x407a_a000_0000_0000,
        200,
        24,
        1,
        &[
            &[3, 4, 6],
            &[9, 12],
            &[13, 1],
            &[7, 11, 10],
            &[5, 8],
            &[2, 15, 14],
        ],
    ),
];
