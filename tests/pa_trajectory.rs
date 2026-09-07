//! Trajectory pins for `PopulationAnnealing<P, N>`.
//!
//! Resampling is where this search is most likely to drift without saying so.
//! The two loops that restore the population to exactly `R` pick a replica to
//! drop and one to copy, and on a plateau both are choosing between equals, so
//! the tie-break alone decides the trajectory.
//!
//! One instance is binary and one is not. Population annealing reads an energy
//! and a move, never a variable, and the TSP pin is what says so: it is not a
//! tuned or measured combination, and nothing here should be read as a
//! recommendation to use it.

use optopus::heuristic::{Heuristic, PopulationAnnealing, StopCondition};
use optopus::problem::MaxCut;
use optopus::problem::max_cut::MaxCutFlipNeighbor;
use optopus::problem::tsp_2d::{TspTwoOptNeighbor, TspWithCoordinates};
use optopus::search_state::SearchState;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// A plateau-rich instance: unit weights on a ring lattice put many replicas on
/// the same cut, which is exactly when a tie-break decides the trajectory.
fn ring_lattice(n: usize) -> MaxCut {
    let mut edges = Vec::new();
    for i in 0..n {
        edges.push((i, (i + 1) % n, 1.0));
        edges.push((i, (i + 2) % n, 1.0));
    }
    MaxCut::from_edges(edges)
}

fn scattered_cities(n: usize) -> TspWithCoordinates {
    let mut rng = SmallRng::seed_from_u64(77);
    let coordinates = (0..n)
        .map(|_| (rng.random_range(0.0..100.0), rng.random_range(0.0..100.0)))
        .collect();
    TspWithCoordinates::new("pa-pin".to_string(), coordinates)
}

fn digest<T: std::hash::Hash>(items: &[T]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    items.len().hash(&mut h);
    for item in items {
        item.hash(&mut h);
    }
    h.finish()
}

fn run_max_cut(mc: &MaxCut, seed: u64) -> (u32, u64, u64, u64) {
    let mut state = SearchState::new_with_seed(mc, seed);
    let mut pa = PopulationAnnealing::<MaxCut, MaxCutFlipNeighbor>::new(
        StopCondition::iterations(2_000),
        16,
        0.1,
        0.05,
        5,
        Some(20),
    );
    pa.run(&mut state).unwrap();
    (
        state.best_solution.objective.to_bits(),
        state.iteration,
        state.best_iteration,
        digest(&state.best_solution.x),
    )
}

fn run_tsp(tsp: &TspWithCoordinates, seed: u64) -> (u64, u64, u64, u64) {
    let mut state = SearchState::new_with_seed(tsp, seed);
    // 2-opt has an O(n²) neighborhood, so the sweep length is pinned rather
    // than counted.
    let mut pa = PopulationAnnealing::<TspWithCoordinates, TspTwoOptNeighbor>::new(
        StopCondition::iterations(1_000),
        8,
        0.05,
        0.02,
        5,
        Some(50),
    )
    .with_sweep_length(30);
    pa.run(&mut state).unwrap();
    (
        state.best_solution.objective.to_bits(),
        state.iteration,
        state.best_iteration,
        digest(&state.best_solution.tour),
    )
}

/// Prints the pins below. Ignored so it never runs as part of the suite.
#[test]
#[ignore = "capture helper: prints the PA pins"]
fn capture_pins() {
    let mc = ring_lattice(40);
    for seed in [1u64, 42, 2024] {
        let (bits, iter, best_iter, d) = run_max_cut(&mc, seed);
        println!("MAXCUT ({seed}, {bits:#x}, {iter}, {best_iter}, {d:#x}),");
    }
    let tsp = scattered_cities(30);
    for seed in [1u64, 42, 2024] {
        let (bits, iter, best_iter, d) = run_tsp(&tsp, seed);
        println!("TSP ({seed}, {bits:#x}, {iter}, {best_iter}, {d:#x}),");
    }
}

#[test]
fn the_max_cut_trajectories_are_pinned() {
    let mc = ring_lattice(40);
    for (seed, objective_bits, iter, best_iter, want_digest) in MAX_CUT_PINS {
        let got = run_max_cut(&mc, seed);
        assert_eq!(
            got,
            (objective_bits, iter, best_iter, want_digest),
            "MaxCut trajectory, seed {seed}"
        );
    }
}

#[test]
fn the_non_binary_trajectories_are_pinned() {
    let tsp = scattered_cities(30);
    for (seed, objective_bits, iter, best_iter, want_digest) in TSP_PINS {
        let got = run_tsp(&tsp, seed);
        assert_eq!(
            got,
            (objective_bits, iter, best_iter, want_digest),
            "TSP trajectory, seed {seed}"
        );
    }
}

/// `(seed, objective bits, iteration, best_iteration, assignment digest)`.
const MAX_CUT_PINS: [(u64, u32, u64, u64, u64); 3] = [
    (1, 0x4268_0000, 2_000, 90, 0x40dc_954b_4fa2_9153),
    (42, 0x4268_0000, 2_000, 1_290, 0x5966_20b7_7081_e600),
    (2024, 0x4268_0000, 2_000, 980, 0xca10_3386_c610_c66d),
];

/// `(seed, objective bits, iteration, best_iteration, tour digest)`.
const TSP_PINS: [(u64, u64, u64, u64, u64); 3] = [
    (1, 0x4079_f6ac_7643_0e01, 1_000, 610, 0x97b5_f7d9_09eb_972b),
    (42, 0x4079_f6ac_7643_0dea, 1_000, 850, 0xc78e_03c0_ead0_8dfe),
    (
        2024,
        0x4079_f6ac_7643_0de7,
        1_000,
        685,
        0x7728_958c_b3a4_97f2,
    ),
];
