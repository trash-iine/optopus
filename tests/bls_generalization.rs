//! Trajectory pins for Breakout Local Search, captured from the MaxCut-only
//! implementation before it was made generic over the problem.
//!
//! The generalization replaces the hand-written round with the same three
//! heuristics driven through `Box<dyn Heuristic<P>>`, so it has to select the
//! same moves in the same order. These pin the whole trajectory rather than the
//! objective alone: a perturbation that ran one step short, or an operator bank
//! indexed in a different order, moves an iteration counter while leaving the
//! cut plausible.

use optopus::common::{Graph, seeded_rng};
use optopus::heuristic::{Heuristic, StopCondition, bls_for_max_cut};
use optopus::problem::MaxCut;
use optopus::search_state::SearchState;

/// Two shapes with different tie structure: a unit-weight ring lattice, where
/// hundreds of vertices share a gain, and a weighted random graph, where almost
/// none do.
fn ring_lattice(n: usize) -> MaxCut {
    let mut edges = Vec::new();
    for i in 0..n {
        edges.push((i, (i + 1) % n, 1.0));
        edges.push((i, (i + 2) % n, 1.0));
    }
    MaxCut::from_edges(edges)
}

fn weighted(n: usize, seed: u64) -> MaxCut {
    let mut rng = seeded_rng(seed);
    MaxCut::new(Graph::erdos_renyi(n, 0.05, &mut rng).with_random_weights((1, 10), &mut rng))
}

/// Everything a run leaves behind that a divergence could show up in. The
/// assignment is folded to a digest so a `const` can hold it.
type Trace = (f32, u64, u64, u64);

fn digest(x: &[bool]) -> u64 {
    x.iter()
        .enumerate()
        .filter(|&(_, &v)| v)
        .map(|(i, _)| (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .fold(0u64, |a, b| a ^ b)
}

/// The two instances the pins are taken on, by the name the pins use.
fn instance(name: &str) -> MaxCut {
    match name {
        "ring" => ring_lattice(60),
        "weighted" => weighted(80, 5),
        other => panic!("no pinned instance named {other}"),
    }
}

fn run(mc: &MaxCut, seed: u64, iterations: u64) -> Trace {
    let mut state = SearchState::new_with_seed(mc, seed);
    let mut bls = bls_for_max_cut(
        StopCondition::iterations(iterations),
        (10, 30),
        1_000,
        8,
        0.8,
        0.5,
    );
    bls.run(&mut state).unwrap();
    (
        state.best_solution.objective,
        state.iteration,
        state.best_iteration,
        digest(&state.best_solution.x),
    )
}

/// Prints the pins below. Ignored so it never runs as part of the suite.
#[test]
#[ignore = "capture helper: prints the BLS pins"]
fn capture_pins() {
    for name in ["ring", "weighted"] {
        let mc = instance(name);
        for seed in [1u64, 42, 2024] {
            let (objective, iter, best_iter, digest) = run(&mc, seed, 3_000);
            println!(
                "({name}, seed {seed}) objective {objective:?} bits {:#x} iter {iter} best_iter {best_iter} digest {digest:#x}",
                objective.to_bits()
            );
        }
    }
}

#[test]
fn the_bls_trajectories_are_pinned() {
    for (name, seed, objective_bits, iter, best_iter, digest) in PINS {
        let (objective, got_iter, got_best_iter, got_digest) = run(&instance(name), seed, 3_000);
        assert_eq!(
            objective.to_bits(),
            objective_bits,
            "objective, {name}/{seed}"
        );
        assert_eq!(got_iter, iter, "iteration, {name}/{seed}");
        assert_eq!(got_best_iter, best_iter, "best_iteration, {name}/{seed}");
        assert_eq!(got_digest, digest, "assignment, {name}/{seed}");
    }
}

/// `(instance, seed, objective bits, iteration, best_iteration, assignment digest)`.
const PINS: [(&str, u64, u32, u64, u64, u64); 6] = [
    ("ring", 1, 0x42b4_0000, 3_005, 1_086, 0xbf47_f8ea_9b3e_84fd),
    ("ring", 42, 0x42b0_0000, 3_013, 102, 0x7ff9_cd07_8661_1857),
    (
        "ring",
        2024,
        0x42b4_0000,
        3_004,
        1_249,
        0xe1c2_afbf_b765_dc57,
    ),
    (
        "weighted",
        1,
        0x443f_8000,
        3_016,
        1_192,
        0xfac4_0fdb_9f78_2f1f,
    ),
    (
        "weighted",
        42,
        0x443f_8000,
        3_016,
        732,
        0x8a1d_bfa4_7fae_6a1f,
    ),
    (
        "weighted",
        2024,
        0x443f_8000,
        3_001,
        698,
        0x8a1d_bfa4_7fae_6a1f,
    ),
];
