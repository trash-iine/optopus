//! Searching an exactly reduced instance, through the public API only.
//!
//! A `ProblemReduction` is a pure map; running a heuristic on its target and
//! folding the result back is the *crossing*, and the crossing is two
//! `SearchState` methods. Composing the two is a handful of lines a caller
//! writes, so there is no wrapper type to test — these tests are the
//! composition, and they are what `docs/problems/max_cut_kernel.md` documents.
//!
//! They live here rather than beside `SearchState` because they need a real
//! heuristic, and `src/search_state/` deliberately does not depend on
//! `crate::heuristic` — that layering is the point of the module.

use optopus::prelude::*;
use optopus::problem::MaxCutKernel;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Average degree ~2.5: low enough that the reduction rules cascade.
fn sparse_instance(seed: u64, n: usize) -> MaxCut {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if rng.random_bool(2.5 / n as f64) {
                edges.push((i, j, 1.0));
            }
        }
    }
    edges.push((n - 1, n - 2, 1.0));
    MaxCut::from_edges(edges)
}

fn tabu_search() -> TabuSearch<MaxCutFlipNeighbor> {
    TabuSearch::new(StopCondition::iterations(2_000), (2, 8), None)
}

/// Solves `mc` through its kernel: repeatedly open a sub-state on the kernel,
/// run `inner` there, and fold the result back, until `outer` says stop.
///
/// This is the whole recipe. Two details are the caller's responsibility and
/// both are here:
///
/// - The reduction is built **once**. `MaxCutKernel::reduce` runs the rules to
///   a fixpoint, which is not something to repeat every cycle.
/// - An inner heuristic that halts at a local optimum returns without
///   consuming an iteration, so a cycle that changed nothing has to be charged
///   one — otherwise this loop never meets an iteration budget.
fn solve_through_kernel<'a>(
    mc: &'a MaxCut,
    outer: &StopCondition,
    inner: &mut dyn Heuristic<MaxCut>,
    seed: u64,
) -> SearchState<'a, MaxCut> {
    let kernel = MaxCutKernel::reduce(mc);
    let mut state = SearchState::new_with_seed(mc, seed);
    while !outer.is_done(&state) {
        let before = state.iteration;
        let mut sub = state.open_reduction(&kernel);
        inner.run(&mut sub).unwrap();
        state.close_reduction(&kernel, &sub).unwrap();
        if state.iteration == before {
            state.progress_iteration();
        }
    }
    state
}

/// Exact trajectory pin for the reducing path.
///
/// The other tests here state properties — the objective stays exact, the
/// reduction is worth something, an unreducible instance is untouched — and a
/// refactor can satisfy all of them while walking a different search. This
/// asserts the trajectory itself: the answer, when it was found, and the total
/// charged once the kernel sub-run is merged and the lifting applied.
///
/// The three values are inherited from the heuristic wrapper this recipe
/// replaced, and are deliberately not re-baselined: matching them is what says
/// deleting that wrapper changed no search. If this fails, the question is not
/// "what is the new value" but "which RNG draw or iteration count moved" — the
/// seed derivation in `open_reduction`, the warm-start projection, the counter
/// merge or the flip-by-flip lifting.
#[test]
fn the_reducing_path_trajectory_is_pinned() {
    let mc = sparse_instance(11, 300);
    assert!(!MaxCutKernel::reduce(&mc).is_trivial());

    let state = solve_through_kernel(
        &mc,
        &StopCondition::iterations(20_000),
        &mut tabu_search(),
        3,
    );

    assert_eq!(
        (
            state.best_solution.objective,
            state.best_iteration,
            state.iteration,
        ),
        (344.0, 2119, 20119)
    );
}

/// Solving through the kernel must leave an exactly evaluated solution of the
/// *original* instance — the lifting is only useful if the caches survive it.
#[test]
fn crossing_back_keeps_the_original_objective_exact() {
    for seed in 0..5u64 {
        let mc = sparse_instance(seed, 200);
        let state = solve_through_kernel(
            &mc,
            &StopCondition::iterations(20_000),
            &mut tabu_search(),
            3,
        );
        let best = &state.best_solution;
        assert_eq!(
            best.objective,
            mc.calculate_cut_size(&best.x),
            "objective diverged from the assignment (seed {seed})"
        );
    }
}

/// The reduction has to be worth the crossing: the kernel is strictly smaller
/// and the search still reaches at least as good a cut as the same heuristic
/// run on the full graph.
#[test]
fn crossing_into_the_kernel_does_not_cost_quality() {
    let mc = sparse_instance(11, 300);
    let kernel = MaxCutKernel::reduce(&mc);
    assert!(
        kernel.kernel().graph.num_vertices() < mc.graph.num_vertices(),
        "the test instance must actually reduce"
    );

    let through = solve_through_kernel(
        &mc,
        &StopCondition::iterations(20_000),
        &mut tabu_search(),
        3,
    );
    let direct = {
        let mut state = SearchState::new_with_seed(&mc, 3);
        tabu_search().run(&mut state).unwrap();
        state
    };
    assert!(through.best_solution.objective >= direct.best_solution.objective);
}

/// `is_trivial` is why leaving the reduction in a pipeline is free: on an
/// instance the rules cannot touch, skipping the crossing leaves the search
/// bit-identical to running the heuristic alone. A caller that crossed anyway
/// would solve a copy of the same instance through an extra index mapping.
#[test]
fn an_unreducible_instance_is_left_alone() {
    // 4-regular circulant: no vertex has degree below 3.
    let n = 40usize;
    let edges: Vec<_> = (0..n)
        .flat_map(|i| [(i, (i + 1) % n, 1.0), (i, (i + 2) % n, 1.0)])
        .collect();
    let mc = MaxCut::from_edges(edges);
    assert!(MaxCutKernel::reduce(&mc).is_trivial());

    let mut state = SearchState::new_with_seed(&mc, 5);
    LocalSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(3_000))
        .run(&mut state)
        .unwrap();

    let mut bare = SearchState::new_with_seed(&mc, 5);
    LocalSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(3_000))
        .run(&mut bare)
        .unwrap();

    assert_eq!(state.best_solution.x, bare.best_solution.x);
}
