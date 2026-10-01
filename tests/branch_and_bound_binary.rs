//! Branch-and-bound on the binary problems, against enumeration: the folding
//! of fixed variables keeps the objective, the bound covers every assignment
//! of a node, and the optimum a native heuristic helps prove is the one
//! enumeration finds.

use optopus::common::{BinaryFixing, variable_slots};
use optopus::prelude::*;
use optopus::problem::{BranchSpace, EigenvalueRelaxation, Relaxation};
use optopus::trait_defs::{BinaryProblem, FixVariables, ProblemReduction};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn random_max_cut(rng: &mut SmallRng, n: usize) -> MaxCut {
    let mut edges = vec![];
    for i in 0..n {
        for j in i + 1..n {
            if rng.random_bool(0.5) {
                let w = [-3.0, -1.0, 1.0, 2.0, 5.0][rng.random_range(0..5)];
                edges.push((i, j, w));
            }
        }
    }
    MaxCut::new(Graph::from_edges(edges))
}

fn random_qubo(rng: &mut SmallRng, n: usize) -> Qubo {
    let mut entries = vec![];
    for i in 0..n {
        for j in i..n {
            if rng.random_bool(0.5) {
                entries.push((i, j, rng.random_range(-5..=5)));
            }
        }
    }
    Qubo::from_entries(entries)
}

/// Literals drawn with replacement, so some clauses hold a variable twice,
/// repeated or together with its negation.
fn random_sat(rng: &mut SmallRng, n: usize) -> Sat {
    let mut sat = Sat::new(n);
    for _ in 0..3 * n {
        let len = rng.random_range(1..=3);
        sat.add_clause((0..len).map(|_| {
            let var = rng.random_range(1..=n) as i64;
            if rng.random_bool(0.5) { var } else { -var }
        }));
    }
    sat
}

fn random_vertex_cover(rng: &mut SmallRng, n: usize) -> VertexCover {
    VertexCover::new(Graph::erdos_renyi(n, 0.35, rng))
}

/// A random partial assignment, every index that is not a variable at `false`.
fn random_fixing<P: BinaryProblem>(rng: &mut SmallRng, prob: &P) -> Vec<Option<bool>> {
    let mut fixed = vec![Some(false); variable_slots(prob)];
    for i in prob.variable_indices() {
        fixed[i] = if rng.random_bool(0.4) {
            None
        } else {
            Some(rng.random_bool(0.5))
        };
    }
    fixed
}

fn ranges_of(fixed: &[Option<bool>]) -> IntVars {
    fixed
        .iter()
        .map(|f| match f {
            Some(v) => IntVar::new(i64::from(*v), i64::from(*v)),
            None => IntVar::binary(),
        })
        .collect()
}

/// The objective, lower being better, of every assignment of `prob` that
/// agrees with `fixed`.
fn objectives_within<P: FixVariables>(prob: &P, fixed: &[Option<bool>]) -> Vec<f64> {
    let free: Vec<usize> = prob
        .variable_indices()
        .filter(|&i| fixed[i].is_none())
        .collect();
    (0..1u32 << free.len())
        .map(|mask| {
            let mut values: Vec<bool> = fixed.iter().map(|f| f.unwrap_or(false)).collect();
            for (k, &i) in free.iter().enumerate() {
                values[i] = mask >> k & 1 == 1;
            }
            prob.solution_from_assignment(&values)
                .evaluate()
                .minimized()
        })
        .collect()
}

fn best(objectives: &[f64]) -> f64 {
    objectives.iter().copied().fold(f64::INFINITY, f64::min)
}

/// The offset added to the folded objective, lower being better.
fn offset_minimized<P: FixVariables>(prob: &P, reduction: &BinaryFixing<P>) -> f64 {
    match prob.trivial_bound() {
        Evaluable::Maximize(_) => -reduction.offset(),
        Evaluable::Minimize(_) => reduction.offset(),
    }
}

/// For every solution of the folded instance, its lift keeps the fixed values
/// and its objective is the folded one plus the offset.
fn check_exact_folding<P: FixVariables>(prob: &P, fixed: &[Option<bool>]) {
    let reduction = BinaryFixing::new(prob.fix(fixed));
    let offset = offset_minimized(prob, &reduction);
    let target = reduction.target();
    let base = prob.solution_from_assignment(&vec![false; variable_slots(prob)]);
    let vars: Vec<usize> = target.variable_indices().collect();
    for mask in 0..1u32 << vars.len() {
        let mut values = vec![false; variable_slots(target)];
        for (k, &t) in vars.iter().enumerate() {
            values[t] = mask >> k & 1 == 1;
        }
        let folded = target.solution_from_assignment(&values);
        let lifted = reduction.lift(prob, &base, &folded);
        for (i, f) in fixed.iter().enumerate() {
            if let Some(v) = f {
                assert_eq!(P::variable(&lifted, i), *v, "fixed variable {i} moved");
            }
        }
        let whole = lifted.evaluate().minimized();
        let parts = folded.evaluate().minimized() + offset;
        assert!(
            (whole - parts).abs() < 1e-6,
            "lifted {whole} against folded plus offset {parts}"
        );
    }
}

/// The folded instance's best plus the offset is the node's best.
fn check_folding_keeps_the_optimum<P: FixVariables>(prob: &P, fixed: &[Option<bool>]) {
    let reduction = BinaryFixing::new(prob.fix(fixed));
    let offset = offset_minimized(prob, &reduction);
    let target = reduction.target();
    let folded_best = best(&objectives_within(
        target,
        &vec![None; variable_slots(target)],
    ));
    let node_best = best(&objectives_within(prob, fixed));
    assert!(
        (folded_best + offset - node_best).abs() < 1e-6,
        "folded best {folded_best} plus offset {offset} against the node's {node_best}"
    );
}

fn check_bound_covers_the_node<P: FixVariables>(prob: &P, fixed: &[Option<bool>]) {
    let bound = BinaryRelaxation.bound(prob, &ranges_of(fixed)).minimized();
    let node_best = best(&objectives_within(prob, fixed));
    assert!(
        bound <= node_best + 1e-6,
        "bound {bound} above the node's best {node_best}"
    );
}

/// Runs branch-and-bound with `inner` and `relaxation` and checks it proves
/// the enumerated optimum.
fn check_proves_optimum<P>(
    prob: &P,
    inner: Box<dyn Heuristic<P>>,
    relaxation: impl Relaxation<P>,
    seed: u64,
) where
    P: FixVariables + BranchSpace,
{
    let mut bnb = BranchAndBound::new(StopCondition::new(None, None, None), inner, relaxation);
    let mut state = SearchState::new_with_seed(prob, seed);
    bnb.run(&mut state).unwrap();
    let found = state.best_solution.evaluate().minimized();
    let enumerated = best(&objectives_within(prob, &vec![None; variable_slots(prob)]));
    assert!(
        (found - enumerated).abs() < 1e-6,
        "found {found}, enumeration {enumerated}"
    );
    assert!(bnb.is_proven_optimal(&state.best_solution));
}

#[test]
fn max_cut_folds_exactly_and_is_bounded() {
    let mut rng = SmallRng::seed_from_u64(1);
    for _ in 0..40 {
        let prob = random_max_cut(&mut rng, 9);
        let fixed = random_fixing(&mut rng, &prob);
        check_exact_folding(&prob, &fixed);
        check_bound_covers_the_node(&prob, &fixed);
    }
}

#[test]
fn qubo_folds_exactly_and_is_bounded() {
    let mut rng = SmallRng::seed_from_u64(2);
    for _ in 0..40 {
        let prob = random_qubo(&mut rng, 9);
        let fixed = random_fixing(&mut rng, &prob);
        check_exact_folding(&prob, &fixed);
        check_bound_covers_the_node(&prob, &fixed);
    }
}

#[test]
fn sat_folds_exactly_and_is_bounded() {
    let mut rng = SmallRng::seed_from_u64(3);
    for _ in 0..40 {
        let prob = random_sat(&mut rng, 9);
        let fixed = random_fixing(&mut rng, &prob);
        check_exact_folding(&prob, &fixed);
        check_bound_covers_the_node(&prob, &fixed);
    }
}

#[test]
fn vertex_cover_folding_keeps_the_optimum_and_is_bounded() {
    let mut rng = SmallRng::seed_from_u64(4);
    for _ in 0..40 {
        let prob = random_vertex_cover(&mut rng, 10);
        let fixed = random_fixing(&mut rng, &prob);
        check_folding_keeps_the_optimum(&prob, &fixed);
        check_bound_covers_the_node(&prob, &fixed);
    }
}

#[test]
fn max_cut_optimum_is_proven_with_native_heuristics() {
    let mut rng = SmallRng::seed_from_u64(5);
    for seed in 0..4 {
        let prob = random_max_cut(&mut rng, 13);
        check_proves_optimum(
            &prob,
            Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
                StopCondition::iterations(50),
            )),
            BinaryRelaxation,
            seed,
        );
        check_proves_optimum(
            &prob,
            Box::new(TabuSearch::<MaxCutFlipNeighbor>::new(
                StopCondition::iterations(30),
                (2, 4),
            )),
            BinaryRelaxation,
            seed,
        );
        check_proves_optimum(
            &prob,
            Box::new(bls_for_max_cut(
                StopCondition::iterations(50),
                (2, 4),
                5,
                2,
                0.5,
                0.5,
            )),
            BinaryRelaxation,
            seed,
        );
    }
}

#[test]
fn qubo_optimum_is_proven_with_native_heuristics() {
    let mut rng = SmallRng::seed_from_u64(6);
    for seed in 0..4 {
        let prob = random_qubo(&mut rng, 13);
        check_proves_optimum(
            &prob,
            Box::new(TabuSearch::<QuboFlipNeighbor>::new(
                StopCondition::iterations(30),
                (2, 4),
            )),
            BinaryRelaxation,
            seed,
        );
        check_proves_optimum(
            &prob,
            Box::new(SimulatedAnnealing::<QuboFlipNeighbor>::new(
                StopCondition::iterations(200),
                2.0,
                0.98,
            )),
            BinaryRelaxation,
            seed,
        );
    }
}

#[test]
fn sat_optimum_is_proven_with_native_heuristics() {
    let mut rng = SmallRng::seed_from_u64(7);
    for seed in 0..4 {
        let prob = random_sat(&mut rng, 13);
        check_proves_optimum(
            &prob,
            Box::new(LocalSearch::<SatFlipNeighbor>::new(
                StopCondition::iterations(50),
            )),
            BinaryRelaxation,
            seed,
        );
        check_proves_optimum(
            &prob,
            Box::new(WalkSatForSat::new(
                StopCondition::iterations(100),
                0.3,
                false,
            )),
            BinaryRelaxation,
            seed,
        );
    }
}

#[test]
fn vertex_cover_optimum_is_proven_with_native_heuristics() {
    let mut rng = SmallRng::seed_from_u64(8);
    for seed in 0..4 {
        let prob = random_vertex_cover(&mut rng, 14);
        check_proves_optimum(
            &prob,
            Box::new(LocalSearch::<VertexCoverFlipNeighbor>::new(
                StopCondition::iterations(50),
            )),
            BinaryRelaxation,
            seed,
        );
    }
}

#[test]
fn a_seeded_binary_run_is_reproducible() {
    let prob = random_max_cut(&mut SmallRng::seed_from_u64(9), 14);
    let run = || {
        let mut bnb = BranchAndBound::new(
            StopCondition::new(None, None, None),
            Box::new(SimulatedAnnealing::<MaxCutFlipNeighbor>::new(
                StopCondition::iterations(100),
                2.0,
                0.95,
            )),
            BinaryRelaxation,
        );
        let mut state = SearchState::new_with_seed(&prob, 3);
        bnb.run(&mut state).unwrap();
        (
            state.best_solution.x.clone(),
            state.iteration,
            state.best_iteration,
            state.n_accepted,
        )
    };
    assert_eq!(run(), run());
}

fn random_fractional_max_cut(rng: &mut SmallRng, n: usize) -> MaxCut {
    let mut edges = vec![];
    for i in 0..n {
        for j in i + 1..n {
            if rng.random_bool(0.5) {
                edges.push((i, j, rng.random_range(-2.0..4.0) as f32));
            }
        }
    }
    MaxCut::new(Graph::from_edges(edges))
}

#[test]
fn the_eigenvalue_bound_covers_every_node() {
    let mut rng = SmallRng::seed_from_u64(10);
    for case in 0..60 {
        let prob = if case % 2 == 0 {
            random_max_cut(&mut rng, 11)
        } else {
            random_fractional_max_cut(&mut rng, 11)
        };
        let fixed = if case % 3 == 0 {
            vec![None; variable_slots(&prob)]
        } else {
            random_fixing(&mut rng, &prob)
        };
        let node_best = best(&objectives_within(&prob, &fixed));
        let ranges = ranges_of(&fixed);
        let plain = EigenvalueRelaxation::new()
            .bound(&prob, &ranges)
            .minimized();
        // An incumbent at the node's best stops the descent early, which
        // must not make the bound invalid.
        let aimed = EigenvalueRelaxation::new()
            .bound_against(&prob, &ranges, Evaluable::Maximize(-node_best))
            .minimized();
        for bound in [plain, aimed] {
            assert!(
                bound <= node_best + 1e-6,
                "case {case}: bound {bound} above the node's best {node_best}"
            );
        }
        let positive = BinaryRelaxation.bound(&prob, &ranges).minimized();
        assert!(
            plain >= positive - 1e-9,
            "never looser than the positive weights"
        );
    }
}

#[test]
fn max_cut_optimum_is_proven_with_the_eigenvalue_bound() {
    let mut rng = SmallRng::seed_from_u64(11);
    for seed in 0..4 {
        let prob = random_max_cut(&mut rng, 14);
        let mut bnb = BranchAndBound::new(
            StopCondition::new(None, None, None),
            Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
                StopCondition::iterations(50),
            )),
            EigenvalueRelaxation::new(),
        );
        let mut state = SearchState::new_with_seed(&prob, seed);
        bnb.run(&mut state).unwrap();
        let enumerated = best(&objectives_within(
            &prob,
            &vec![None; variable_slots(&prob)],
        ));
        assert!((state.best_solution.evaluate().minimized() - enumerated).abs() < 1e-6);
        assert!(bnb.is_proven_optimal(&state.best_solution));
    }
}

/// A graph sized past its last vertex with an edge, as the generators leave
/// one, still gets solutions as long as the graph from a node.
#[test]
fn a_max_cut_with_trailing_isolated_vertices_keeps_full_length_solutions() {
    let mut graph = Graph::erdos_renyi(6, 0.0, &mut seeded_rng(0));
    graph.add_weight(0, 1, 1.0);
    graph.add_weight(1, 2, 2.0);
    let prob = MaxCut::new(graph);
    assert_eq!(variable_slots(&prob), 3);
    assert_eq!(prob.graph.len(), 6);

    let sol = prob.solution_from_assignment(&[true, false, true]);
    assert_eq!((sol.x.len(), sol.gain.len()), (6, 6));

    let mut bnb = BranchAndBound::new(
        StopCondition::new(None, None, None),
        Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
            StopCondition::iterations(10),
        )),
        BinaryRelaxation,
    );
    let mut state = SearchState::new_with_seed(&prob, 0);
    bnb.run(&mut state).unwrap();
    assert_eq!(state.best_solution.x.len(), 6);
    assert_eq!(state.best_solution.objective, 3.0);
    assert!(bnb.is_proven_optimal(&state.best_solution));
}
