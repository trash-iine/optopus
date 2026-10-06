//! Branch-and-bound against enumeration: the optimum it proves is the one
//! enumeration finds, whatever heuristic searches the nodes, and the interval
//! relaxation is a bound on every assignment it covers.

use optopus::prelude::*;
use optopus::problem::IntervalRelaxation;
use optopus::trait_defs::Relaxation;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Every assignment of `vars`, in lexicographic order.
fn assignments(vars: &IntVars) -> Vec<Vec<i64>> {
    vars.iter().fold(vec![vec![]], |acc, v| {
        acc.into_iter()
            .flat_map(|prefix| {
                (v.lower()..=v.upper()).map(move |x| {
                    let mut next = prefix.clone();
                    next.push(x);
                    next
                })
            })
            .collect()
    })
}

fn unbounded() -> StopCondition {
    StopCondition::new(None, None, None)
}

fn local_search() -> Box<dyn Heuristic<FormulaProblem>> {
    Box::new(LocalSearch::<IntChangeNeighbor>::new(
        StopCondition::iterations(50),
    ))
}

/// A quadratic over six variables in `0..=2`, with a knapsack-like
/// constraint and an equality, in the given direction.
fn quadratic(maximize: bool, seed: u64) -> FormulaProblem {
    let mut rng = SmallRng::seed_from_u64(seed);
    let n = 6;
    let vars: IntVars = (0..n).map(|_| IntVar::new(0, 2)).collect();
    let mut objective = Expr::Const(0.0);
    for i in 0..n {
        objective = objective + rng.random_range(-3.0..3.0) * Expr::Var(i);
        for j in i..n {
            if rng.random_bool(0.4) {
                objective = objective + rng.random_range(-2.0..2.0) * Expr::Var(i) * Expr::Var(j);
            }
        }
    }
    let prob = if maximize {
        FormulaProblem::maximize(vars, objective)
    } else {
        FormulaProblem::minimize(vars, objective)
    };
    let capacity = (0..n).fold(Expr::Const(0.0), |e, i| {
        e + ((i % 3 + 1) as f64) * Expr::Var(i)
    });
    prob.with_constraint(Constraint::Comparison {
        lhs: capacity,
        rel: ConstraintRel::Le,
        rhs: Expr::Const(8.0),
        penalty_weight: 5.0,
    })
    .with_constraint(Constraint::Comparison {
        lhs: Expr::Var(0) + Expr::Var(5),
        rel: ConstraintRel::Eq,
        rhs: Expr::Const(2.0),
        penalty_weight: 3.0,
    })
}

fn enumerated_best(prob: &FormulaProblem) -> f64 {
    assignments(prob.variables())
        .iter()
        .map(|x| prob.objective(x).minimized())
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn proves_the_enumerated_optimum_of_a_constrained_quadratic() {
    for seed in 0..6 {
        for maximize in [true, false] {
            let prob = quadratic(maximize, seed);
            let mut bnb = BranchAndBound::new(unbounded(), local_search(), IntervalRelaxation);
            let mut state = SearchState::new_with_seed(&prob, seed);
            bnb.run(&mut state).unwrap();
            let found = state.best_solution.evaluate().minimized();
            assert!(
                (found - enumerated_best(&prob)).abs() < 1e-9,
                "seed {seed}, maximize {maximize}: {found} against {}",
                enumerated_best(&prob)
            );
            assert!(bnb.is_proven_optimal(&state.best_solution));
            assert_eq!(
                bnb.dual_bound(&state.best_solution).minimized(),
                found,
                "a finished search closes the gap"
            );
        }
    }
}

#[test]
fn the_optimum_does_not_depend_on_the_inner_heuristic() {
    let prob = quadratic(true, 7);
    let best = enumerated_best(&prob);
    let inners: Vec<Box<dyn Heuristic<FormulaProblem>>> = vec![
        local_search(),
        Box::new(TabuSearch::<IntChangeNeighbor>::new(
            StopCondition::iterations(30),
            (2, 4),
        )),
        Box::new(SimulatedAnnealing::<IntChangeNeighbor>::new(
            StopCondition::iterations(200),
            2.0,
            0.98,
        )),
    ];
    for inner in inners {
        let mut bnb = BranchAndBound::new(unbounded(), inner, IntervalRelaxation);
        let mut state = SearchState::new_with_seed(&prob, 3);
        bnb.run(&mut state).unwrap();
        assert!((state.best_solution.evaluate().minimized() - best).abs() < 1e-9);
        assert!(bnb.is_proven_optimal(&state.best_solution));
    }
}

#[test]
fn a_seeded_run_is_reproducible() {
    let prob = quadratic(false, 11);
    let run = || {
        let mut bnb = BranchAndBound::new(
            unbounded(),
            Box::new(SimulatedAnnealing::<IntChangeNeighbor>::new(
                StopCondition::iterations(100),
                2.0,
                0.95,
            )),
            IntervalRelaxation,
        );
        let mut state = SearchState::new_with_seed(&prob, 5);
        bnb.run(&mut state).unwrap();
        (
            state.best_solution.values().to_vec(),
            state.iteration,
            state.best_iteration,
            state.n_accepted,
        )
    };
    assert_eq!(run(), run());
}

#[test]
fn a_stopped_search_bounds_what_it_has_not_seen() {
    let prob = quadratic(true, 2);
    let best = enumerated_best(&prob);
    let mut bnb = BranchAndBound::new(
        StopCondition::iterations(3),
        Box::new(LocalSearch::<IntChangeNeighbor>::new(
            StopCondition::iterations(1),
        )),
        IntervalRelaxation,
    );
    let mut state = SearchState::new_with_seed(&prob, 1);
    bnb.run(&mut state).unwrap();
    let dual = bnb.dual_bound(&state.best_solution).minimized();
    assert!(
        dual <= best + 1e-9,
        "the bound {dual} excludes the optimum {best}"
    );
    assert!(dual <= state.best_solution.evaluate().minimized());
}

/// 0-1 knapsack as an `IntegerProblem`, the overweight penalized at twice the
/// best value per weight, bounded by Dantzig's fractional fill.
#[test]
fn proves_a_knapsack_optimum_with_a_closure_bound() {
    let mut rng = SmallRng::seed_from_u64(42);
    let n = 14;
    let value: Vec<f64> = (0..n).map(|_| rng.random_range(1..30) as f64).collect();
    let weight: Vec<f64> = (0..n).map(|_| rng.random_range(1..20) as f64).collect();
    let capacity = weight.iter().sum::<f64>() / 2.0;
    let lambda = 2.0
        * value
            .iter()
            .zip(&weight)
            .map(|(v, w)| v / w)
            .fold(0.0, f64::max);
    let (value, weight) = (&value, &weight);

    let objective = move |x: &[i64]| {
        let (v, w) = x.iter().enumerate().fold((0.0, 0.0), |(v, w), (i, &xi)| {
            (v + xi as f64 * value[i], w + xi as f64 * weight[i])
        });
        v - lambda * (w - capacity).max(0.0)
    };
    let vars: IntVars = (0..n).map(|_| IntVar::binary()).collect();
    let prob = IntegerProblem::maximize(vars, objective);

    let dantzig = move |_: &_, vars: &IntVars| {
        let (mut v, mut w) = (0.0, 0.0);
        let mut free = vec![];
        for (i, r) in vars.iter().enumerate() {
            if r.lower() == 1 {
                v += value[i];
                w += weight[i];
            } else if r.upper() == 1 {
                free.push(i);
            }
        }
        if w > capacity {
            return Evaluable::Maximize(v - lambda * (w - capacity));
        }
        free.sort_by(|&a, &b| (value[b] / weight[b]).total_cmp(&(value[a] / weight[a])));
        let mut room = capacity - w;
        for i in free {
            if weight[i] <= room {
                v += value[i];
                room -= weight[i];
            } else {
                v += value[i] * room / weight[i];
                break;
            }
        }
        Evaluable::Maximize(v)
    };

    let mut bnb = BranchAndBound::new(
        unbounded(),
        Box::new(LocalSearch::<IntChangeNeighbor>::new(
            StopCondition::iterations(50),
        )),
        dantzig,
    );
    let mut state = SearchState::new_with_seed(&prob, 9);
    bnb.run(&mut state).unwrap();

    let best = assignments(prob.variables())
        .iter()
        .map(|x| objective(x))
        .fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(state.best_solution.evaluate().minimized(), -best);
    assert!(bnb.is_proven_optimal(&state.best_solution));
}

#[test]
fn the_interval_bound_covers_every_assignment_in_the_ranges() {
    let mut rng = SmallRng::seed_from_u64(0);
    for case in 0..200 {
        let maximize = case % 2 == 0;
        let prob = {
            let n = 4;
            let vars: IntVars = (0..n)
                .map(|_| {
                    let lo = rng.random_range(-3..=2);
                    IntVar::new(lo, lo + rng.random_range(0..=3))
                })
                .collect();
            let mut objective = Expr::Const(rng.random_range(-1.0..1.0));
            for _ in 0..4 {
                let mut term = Expr::Const(rng.random_range(-2.0..2.0));
                for _ in 0..rng.random_range(1..=3) {
                    term = term * Expr::Var(rng.random_range(0..n));
                }
                objective = objective + term;
            }
            let prob = if maximize {
                FormulaProblem::maximize(vars, objective)
            } else {
                FormulaProblem::minimize(vars, objective)
            };
            prob.with_constraint(Constraint::Clamp {
                expr: Expr::Var(0) * Expr::Var(1) - Expr::Var(2),
                lo: -1.0,
                hi: 2.0,
                penalty_weight: 1.5,
            })
            .with_constraint(Constraint::Comparison {
                lhs: Expr::Var(3) * Expr::Var(3),
                rel: ConstraintRel::Gt,
                rhs: Expr::Const(1.0),
                penalty_weight: 2.0,
            })
        };
        let narrowed: IntVars = prob
            .variables()
            .iter()
            .map(|v| {
                let lo = rng.random_range(v.lower()..=v.upper());
                IntVar::new(lo, rng.random_range(lo..=v.upper()))
            })
            .collect();
        let bound = IntervalRelaxation.bound(&prob, &narrowed).minimized();
        for x in assignments(&narrowed) {
            let value = prob.objective(&x).minimized();
            assert!(
                bound <= value + 1e-9,
                "case {case}: bound {bound} above {value} at {x:?}"
            );
        }
    }
}

#[test]
fn a_bound_in_the_wrong_direction_is_an_error() {
    let prob = quadratic(true, 0);
    let wrong = |_: &FormulaProblem, _: &IntVars| Evaluable::Minimize(0.0);
    let mut bnb = BranchAndBound::new(unbounded(), local_search(), wrong);
    let mut state = SearchState::new_with_seed(&prob, 0);
    assert!(bnb.run(&mut state).is_err());
}

#[test]
fn a_permutation_is_refused() {
    let prob = FormulaProblem::minimize(IntVars::permutation(4), Expr::Var(0));
    let mut bnb = BranchAndBound::new(unbounded(), local_search(), IntervalRelaxation);
    let mut state = SearchState::new_with_seed(&prob, 0);
    assert!(bnb.run(&mut state).is_err());
}

/// A problem of the user's own, implementing only `restricted`, so that
/// solutions cross between nodes through the default `solution_from_values`.
struct OwnProblem(IntegerProblem<fn(&[i64]) -> f64>);

impl ProblemTrait for OwnProblem {
    type Solution = IntSolution;
    fn new_solution(&self, rng: &mut impl rand::Rng) -> IntSolution {
        self.0.new_solution(rng)
    }
}

impl IntAssignment for OwnProblem {
    fn domains(&self) -> &IntVars {
        self.0.domains()
    }
    fn get(sol: &IntSolution, i: usize) -> i64 {
        sol.value(i)
    }
    fn assign(&self, sol: &mut IntSolution, i: usize, value: i64) {
        self.0.assign(sol, i, value)
    }
}

impl Branchable for OwnProblem {
    fn restricted(&self, vars: IntVars) -> Self {
        OwnProblem(self.0.restricted(vars))
    }
}

#[test]
fn a_problem_of_your_own_needs_only_restricted() {
    fn objective(x: &[i64]) -> f64 {
        let s: i64 = x.iter().enumerate().map(|(i, &v)| (i as i64 + 1) * v).sum();
        ((s - 17) * (s - 17)) as f64 + x.iter().filter(|&&v| v == 2).count() as f64
    }
    let vars: IntVars = (0..5).map(|_| IntVar::new(0, 3)).collect();
    let prob = OwnProblem(IntegerProblem::minimize(
        vars,
        objective as fn(&[i64]) -> f64,
    ));
    // Nothing is known about the objective beyond it being at least zero.
    let relaxation = |_: &OwnProblem, _: &IntVars| Evaluable::Minimize(0.0);

    let mut bnb = BranchAndBound::new(
        unbounded(),
        Box::new(LocalSearch::<IntChangeNeighbor>::new(
            StopCondition::iterations(20),
        )),
        relaxation,
    );
    let mut state = SearchState::new_with_seed(&prob, 4);
    bnb.run(&mut state).unwrap();

    let best = assignments(prob.domains())
        .iter()
        .map(|x| objective(x))
        .fold(f64::INFINITY, f64::min);
    assert_eq!(state.best_solution.evaluate().minimized(), best);
    assert_eq!(
        state.best_solution.evaluate().minimized(),
        objective(state.best_solution.values()),
        "the objective carried across nodes matches the values"
    );
    assert!(bnb.is_proven_optimal(&state.best_solution));
}
