//! Crossover for the problems of this module.

use super::formula::{FormulaProblem, FormulaSolution};
use super::problem::{ChangeDelta, IntSolution, IntVars, IntegerProblem, PairDelta};
use crate::common::permutation::order_crossover;
use crate::error::OptError;
use crate::search_state::Crossover;
use rand::Rng;
use rand::rngs::SmallRng;

/// Crossover for [`IntegerProblem`] and [`FormulaProblem`].
///
/// Each variable takes its value from either parent with equal probability,
/// or, on a [permutation](IntVars::permutation), the order crossover keeps a
/// segment of the first parent and fills the rest in the second parent's
/// order, so the child is a permutation again.
#[derive(Clone, Copy, Debug, Default)]
pub struct IntCrossover;

/// The values of a child of `a` and `b`.
fn child_values(vars: &IntVars, a: &[i64], b: &[i64], rng: &mut SmallRng) -> Vec<i64> {
    if vars.is_permutation() {
        let a: Vec<usize> = a.iter().map(|&v| v as usize).collect();
        let b: Vec<usize> = b.iter().map(|&v| v as usize).collect();
        order_crossover(&a, &b, rng)
            .into_iter()
            .map(|v| v as i64)
            .collect()
    } else {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| if rng.random_bool(0.5) { x } else { y })
            .collect()
    }
}

impl<F, D, S, R> Crossover<IntegerProblem<F, D, S, R>> for IntCrossover
where
    F: Fn(&[i64]) -> f64 + Sync,
    D: ChangeDelta,
    S: PairDelta,
    R: PairDelta,
{
    fn crossover(
        &mut self,
        prob: &IntegerProblem<F, D, S, R>,
        sol1: &IntSolution,
        sol2: &IntSolution,
        rng: &mut SmallRng,
    ) -> Result<IntSolution, OptError> {
        prob.solution_from(child_values(
            prob.variables(),
            sol1.values(),
            sol2.values(),
            rng,
        ))
    }
}

impl Crossover<FormulaProblem> for IntCrossover {
    fn crossover(
        &mut self,
        prob: &FormulaProblem,
        sol1: &FormulaSolution,
        sol2: &FormulaSolution,
        rng: &mut SmallRng,
    ) -> Result<FormulaSolution, OptError> {
        prob.solution_from(child_values(
            prob.variables(),
            sol1.values(),
            sol2.values(),
            rng,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::{
        GeneticAlgorithm, Heuristic, LocalSearch, ParentSelection, StopCondition,
        SubProblemBasedCrossover,
    };
    use crate::problem::{Constraint, ConstraintRel, Expr, IntChangeNeighbor, IntVar};
    use crate::search_state::{Distance, Evaluable, Evaluate, ProblemTrait, SearchState};
    use rand::SeedableRng;

    #[test]
    fn a_child_takes_every_value_from_a_parent() {
        let vars: IntVars = (0..20).map(|_| IntVar::new(0, 9)).collect();
        let prob = IntegerProblem::minimize(vars, |x: &[i64]| x.iter().sum::<i64>() as f64);
        let mut rng = SmallRng::seed_from_u64(1);
        let (a, b) = (prob.new_solution(&mut rng), prob.new_solution(&mut rng));
        let child = IntCrossover.crossover(&prob, &a, &b, &mut rng).unwrap();
        for (k, &v) in child.values().iter().enumerate() {
            assert!(v == a.values()[k] || v == b.values()[k]);
        }
        let same = IntCrossover.crossover(&prob, &a, &a, &mut rng).unwrap();
        assert_eq!(same.values(), a.values());
        assert_eq!(a.distance(&a), 0);
    }

    #[test]
    fn a_child_of_permutations_is_a_permutation() {
        let prob = IntegerProblem::minimize(IntVars::permutation(12), |x: &[i64]| x[0] as f64);
        let mut rng = SmallRng::seed_from_u64(2);
        for _ in 0..50 {
            let (a, b) = (prob.new_solution(&mut rng), prob.new_solution(&mut rng));
            let child = IntCrossover.crossover(&prob, &a, &b, &mut rng).unwrap();
            let mut sorted = child.values().to_vec();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..12).collect::<Vec<i64>>());
        }
    }

    #[test]
    fn a_formula_child_prices_its_moves_like_a_fresh_solution() {
        let vars: IntVars = (0..6).map(|_| IntVar::binary()).collect();
        let objective = Expr::Var(0) * Expr::Var(1) - 2.0 * Expr::Var(2) * Expr::Var(3)
            + Expr::Var(4)
            + Expr::Var(5);
        let prob = FormulaProblem::maximize(vars, objective);
        let mut rng = SmallRng::seed_from_u64(3);
        let (a, b) = (prob.new_solution(&mut rng), prob.new_solution(&mut rng));
        let child = IntCrossover.crossover(&prob, &a, &b, &mut rng).unwrap();
        let fresh = prob.solution_from(child.values().to_vec()).unwrap();
        assert!(matches!(
            (child.evaluate(), fresh.evaluate()),
            (Evaluable::Maximize(x), Evaluable::Maximize(y)) if x == y
        ));
    }

    #[test]
    fn the_genetic_algorithm_runs_on_an_integer_problem() {
        let vars: IntVars = (0..8).map(|_| IntVar::new(0, 5)).collect();
        let prob = IntegerProblem::minimize(vars, |x: &[i64]| {
            x.iter().map(|&v| ((v - 2) * (v - 2)) as f64).sum()
        });
        let mut state = SearchState::new_with_seed(&prob, 4);
        GeneticAlgorithm::new(
            StopCondition::iterations(200),
            16,
            IntCrossover,
            Box::new(LocalSearch::<IntChangeNeighbor>::new(
                StopCondition::failed_updates(1),
            )),
            ParentSelection::Tournament,
        )
        .run(&mut state)
        .unwrap();
        assert_eq!(state.best_solution.values(), &[2; 8]);
    }

    #[test]
    fn the_sub_problem_crossover_runs_on_a_formula() {
        // maximize the number of set bits, at most 4 of 10
        let vars: IntVars = (0..10).map(|_| IntVar::binary()).collect();
        let sum = || (0..10).map(Expr::Var).fold(Expr::Const(0.0), |a, b| a + b);
        let prob = FormulaProblem::maximize(vars, sum()).with_constraint(Constraint::Comparison {
            lhs: sum(),
            rel: ConstraintRel::Le,
            rhs: Expr::Const(4.0),
            penalty_weight: 10.0,
        });
        let local = || {
            Box::new(LocalSearch::<IntChangeNeighbor>::new(
                StopCondition::failed_updates(1),
            ))
        };
        let mut state = SearchState::new_with_seed(&prob, 5);
        GeneticAlgorithm::new(
            StopCondition::iterations(100),
            8,
            SubProblemBasedCrossover {
                sub_heuristic: local(),
            },
            local(),
            ParentSelection::Tournament,
        )
        .run(&mut state)
        .unwrap();
        assert!(matches!(state.best_solution.evaluate(), Evaluable::Maximize(v) if v == 4.0));
    }
}
