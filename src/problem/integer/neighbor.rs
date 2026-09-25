//! The single variable change move for every [`IntegerProblem`].

use super::problem::{IntSolution, IntegerProblem, raw, with_value};
use crate::{
    common::TabuMemory,
    error::OptError,
    search_state::{EnabledTabu, Evaluable, Evaluate, MoveToNeighbor},
};
use rand::Rng;
use rand::rngs::SmallRng;

/// Sets variable `var` to `value`, any value in its range other than the
/// current one. On a binary variable this is a flip.
///
/// `gain` is the change in objective the move makes, with the direction of
/// the solution it was built from.
#[derive(Debug, Clone, Copy)]
pub struct IntChangeNeighbor {
    /// The variable to change.
    pub var: usize,
    /// The value it takes.
    pub value: i64,
    /// Change in objective after the move.
    pub gain: Evaluable<f64>,
}

impl IntChangeNeighbor {
    /// Builds the move setting `var` to `value`, its gain from
    /// [`IntegerProblem::delta`]. Every construction site goes through here.
    pub fn new<P: IntegerProblem>(prob: &P, sol: &IntSolution, var: usize, value: i64) -> Self {
        Self {
            var,
            value,
            gain: with_value(sol.evaluate(), prob.delta(sol, var, value)),
        }
    }
}

impl Evaluate for IntChangeNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        self.gain
    }
}

impl EnabledTabu for IntChangeNeighbor {
    /// The move is tabu while variable `var` is still blocked at the current
    /// iteration.
    fn is_move_enabled(&self, tabu: &TabuMemory, iteration: u64) -> bool {
        tabu.is_enabled(self.var, iteration)
    }

    /// Applying the move forbids variable `var` for a tenure the memory draws.
    fn add_to_tabu_map(&self, tabu: &mut TabuMemory, iteration: u64, rng: &mut SmallRng) {
        tabu.forbid(self.var, iteration, rng);
    }
}

impl<P: IntegerProblem> MoveToNeighbor<P> for IntChangeNeighbor {
    /// Hands this move's [`EnabledTabu`] policy to the search state, which is
    /// what holds the tabu map.
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

    /// Moves the cached objective by the gain, without evaluating it again.
    fn apply_to_solution(&self, _prob: &P, sol: &mut IntSolution) -> Result<(), OptError> {
        sol.set(self.var, self.value, raw(self.gain));
        Ok(())
    }

    fn iter(prob: &P, sol: &IntSolution) -> impl Iterator<Item = Self> + Send {
        prob.variables()
            .iter()
            .enumerate()
            .flat_map(move |(var, v)| {
                let cur = sol.value(var);
                (v.lower()..=v.upper())
                    .filter(move |&value| value != cur)
                    .map(move |value| Self::new(prob, sol, var, value))
            })
    }

    fn move_to_be_better_than(&self, _: &P, src: &IntSolution, other: &IntSolution) -> bool {
        self.evaluate()
            .improves_over(src.evaluate(), other.evaluate())
    }

    /// `O(log n)`. A variable is drawn with weight equal to its number of
    /// changes, then a value other than its current one uniformly, so every
    /// move [`iter`](MoveToNeighbor::iter) yields is equally likely.
    fn random_neighbor(prob: &P, sol: &IntSolution, rng: &mut SmallRng) -> Option<Self> {
        let vars = prob.variables();
        let total = vars.total_changes();
        if total == 0 {
            return None;
        }
        let var = vars.var_of_change(rng.random_range(0..total));
        let v = vars[var];
        let cur = sol.value(var);
        // Uniform in `lower..=upper` excluding the current value.
        let mut value = v
            .lower()
            .checked_add_unsigned(rng.random_range(0..v.num_changes()))
            .expect("stays below upper");
        if value >= cur {
            value += 1;
        }
        Some(Self::new(prob, sol, var, value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::{Heuristic, LocalSearch, StopCondition, TabuSearch};
    use crate::problem::{IntVar, IntVars};
    use crate::search_state::{ProblemTrait, SearchState};
    use rand::SeedableRng;

    /// Minimizes the sum of (x_i - t_i)^2, with ranges of different widths and
    /// one fixed variable. Keeps the default `delta`.
    struct Target {
        vars: IntVars,
        targets: Vec<i64>,
    }

    impl Target {
        fn new() -> Self {
            Self {
                vars: [(0, 5), (-3, 3), (7, 7), (0, 1), (-10, 20)]
                    .into_iter()
                    .map(|(l, u)| IntVar::new(l, u))
                    .collect(),
                targets: vec![4, -2, 7, 1, 13],
            }
        }

        fn term(&self, i: usize, x: i64) -> f64 {
            ((x - self.targets[i]) * (x - self.targets[i])) as f64
        }
    }

    impl IntegerProblem for Target {
        fn variables(&self) -> &IntVars {
            &self.vars
        }
        fn objective(&self, values: &[i64]) -> Evaluable<f64> {
            Evaluable::Minimize(
                values
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| self.term(i, x))
                    .sum(),
            )
        }
    }

    /// The same problem with an incremental `delta`.
    struct Fast(Target);

    impl IntegerProblem for Fast {
        fn variables(&self) -> &IntVars {
            &self.0.vars
        }
        fn objective(&self, values: &[i64]) -> Evaluable<f64> {
            self.0.objective(values)
        }
        fn delta(&self, sol: &IntSolution, i: usize, value: i64) -> f64 {
            self.0.term(i, value) - self.0.term(i, sol.value(i))
        }
    }

    fn rng(seed: u64) -> SmallRng {
        SmallRng::seed_from_u64(seed)
    }

    #[test]
    fn new_solution_is_in_range_and_caches_its_objective() {
        let prob = Fast(Target::new());
        for seed in 0..20 {
            let sol = prob.new_solution(&mut rng(seed));
            for (v, &x) in prob.0.vars.iter().zip(sol.values()) {
                assert!(v.contains(x));
            }
            assert_eq!(raw(sol.evaluate()), raw(prob.objective(sol.values())));
        }
    }

    #[test]
    fn iter_yields_every_other_value_of_every_variable() {
        let prob = Fast(Target::new());
        let sol = prob.new_solution(&mut rng(3));
        let moves: Vec<_> = IntChangeNeighbor::iter(&prob, &sol).collect();
        assert_eq!(moves.len() as u64, prob.0.vars.total_changes());
        assert_eq!(prob.0.vars.total_changes(), 42); // 5 + 6 + 0 + 1 + 30
        assert!(moves.iter().all(|m| m.value != sol.value(m.var)));
    }

    #[test]
    fn applying_a_move_keeps_the_cached_objective_exact() {
        let prob = Fast(Target::new());
        let mut sol = prob.new_solution(&mut rng(5));
        let mut r = rng(6);
        for _ in 0..200 {
            let m = IntChangeNeighbor::random_neighbor(&prob, &sol, &mut r).unwrap();
            m.apply_to_solution(&prob, &mut sol).unwrap();
            assert_eq!(raw(sol.evaluate()), raw(prob.objective(sol.values())));
        }
    }

    #[test]
    fn default_delta_agrees_with_the_incremental_one() {
        let fast = Fast(Target::new());
        let slow = Target::new();
        let sol = fast.new_solution(&mut rng(9));
        for (a, b) in IntChangeNeighbor::iter(&fast, &sol).zip(IntChangeNeighbor::iter(&slow, &sol))
        {
            assert_eq!(raw(a.gain), raw(b.gain));
        }
    }

    #[test]
    fn random_neighbor_is_uniform_over_iter() {
        let prob = Fast(Target::new());
        let sol = prob.new_solution(&mut rng(11));
        let moves: Vec<_> = IntChangeNeighbor::iter(&prob, &sol)
            .map(|m| (m.var, m.value))
            .collect();
        let mut counts = std::collections::HashMap::new();
        let mut r = rng(12);
        let draws = 42_000;
        for _ in 0..draws {
            let m = IntChangeNeighbor::random_neighbor(&prob, &sol, &mut r).unwrap();
            *counts.entry((m.var, m.value)).or_insert(0usize) += 1;
        }
        assert_eq!(counts.len(), moves.len());
        let expected = draws / moves.len();
        for key in &moves {
            let c = counts[key];
            assert!(
                c.abs_diff(expected) < expected / 4,
                "{key:?} drawn {c} times"
            );
        }
    }

    #[test]
    fn all_fixed_variables_have_no_neighbor() {
        struct Fixed(IntVars);
        impl IntegerProblem for Fixed {
            fn variables(&self) -> &IntVars {
                &self.0
            }
            fn objective(&self, _: &[i64]) -> Evaluable<f64> {
                Evaluable::Maximize(0.0)
            }
        }
        let prob = Fixed(
            [IntVar::new(2, 2), IntVar::new(-1, -1)]
                .into_iter()
                .collect(),
        );
        let sol = prob.new_solution(&mut rng(0));
        assert!(IntChangeNeighbor::random_neighbor(&prob, &sol, &mut rng(1)).is_none());
        assert_eq!(IntChangeNeighbor::iter(&prob, &sol).count(), 0);
    }

    #[test]
    fn solution_from_rejects_bad_values() {
        let prob = Fast(Target::new());
        assert!(prob.solution_from(vec![0; 4]).is_err());
        assert!(prob.solution_from(vec![0, 0, 6, 0, 0]).is_err());
        let sol = prob.solution_from(vec![4, -2, 7, 1, 13]).unwrap();
        assert_eq!(raw(sol.evaluate()), 0.0);
    }

    #[test]
    fn local_search_and_tabu_search_reach_the_optimum() {
        let prob = Fast(Target::new());
        let mut state = SearchState::new_with_seed(&prob, 1);
        LocalSearch::<IntChangeNeighbor>::new(StopCondition::iterations(50))
            .run(&mut state)
            .unwrap();
        assert_eq!(state.best_solution.values(), &prob.0.targets[..]);

        let mut state = SearchState::new_with_seed(&prob, 2);
        TabuSearch::<IntChangeNeighbor>::new(StopCondition::iterations(50), (1, 2))
            .run(&mut state)
            .unwrap();
        assert_eq!(state.best_solution.values(), &prob.0.targets[..]);
    }
}
