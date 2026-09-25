//! The moves every [`IntAssignment`] gets.

use super::assignment::IntAssignment;
use super::problem::with_value;
use crate::{
    common::TabuMemory,
    error::OptError,
    search_state::{EnabledTabu, Evaluable, Evaluate, MoveToNeighbor},
};
use rand::Rng;
use rand::rngs::SmallRng;

/// Every value but the current one of every variable, in index order and
/// ascending, as one flat walk over the changes.
struct Changes<'p, 's, P: IntAssignment> {
    prob: &'p P,
    sol: &'s P::Solution,
    var: usize,
    k: u64,
}

impl<P: IntAssignment> Iterator for Changes<'_, '_, P> {
    type Item = IntChangeNeighbor;

    #[inline]
    fn next(&mut self) -> Option<IntChangeNeighbor> {
        let vars = self.prob.domains();
        loop {
            let v = vars.get(self.var)?;
            if self.k < v.num_changes() {
                let var = self.var;
                let cur = P::get(self.sol, var);
                let value = v.lower().wrapping_add_unsigned(self.k);
                let value = if value >= cur { value + 1 } else { value };
                self.k += 1;
                return Some(IntChangeNeighbor {
                    var,
                    value,
                    gain: with_value(
                        self.sol.evaluate(),
                        self.prob.assign_delta(self.sol, var, value),
                    ),
                });
            }
            self.var += 1;
            self.k = 0;
        }
    }
}

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
    /// [`IntAssignment::assign_delta`]. Every construction site goes through
    /// here.
    pub fn new<P: IntAssignment>(prob: &P, sol: &P::Solution, var: usize, value: i64) -> Self {
        Self {
            var,
            value,
            gain: with_value(sol.evaluate(), prob.assign_delta(sol, var, value)),
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

impl<P: IntAssignment> MoveToNeighbor<P> for IntChangeNeighbor {
    /// Hands this move's [`EnabledTabu`] policy to the search state, which is
    /// what holds the tabu map.
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

    fn apply_to_solution(&self, prob: &P, sol: &mut P::Solution) -> Result<(), OptError> {
        prob.assign(sol, self.var, self.value);
        Ok(())
    }

    /// Empty on a [permutation](super::IntVars::permutation), where changing
    /// one value always repeats another.
    fn iter(prob: &P, sol: &P::Solution) -> impl Iterator<Item = Self> + Send {
        let vars = prob.domains();
        Changes {
            prob,
            sol,
            var: if vars.is_permutation() { vars.len() } else { 0 },
            k: 0,
        }
    }

    fn move_to_be_better_than(&self, _: &P, src: &P::Solution, other: &P::Solution) -> bool {
        self.evaluate()
            .improves_over(src.evaluate(), other.evaluate())
    }

    /// `O(log n)`. A variable is drawn with weight equal to its number of
    /// changes, then a value other than its current one uniformly, so every
    /// move [`iter`](MoveToNeighbor::iter) yields is equally likely.
    fn random_neighbor(prob: &P, sol: &P::Solution, rng: &mut SmallRng) -> Option<Self> {
        let vars = prob.domains();
        let total = vars.total_changes();
        if total == 0 || vars.is_permutation() {
            return None;
        }
        let var = vars.var_of_change(rng.random_range(0..total));
        let v = vars[var];
        let cur = P::get(sol, var);
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

/// Exchanges the values of variables `i` and `j`, which differ. On a
/// [permutation](super::IntVars::permutation) the result is again one.
#[derive(Debug, Clone, Copy)]
pub struct IntSwapNeighbor {
    /// The smaller of the two variables.
    pub i: usize,
    /// The larger of the two variables.
    pub j: usize,
    /// Change in objective after the move.
    pub gain: Evaluable<f64>,
}

impl IntSwapNeighbor {
    /// Builds the swap of `i` and `j`, its gain from
    /// [`IntAssignment::assign_swap_delta`].
    pub fn new<P: IntAssignment>(prob: &P, sol: &P::Solution, i: usize, j: usize) -> Self {
        Self {
            i,
            j,
            gain: with_value(sol.evaluate(), prob.assign_swap_delta(sol, i, j)),
        }
    }
}

impl Evaluate for IntSwapNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        self.gain
    }
}

impl EnabledTabu for IntSwapNeighbor {
    /// Keyed by the pair of variables.
    fn is_move_enabled(&self, tabu: &TabuMemory, iteration: u64) -> bool {
        tabu.is_enabled((self.i, self.j), iteration)
    }

    fn add_to_tabu_map(&self, tabu: &mut TabuMemory, iteration: u64, rng: &mut SmallRng) {
        tabu.forbid((self.i, self.j), iteration, rng);
    }
}

impl<P: IntAssignment> MoveToNeighbor<P> for IntSwapNeighbor {
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

    fn apply_to_solution(&self, prob: &P, sol: &mut P::Solution) -> Result<(), OptError> {
        prob.assign_swap(sol, self.i, self.j);
        Ok(())
    }

    /// Every pair `i < j` whose values differ.
    fn iter(prob: &P, sol: &P::Solution) -> impl Iterator<Item = Self> + Send {
        let n = prob.domains().len();
        (0..n).flat_map(move |i| {
            (i + 1..n)
                .filter(move |&j| P::get(sol, i) != P::get(sol, j))
                .map(move |j| Self::new(prob, sol, i, j))
        })
    }

    fn move_to_be_better_than(&self, _: &P, src: &P::Solution, other: &P::Solution) -> bool {
        self.evaluate()
            .improves_over(src.evaluate(), other.evaluate())
    }

    /// Rejection-samples a pair with different values, falling back to the
    /// full neighborhood after 64 misses.
    fn random_neighbor(prob: &P, sol: &P::Solution, rng: &mut SmallRng) -> Option<Self> {
        let n = prob.domains().len();
        if n < 2 {
            return None;
        }
        for _ in 0..64 {
            let (a, b) = (rng.random_range(0..n), rng.random_range(0..n));
            let (i, j) = (a.min(b), a.max(b));
            if i != j && P::get(sol, i) != P::get(sol, j) {
                return Some(Self::new(prob, sol, i, j));
            }
        }
        use rand::seq::IteratorRandom;
        Self::iter(prob, sol).choose(rng)
    }
}

/// Reverses the order of the values of variables `i..=j`, `i < j`. On a
/// [permutation](super::IntVars::permutation) read as a tour this is a 2-opt
/// move, and the result is again a permutation.
#[derive(Debug, Clone, Copy)]
pub struct IntReverseNeighbor {
    /// The first variable of the range.
    pub i: usize,
    /// The last variable of the range.
    pub j: usize,
    /// Change in objective after the move.
    pub gain: Evaluable<f64>,
}

impl IntReverseNeighbor {
    /// Builds the reversal of `i..=j`, its gain from
    /// [`IntAssignment::assign_reverse_delta`].
    pub fn new<P: IntAssignment>(prob: &P, sol: &P::Solution, i: usize, j: usize) -> Self {
        Self {
            i,
            j,
            gain: with_value(sol.evaluate(), prob.assign_reverse_delta(sol, i, j)),
        }
    }
}

impl Evaluate for IntReverseNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        self.gain
    }
}

impl EnabledTabu for IntReverseNeighbor {
    /// Keyed by the two ends of the reversed range.
    fn is_move_enabled(&self, tabu: &TabuMemory, iteration: u64) -> bool {
        tabu.is_enabled((self.i, self.j), iteration)
    }

    fn add_to_tabu_map(&self, tabu: &mut TabuMemory, iteration: u64, rng: &mut SmallRng) {
        tabu.forbid((self.i, self.j), iteration, rng);
    }
}

impl<P: IntAssignment> MoveToNeighbor<P> for IntReverseNeighbor {
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

    fn apply_to_solution(&self, prob: &P, sol: &mut P::Solution) -> Result<(), OptError> {
        prob.assign_reverse(sol, self.i, self.j);
        Ok(())
    }

    /// Every range `i..=j` with `i < j`.
    fn iter(prob: &P, sol: &P::Solution) -> impl Iterator<Item = Self> + Send {
        let n = prob.domains().len();
        (0..n).flat_map(move |i| (i + 1..n).map(move |j| Self::new(prob, sol, i, j)))
    }

    fn move_to_be_better_than(&self, _: &P, src: &P::Solution, other: &P::Solution) -> bool {
        self.evaluate()
            .improves_over(src.evaluate(), other.evaluate())
    }

    /// Rejection-samples two distinct ends, which succeeds with probability
    /// `1 - 1/n` a draw.
    fn random_neighbor(prob: &P, sol: &P::Solution, rng: &mut SmallRng) -> Option<Self> {
        let n = prob.domains().len();
        if n < 2 {
            return None;
        }
        loop {
            let (a, b) = (rng.random_range(0..n), rng.random_range(0..n));
            if a != b {
                return Some(Self::new(prob, sol, a.min(b), a.max(b)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::{Heuristic, LocalSearch, StopCondition, TabuSearch};
    use crate::problem::integer::problem::raw;
    use crate::problem::{IntAssignment, IntSolution, IntegerProblem};
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

    /// `Fast` written against `IntAssignment` with a solution of its own,
    /// which caches each variable's term.
    struct Own(Target);

    #[derive(Clone)]
    struct OwnSolution {
        x: Vec<i64>,
        terms: Vec<f64>,
        total: f64,
    }

    impl Evaluate for OwnSolution {
        fn evaluate(&self) -> Evaluable<f64> {
            Evaluable::Minimize(self.total)
        }
    }

    impl ProblemTrait for Own {
        type Solution = OwnSolution;
        fn new_solution(&self, rng: &mut impl rand::Rng) -> OwnSolution {
            let x: Vec<i64> = self
                .0
                .vars
                .iter()
                .map(|v| rng.random_range(v.lower()..=v.upper()))
                .collect();
            let terms: Vec<f64> = x
                .iter()
                .enumerate()
                .map(|(i, &v)| self.0.term(i, v))
                .collect();
            let total = terms.iter().sum();
            OwnSolution { x, terms, total }
        }
    }

    impl IntAssignment for Own {
        fn domains(&self) -> &IntVars {
            &self.0.vars
        }
        fn get(sol: &OwnSolution, i: usize) -> i64 {
            sol.x[i]
        }
        fn assign(&self, sol: &mut OwnSolution, i: usize, value: i64) {
            let t = self.0.term(i, value);
            sol.total += t - sol.terms[i];
            sol.terms[i] = t;
            sol.x[i] = value;
        }
        fn assign_delta(&self, sol: &OwnSolution, i: usize, value: i64) -> f64 {
            self.0.term(i, value) - sol.terms[i]
        }
    }

    #[test]
    fn own_solution_follows_the_integer_problem_trajectory() {
        let own = Own(Target::new());
        let kit = Fast(Target::new());
        let mut a = SearchState::new_with_seed(&own, 4);
        let init = kit.solution_from(a.solution.x.clone()).unwrap();
        let mut b = SearchState::with_solution_and_seed(&kit, init, 4);
        let ls = || LocalSearch::<IntChangeNeighbor>::new(StopCondition::iterations(50));
        ls().run(&mut a).unwrap();
        ls().run(&mut b).unwrap();
        assert_eq!(a.best_solution.x, b.best_solution.values());
        assert_eq!(a.best_solution.total, 0.0);
    }

    #[test]
    fn default_assign_delta_agrees_with_the_cached_one() {
        struct NoDelta(Own);
        impl ProblemTrait for NoDelta {
            type Solution = OwnSolution;
            fn new_solution(&self, rng: &mut impl rand::Rng) -> OwnSolution {
                self.0.new_solution(rng)
            }
        }
        impl IntAssignment for NoDelta {
            fn domains(&self) -> &IntVars {
                self.0.domains()
            }
            fn get(sol: &OwnSolution, i: usize) -> i64 {
                sol.x[i]
            }
            fn assign(&self, sol: &mut OwnSolution, i: usize, value: i64) {
                self.0.assign(sol, i, value)
            }
        }
        let prob = NoDelta(Own(Target::new()));
        let sol = prob.new_solution(&mut rng(8));
        for m in IntChangeNeighbor::iter(&prob, &sol) {
            assert_eq!(raw(m.gain), prob.0.assign_delta(&sol, m.var, m.value));
        }
    }

    /// A tour over eight points on a grid, positions as variables. `fast`
    /// overrides the swap and reverse deltas with the local edge change.
    struct Tour {
        vars: IntVars,
        pts: Vec<(i64, i64)>,
        fast: bool,
    }

    impl Tour {
        fn new(fast: bool) -> Self {
            let pts = vec![
                (0, 0),
                (3, 1),
                (5, 4),
                (2, 6),
                (7, 7),
                (1, 3),
                (6, 2),
                (4, 5),
            ];
            Self {
                vars: IntVars::permutation(pts.len()),
                pts,
                fast,
            }
        }
        fn d(&self, a: i64, b: i64) -> f64 {
            let (p, q) = (self.pts[a as usize], self.pts[b as usize]);
            ((p.0 - q.0).abs() + (p.1 - q.1).abs()) as f64
        }
        /// The edge leaving position `p`.
        fn edge(&self, v: &[i64], p: usize) -> f64 {
            self.d(v[p], v[(p + 1) % v.len()])
        }
    }

    impl IntegerProblem for Tour {
        fn variables(&self) -> &IntVars {
            &self.vars
        }
        fn objective(&self, v: &[i64]) -> Evaluable<f64> {
            Evaluable::Minimize((0..v.len()).map(|p| self.edge(v, p)).sum())
        }
        fn swap_delta(&self, sol: &IntSolution, i: usize, j: usize) -> f64 {
            if !self.fast {
                let mut v = sol.values().to_vec();
                v.swap(i, j);
                return raw(self.objective(&v)) - raw(sol.evaluate());
            }
            let n = sol.values().len();
            let mut edges = vec![(i + n - 1) % n, i, (j + n - 1) % n, j];
            edges.sort_unstable();
            edges.dedup();
            let mut v = sol.values().to_vec();
            let before: f64 = edges.iter().map(|&p| self.edge(&v, p)).sum();
            v.swap(i, j);
            let after: f64 = edges.iter().map(|&p| self.edge(&v, p)).sum();
            after - before
        }
        fn reverse_delta(&self, sol: &IntSolution, i: usize, j: usize) -> f64 {
            let v = sol.values();
            let n = v.len();
            if !self.fast {
                let mut w = v.to_vec();
                w[i..=j].reverse();
                return raw(self.objective(&w)) - raw(sol.evaluate());
            }
            if j - i + 1 >= n - 1 {
                return 0.0;
            }
            let (prev, next) = (v[(i + n - 1) % n], v[(j + 1) % n]);
            self.d(prev, v[j]) + self.d(v[i], next) - self.d(prev, v[i]) - self.d(v[j], next)
        }
    }

    fn is_permutation(v: &[i64]) -> bool {
        let mut w = v.to_vec();
        w.sort_unstable();
        w.iter().enumerate().all(|(k, &x)| x == k as i64)
    }

    #[test]
    fn permutation_moves_keep_a_permutation_and_its_objective() {
        let prob = Tour::new(true);
        let mut sol = prob.new_solution(&mut rng(30));
        assert!(is_permutation(sol.values()));
        let mut r = rng(31);
        for step in 0..400 {
            if step % 2 == 0 {
                let m = IntSwapNeighbor::random_neighbor(&prob, &sol, &mut r).unwrap();
                m.apply_to_solution(&prob, &mut sol).unwrap();
            } else {
                let m = IntReverseNeighbor::random_neighbor(&prob, &sol, &mut r).unwrap();
                m.apply_to_solution(&prob, &mut sol).unwrap();
            }
            assert!(is_permutation(sol.values()));
            assert_eq!(raw(sol.evaluate()), raw(prob.objective(sol.values())));
        }
    }

    #[test]
    fn local_swap_and_reverse_deltas_agree_with_the_defaults() {
        let (fast, slow) = (Tour::new(true), Tour::new(false));
        let sol = fast.new_solution(&mut rng(32));
        for (a, b) in IntSwapNeighbor::iter(&fast, &sol).zip(IntSwapNeighbor::iter(&slow, &sol)) {
            assert_eq!(raw(a.gain), raw(b.gain), "swap {} {}", a.i, a.j);
        }
        for (a, b) in
            IntReverseNeighbor::iter(&fast, &sol).zip(IntReverseNeighbor::iter(&slow, &sol))
        {
            assert_eq!(raw(a.gain), raw(b.gain), "reverse {} {}", a.i, a.j);
        }
    }

    #[test]
    fn a_permutation_has_no_single_value_change() {
        let prob = Tour::new(true);
        let sol = prob.new_solution(&mut rng(33));
        assert_eq!(IntChangeNeighbor::iter(&prob, &sol).count(), 0);
        assert!(IntChangeNeighbor::random_neighbor(&prob, &sol, &mut rng(34)).is_none());
        assert!(prob.solution_from(vec![0, 1, 2, 3, 4, 5, 6, 6]).is_err());
    }

    #[test]
    fn two_opt_local_search_uncrosses_the_tour() {
        let prob = Tour::new(true);
        let mut state = SearchState::new_with_seed(&prob, 5);
        let start = raw(state.solution.evaluate());
        LocalSearch::<IntReverseNeighbor>::new(StopCondition::iterations(100))
            .run(&mut state)
            .unwrap();
        assert!(raw(state.best_solution.evaluate()) < start);
        assert!(is_permutation(state.best_solution.values()));
    }
}
