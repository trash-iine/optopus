//! The moves every [`IntAssignment`] gets.

use super::assignment::IntAssignment;
use super::problem::IntVar;
use super::problem::with_value;
use crate::{
    common::{TabuMemory, permutation::random_distinct_pair},
    error::OptError,
    search_state::{EnabledTabu, Evaluable, Evaluate, MoveToNeighbor},
};
use rand::Rng;
use rand::rngs::SmallRng;

/// Every value but the current one of every variable, in index order and
/// ascending, as one flat walk over the changes. The walk's position is the
/// change's slot, which is how [`IntAssignment::slot_delta`] finds it.
struct Changes<'p, 's, P: IntAssignment> {
    prob: &'p P,
    sol: &'s P::Solution,
    /// The slot of the next change.
    slot: usize,
    /// `-1.0` on a maximized objective and `1.0` on a minimized one, read
    /// once so that no candidate branches on the direction.
    sign: f64,
    walk: Walk,
}

/// How the walk finds the change at a slot.
enum Walk {
    /// Every variable ranges over the same two values, whose sum this is, so
    /// slot `v` is variable `v` going to the other value. `end` is the number
    /// of variables.
    Flip { sum: i64, end: usize },
    /// Any ranges. The variable being walked, its range and its current value,
    /// read once when the walk enters it, and the slots its changes occupy.
    Rows {
        next_var: usize,
        var: usize,
        range: IntVar,
        current: i64,
        row_start: usize,
        row_end: usize,
    },
}

/// The factor that turns a change in the raw objective of `sol` into a cost,
/// lower being better.
#[inline]
fn direction<S: Evaluate>(sol: &S) -> f64 {
    with_value(sol.evaluate(), 1.0).minimized()
}

impl<'p, 's, P: IntAssignment> Changes<'p, 's, P> {
    fn new(prob: &'p P, sol: &'s P::Solution) -> Self {
        let vars = prob.domains();
        // A permutation has no single change, so its walk is empty.
        let first = if vars.is_permutation() { vars.len() } else { 0 };
        let walk = match vars.flip_sum() {
            Some(sum) if first == 0 => Walk::Flip {
                sum,
                end: vars.len(),
            },
            _ => Walk::Rows {
                next_var: first,
                var: 0,
                range: IntVar::binary(),
                current: 0,
                row_start: 0,
                row_end: 0,
            },
        };
        Self {
            prob,
            sol,
            slot: 0,
            sign: direction(sol),
            walk,
        }
    }
}

impl<P: IntAssignment> Iterator for Changes<'_, '_, P> {
    type Item = IntChangeNeighbor;

    #[inline]
    fn next(&mut self) -> Option<IntChangeNeighbor> {
        let slot = self.slot;
        let (var, value) = match &mut self.walk {
            Walk::Flip { sum, end } => {
                if slot == *end {
                    return None;
                }
                (slot, *sum - P::get(self.sol, slot))
            }
            Walk::Rows {
                next_var,
                var,
                range,
                current,
                row_start,
                row_end,
            } => {
                while slot == *row_end {
                    // On to the next variable with a change, past any fixed one.
                    let v = *self.prob.domains().get(*next_var)?;
                    *var = *next_var;
                    *next_var += 1;
                    *row_start = *row_end;
                    *row_end += v.num_changes() as usize;
                    *range = v;
                    *current = P::get(self.sol, *var);
                }
                (*var, range.nth_other((slot - *row_start) as u64, *current))
            }
        };
        self.slot += 1;
        Some(IntChangeNeighbor {
            var,
            value,
            cost: self.sign * self.prob.slot_delta(self.sol, slot, var, value),
        })
    }
}

/// Sets variable `var` to `value`, any value in its range other than the
/// current one. On a binary variable this is a flip.
///
/// `cost` is the change the move makes to the objective with its direction
/// applied, so lower is better. It is what [`Evaluable::minimized`] reports
/// for the move.
#[derive(Debug, Clone, Copy)]
pub struct IntChangeNeighbor {
    /// The variable to change.
    pub var: usize,
    /// The value it takes.
    pub value: i64,
    /// The change the move makes, with the direction applied, lower is better.
    pub cost: f64,
}

impl IntChangeNeighbor {
    /// Builds the move setting `var` to `value`, its cost from
    /// [`IntAssignment::assign_delta`].
    pub fn new<P: IntAssignment>(prob: &P, sol: &P::Solution, var: usize, value: i64) -> Self {
        Self {
            var,
            value,
            cost: direction(sol) * prob.assign_delta(sol, var, value),
        }
    }
}

impl Evaluate for IntChangeNeighbor {
    /// The cost, which is a change of a minimized quantity whatever the
    /// problem's direction.
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.cost)
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
        Changes::new(prob, sol)
    }

    fn move_to_be_better_than(&self, _: &P, src: &P::Solution, other: &P::Solution) -> bool {
        self.evaluate()
            .improves_over(src.evaluate(), other.evaluate())
    }

    /// One draw of a slot, uniform over every move
    /// [`iter`](MoveToNeighbor::iter) yields. Finding its variable is `O(1)`
    /// when every variable has the same range and `O(log n)` otherwise.
    fn random_neighbor(prob: &P, sol: &P::Solution, rng: &mut SmallRng) -> Option<Self> {
        let vars = prob.domains();
        let total = vars.total_changes();
        if total == 0 || vars.is_permutation() {
            return None;
        }
        let slot = rng.random_range(0..total);
        let var = vars.var_of_change(slot);
        let k = slot - vars.row_start(var) as u64;
        let value = vars[var].nth_other(k, P::get(sol, var));
        Some(Self {
            var,
            value,
            cost: direction(sol) * prob.slot_delta(sol, slot as usize, var, value),
        })
    }
}

/// Exchanges the values of variables `i` and `j`. Only pairs whose values
/// differ and fit each other's range are moves. On a
/// [permutation](super::IntVars::permutation) the result is again one.
#[derive(Debug, Clone, Copy)]
pub struct IntSwapNeighbor {
    /// The smaller of the two variables.
    pub i: usize,
    /// The larger of the two variables.
    pub j: usize,
    /// The change the move makes, with the direction applied, lower is better.
    pub cost: f64,
}

impl IntSwapNeighbor {
    /// Builds the swap of `i` and `j`, its cost from
    /// [`IntAssignment::assign_swap_delta`].
    pub fn new<P: IntAssignment>(prob: &P, sol: &P::Solution, i: usize, j: usize) -> Self {
        Self {
            i,
            j,
            cost: direction(sol) * prob.assign_swap_delta(sol, i, j),
        }
    }
}

impl Evaluate for IntSwapNeighbor {
    /// The cost, which is a change of a minimized quantity whatever the
    /// problem's direction.
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.cost)
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

/// A pair `i < j` drawn uniformly among those `ok` accepts, `None` when there
/// is none. Up to 64 draws of two distinct positions are tried, and after
/// that one move is chosen from `all`, the whole neighborhood, so a sparse
/// neighborhood is still sampled uniformly.
fn sample_pair<M>(
    n: usize,
    rng: &mut SmallRng,
    ok: impl Fn(usize, usize) -> bool,
    build: impl Fn(usize, usize) -> M,
    all: impl Iterator<Item = M>,
) -> Option<M> {
    for _ in 0..64 {
        let (a, b) = random_distinct_pair(n, rng)?;
        let (i, j) = (a.min(b), a.max(b));
        if ok(i, j) {
            return Some(build(i, j));
        }
    }
    use rand::seq::IteratorRandom;
    all.choose(rng)
}

/// Whether exchanging `i` and `j` changes the solution and keeps both values
/// in range.
#[inline]
fn swappable<P: IntAssignment>(prob: &P, sol: &P::Solution, i: usize, j: usize) -> bool {
    let (a, b) = (P::get(sol, i), P::get(sol, j));
    let vars = prob.domains();
    a != b && vars[i].contains(b) && vars[j].contains(a)
}

impl<P: IntAssignment> MoveToNeighbor<P> for IntSwapNeighbor {
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

    fn apply_to_solution(&self, prob: &P, sol: &mut P::Solution) -> Result<(), OptError> {
        prob.assign_swap(sol, self.i, self.j);
        Ok(())
    }

    /// Every pair `i < j` whose values differ and fit each other's range.
    fn iter(prob: &P, sol: &P::Solution) -> impl Iterator<Item = Self> + Send {
        let n = prob.domains().len();
        (0..n).flat_map(move |i| {
            (i + 1..n)
                .filter(move |&j| swappable(prob, sol, i, j))
                .map(move |j| Self::new(prob, sol, i, j))
        })
    }

    fn move_to_be_better_than(&self, _: &P, src: &P::Solution, other: &P::Solution) -> bool {
        self.evaluate()
            .improves_over(src.evaluate(), other.evaluate())
    }

    /// Draws two distinct positions until they form a move, up to 64 times,
    /// then picks uniformly from the whole neighborhood.
    fn random_neighbor(prob: &P, sol: &P::Solution, rng: &mut SmallRng) -> Option<Self> {
        sample_pair(
            prob.domains().len(),
            rng,
            |i, j| swappable(prob, sol, i, j),
            |i, j| Self::new(prob, sol, i, j),
            Self::iter(prob, sol),
        )
    }
}

/// Reverses the order of the values of variables `i..=j`, `i < j`, when each
/// value fits its new variable's range. On a
/// [permutation](super::IntVars::permutation) read as a tour this is a 2-opt
/// move, and the result is again a permutation.
#[derive(Debug, Clone, Copy)]
pub struct IntReverseNeighbor {
    /// The first variable of the range.
    pub i: usize,
    /// The last variable of the range.
    pub j: usize,
    /// The change the move makes, with the direction applied, lower is better.
    pub cost: f64,
}

impl IntReverseNeighbor {
    /// Builds the reversal of `i..=j`, its cost from
    /// [`IntAssignment::assign_reverse_delta`].
    pub fn new<P: IntAssignment>(prob: &P, sol: &P::Solution, i: usize, j: usize) -> Self {
        Self {
            i,
            j,
            cost: direction(sol) * prob.assign_reverse_delta(sol, i, j),
        }
    }
}

impl Evaluate for IntReverseNeighbor {
    /// The cost, which is a change of a minimized quantity whatever the
    /// problem's direction.
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.cost)
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

/// Whether reversing `i..=j` keeps every value in its new variable's range.
/// Free when every variable has the same range, as on a permutation.
#[inline]
fn reversible<P: IntAssignment>(prob: &P, sol: &P::Solution, i: usize, j: usize) -> bool {
    let vars = prob.domains();
    vars.same_range() || (i..=j).all(|p| vars[p].contains(P::get(sol, i + j - p)))
}

impl<P: IntAssignment> MoveToNeighbor<P> for IntReverseNeighbor {
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

    fn apply_to_solution(&self, prob: &P, sol: &mut P::Solution) -> Result<(), OptError> {
        prob.assign_reverse(sol, self.i, self.j);
        Ok(())
    }

    /// Every range `i..=j` with `i < j` whose values fit their new variables
    /// once reversed.
    fn iter(prob: &P, sol: &P::Solution) -> impl Iterator<Item = Self> + Send {
        let n = prob.domains().len();
        (0..n).flat_map(move |i| {
            (i + 1..n)
                .filter(move |&j| reversible(prob, sol, i, j))
                .map(move |j| Self::new(prob, sol, i, j))
        })
    }

    fn move_to_be_better_than(&self, _: &P, src: &P::Solution, other: &P::Solution) -> bool {
        self.evaluate()
            .improves_over(src.evaluate(), other.evaluate())
    }

    /// Draws two distinct ends until they form a move, up to 64 times, then
    /// picks uniformly from the whole neighborhood. When every variable has
    /// the same range the first draw always is one.
    fn random_neighbor(prob: &P, sol: &P::Solution, rng: &mut SmallRng) -> Option<Self> {
        sample_pair(
            prob.domains().len(),
            rng,
            |i, j| reversible(prob, sol, i, j),
            |i, j| Self::new(prob, sol, i, j),
            Self::iter(prob, sol),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::{Heuristic, LocalSearch, StopCondition, TabuSearch};
    use crate::problem::integer::problem::raw;
    use crate::problem::{IntAssignment, IntSolution, IntVar, IntVars, IntegerProblem};
    use crate::search_state::{ProblemTrait, SearchState};
    use rand::SeedableRng;

    /// Minimizes the sum of (x_i - t_i)^2, with ranges of different widths and
    /// one fixed variable.
    const TARGETS: [i64; 5] = [4, -2, 7, 1, 13];

    fn target_vars() -> IntVars {
        [(0, 5), (-3, 3), (7, 7), (0, 1), (-10, 20)]
            .into_iter()
            .map(|(l, u)| IntVar::new(l, u))
            .collect()
    }

    fn term(i: usize, x: i64) -> f64 {
        ((x - TARGETS[i]) * (x - TARGETS[i])) as f64
    }

    fn target_sum(x: &[i64]) -> f64 {
        x.iter().enumerate().map(|(i, &v)| term(i, v)).sum()
    }

    /// The objective and delta types of the test problems, named so that the
    /// helpers below can return them.
    type Objective = fn(&[i64]) -> f64;
    type ChangeFn = fn(&IntSolution, usize, i64) -> f64;
    type PairFn = fn(&IntSolution, usize, usize) -> f64;

    /// The target problem pricing every change by the whole objective.
    fn slow() -> IntegerProblem<Objective> {
        IntegerProblem::minimize(target_vars(), target_sum)
    }

    fn target_delta(sol: &IntSolution, i: usize, value: i64) -> f64 {
        term(i, value) - term(i, sol.value(i))
    }

    /// The same problem with an incremental delta.
    fn fast() -> IntegerProblem<Objective, ChangeFn> {
        IntegerProblem::minimize(target_vars(), target_sum as Objective)
            .with_delta(target_delta as ChangeFn)
    }

    fn rng(seed: u64) -> SmallRng {
        SmallRng::seed_from_u64(seed)
    }

    #[test]
    fn new_solution_is_in_range_and_caches_its_objective() {
        let prob = fast();
        for seed in 0..20 {
            let sol = prob.new_solution(&mut rng(seed));
            for (v, &x) in prob.variables().iter().zip(sol.values()) {
                assert!(v.contains(x));
            }
            assert_eq!(raw(sol.evaluate()), raw(prob.objective(sol.values())));
        }
    }

    #[test]
    fn the_direction_is_the_constructor_s() {
        let min = IntegerProblem::minimize(target_vars(), target_sum);
        let max = IntegerProblem::maximize(target_vars(), target_sum);
        let x = [0, 0, 7, 0, 0];
        assert!(matches!(min.objective(&x), Evaluable::Minimize(_)));
        assert!(matches!(max.objective(&x), Evaluable::Maximize(_)));
    }

    #[test]
    fn iter_yields_every_other_value_of_every_variable() {
        let prob = fast();
        let sol = prob.new_solution(&mut rng(3));
        let moves: Vec<_> = IntChangeNeighbor::iter(&prob, &sol).collect();
        assert_eq!(moves.len() as u64, prob.variables().total_changes());
        assert_eq!(prob.variables().total_changes(), 42); // 5 + 6 + 0 + 1 + 30
        assert!(moves.iter().all(|m| m.value != sol.value(m.var)));
    }

    #[test]
    fn applying_a_move_keeps_the_cached_objective_exact() {
        let prob = fast();
        let mut sol = prob.new_solution(&mut rng(5));
        let mut r = rng(6);
        for _ in 0..200 {
            let m = IntChangeNeighbor::random_neighbor(&prob, &sol, &mut r).unwrap();
            m.apply_to_solution(&prob, &mut sol).unwrap();
            assert_eq!(raw(sol.evaluate()), raw(prob.objective(sol.values())));
        }
    }

    #[test]
    fn the_whole_objective_fallback_agrees_with_the_given_delta() {
        let (fast, slow) = (fast(), slow());
        let sol = fast.new_solution(&mut rng(9));
        for (a, b) in IntChangeNeighbor::iter(&fast, &sol).zip(IntChangeNeighbor::iter(&slow, &sol))
        {
            assert_eq!(a.cost, b.cost);
        }
    }

    #[test]
    fn random_neighbor_is_uniform_over_iter() {
        let prob = fast();
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
        let vars = [IntVar::new(2, 2), IntVar::new(-1, -1)]
            .into_iter()
            .collect();
        let prob = IntegerProblem::maximize(vars, |_: &[i64]| 0.0);
        let sol = prob.new_solution(&mut rng(0));
        assert!(IntChangeNeighbor::random_neighbor(&prob, &sol, &mut rng(1)).is_none());
        assert_eq!(IntChangeNeighbor::iter(&prob, &sol).count(), 0);
    }

    #[test]
    fn solution_from_rejects_bad_values() {
        let prob = fast();
        assert!(prob.solution_from(vec![0; 4]).is_err());
        assert!(prob.solution_from(vec![0, 0, 6, 0, 0]).is_err());
        let sol = prob.solution_from(TARGETS.to_vec()).unwrap();
        assert_eq!(raw(sol.evaluate()), 0.0);
    }

    #[test]
    fn local_search_and_tabu_search_reach_the_optimum() {
        let prob = fast();
        let mut state = SearchState::new_with_seed(&prob, 1);
        LocalSearch::<IntChangeNeighbor>::new(StopCondition::iterations(50))
            .run(&mut state)
            .unwrap();
        assert_eq!(state.best_solution.values(), &TARGETS[..]);

        let mut state = SearchState::new_with_seed(&prob, 2);
        TabuSearch::<IntChangeNeighbor>::new(StopCondition::iterations(50), (1, 2))
            .run(&mut state)
            .unwrap();
        assert_eq!(state.best_solution.values(), &TARGETS[..]);
    }

    /// The target problem written against `IntAssignment` with a solution of
    /// its own, which caches each variable's term.
    struct Own(IntVars);

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
            let x = self.0.random_values(rng);
            let terms: Vec<f64> = x.iter().enumerate().map(|(i, &v)| term(i, v)).collect();
            let total = terms.iter().sum();
            OwnSolution { x, terms, total }
        }
    }

    impl IntAssignment for Own {
        fn domains(&self) -> &IntVars {
            &self.0
        }
        fn get(sol: &OwnSolution, i: usize) -> i64 {
            sol.x[i]
        }
        fn assign(&self, sol: &mut OwnSolution, i: usize, value: i64) {
            let t = term(i, value);
            sol.total += t - sol.terms[i];
            sol.terms[i] = t;
            sol.x[i] = value;
        }
        fn assign_delta(&self, sol: &OwnSolution, i: usize, value: i64) -> f64 {
            term(i, value) - sol.terms[i]
        }
    }

    #[test]
    fn own_solution_follows_the_integer_problem_trajectory() {
        let own = Own(target_vars());
        let kit = fast();
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
        struct DefaultDelta(Own);
        impl ProblemTrait for DefaultDelta {
            type Solution = OwnSolution;
            fn new_solution(&self, rng: &mut impl rand::Rng) -> OwnSolution {
                self.0.new_solution(rng)
            }
        }
        impl IntAssignment for DefaultDelta {
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
        let prob = DefaultDelta(Own(target_vars()));
        let sol = prob.new_solution(&mut rng(8));
        for m in IntChangeNeighbor::iter(&prob, &sol) {
            assert_eq!(m.cost, prob.0.assign_delta(&sol, m.var, m.value));
        }
    }

    /// A tour over eight points on a grid, positions as variables.
    const PTS: [(i64, i64); 8] = [
        (0, 0),
        (3, 1),
        (5, 4),
        (2, 6),
        (7, 7),
        (1, 3),
        (6, 2),
        (4, 5),
    ];

    fn d(a: i64, b: i64) -> f64 {
        let (p, q) = (PTS[a as usize], PTS[b as usize]);
        ((p.0 - q.0).abs() + (p.1 - q.1).abs()) as f64
    }

    /// The edge leaving position `p`.
    fn edge(v: &[i64], p: usize) -> f64 {
        d(v[p], v[(p + 1) % v.len()])
    }

    fn tour_len(v: &[i64]) -> f64 {
        (0..v.len()).map(|p| edge(v, p)).sum()
    }

    fn swap_local(sol: &IntSolution, i: usize, j: usize) -> f64 {
        let n = sol.values().len();
        let mut edges = vec![(i + n - 1) % n, i, (j + n - 1) % n, j];
        edges.sort_unstable();
        edges.dedup();
        let mut v = sol.values().to_vec();
        let before: f64 = edges.iter().map(|&p| edge(&v, p)).sum();
        v.swap(i, j);
        let after: f64 = edges.iter().map(|&p| edge(&v, p)).sum();
        after - before
    }

    fn reverse_local(sol: &IntSolution, i: usize, j: usize) -> f64 {
        let v = sol.values();
        let n = v.len();
        if j - i + 1 >= n - 1 {
            return 0.0;
        }
        let (prev, next) = (v[(i + n - 1) % n], v[(j + 1) % n]);
        d(prev, v[j]) + d(v[i], next) - d(prev, v[i]) - d(v[j], next)
    }

    fn tour() -> IntegerProblem<Objective, crate::problem::integer::NoDelta, PairFn, PairFn> {
        IntegerProblem::minimize(IntVars::permutation(PTS.len()), tour_len as Objective)
            .with_swap_delta(swap_local as PairFn)
            .with_reverse_delta(reverse_local as PairFn)
    }

    fn is_permutation(v: &[i64]) -> bool {
        let mut w = v.to_vec();
        w.sort_unstable();
        w.iter().enumerate().all(|(k, &x)| x == k as i64)
    }

    #[test]
    fn permutation_moves_keep_a_permutation_and_its_objective() {
        let prob = tour();
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
    fn local_swap_and_reverse_deltas_agree_with_the_whole_objective() {
        let fast = tour();
        let slow = IntegerProblem::minimize(IntVars::permutation(PTS.len()), tour_len);
        let sol = fast.new_solution(&mut rng(32));
        for (a, b) in IntSwapNeighbor::iter(&fast, &sol).zip(IntSwapNeighbor::iter(&slow, &sol)) {
            assert_eq!(a.cost, b.cost, "swap {} {}", a.i, a.j);
        }
        for (a, b) in
            IntReverseNeighbor::iter(&fast, &sol).zip(IntReverseNeighbor::iter(&slow, &sol))
        {
            assert_eq!(a.cost, b.cost, "reverse {} {}", a.i, a.j);
        }
    }

    #[test]
    fn swaps_and_reversals_stay_inside_each_variable_s_range() {
        // [0, 1] then [5, 6], so no value fits the other variable.
        let vars = [IntVar::new(0, 1), IntVar::new(5, 6), IntVar::new(0, 6)]
            .into_iter()
            .collect();
        let prob = IntegerProblem::minimize(vars, |x: &[i64]| x.iter().sum::<i64>() as f64);
        let sol = prob.solution_from(vec![0, 5, 1]).unwrap();
        let swaps: Vec<_> = IntSwapNeighbor::iter(&prob, &sol)
            .map(|m| (m.i, m.j))
            .collect();
        assert_eq!(swaps, vec![(0, 2)]);
        let reversals: Vec<_> = IntReverseNeighbor::iter(&prob, &sol)
            .map(|m| (m.i, m.j))
            .collect();
        assert_eq!(reversals, vec![(0, 2)]);
    }

    #[test]
    fn a_permutation_has_no_single_value_change() {
        let prob = tour();
        let sol = prob.new_solution(&mut rng(33));
        assert_eq!(IntChangeNeighbor::iter(&prob, &sol).count(), 0);
        assert!(IntChangeNeighbor::random_neighbor(&prob, &sol, &mut rng(34)).is_none());
        assert!(prob.solution_from(vec![0, 1, 2, 3, 4, 5, 6, 6]).is_err());
    }

    #[test]
    fn two_opt_local_search_uncrosses_the_tour() {
        let prob = tour();
        let mut state = SearchState::new_with_seed(&prob, 5);
        let start = raw(state.solution.evaluate());
        LocalSearch::<IntReverseNeighbor>::new(StopCondition::iterations(100))
            .run(&mut state)
            .unwrap();
        assert!(raw(state.best_solution.evaluate()) < start);
        assert!(is_permutation(state.best_solution.values()));
    }
}
