//! Clause prices, the [`WeightedConstraints`] side of [`Sat`].

use super::{Sat, SatFlipNeighbor, SatSolution};
use crate::building_blocks::representation::uniform_binary_crossover;
use crate::building_blocks::search::IndexSet;
use crate::error::OptError;
use crate::trait_defs::{ProblemTrait, WeightedConstraints};
use rand::rngs::SmallRng;

/// What a [`Sat`] keeps to price a flip under clause weights, the weighted
/// flip gains.
///
/// The constraints are the clauses, and they are the whole objective, so the
/// objective apart from them is zero.
#[derive(Debug, Clone)]
pub struct SatPrices {
    /// `gain[i]` is the weight flipping variable `i` satisfies minus the
    /// weight it leaves unsatisfied.
    gain: Vec<i64>,
}

impl WeightedConstraints for Sat {
    type Move = SatFlipNeighbor;
    type Prices = SatPrices;

    /// An empty clause is left out of the unsatisfied, since no flip can
    /// satisfy it.
    fn unit_prices(&self, sol: &SatSolution) -> (SatPrices, IndexSet) {
        let mut unsatisfied = IndexSet::new(self.n_clauses());
        for (c, clause) in self.all_clauses().enumerate() {
            if !clause.is_empty() && !Sat::is_clause_satisfied(clause, &sol.x) {
                unsatisfied.insert(c);
            }
        }
        // At weight one the weighted gain is the count the solution keeps.
        let prices = SatPrices {
            gain: sol.gain.clone(),
        };
        (prices, unsatisfied)
    }

    fn weight_gain(&self, _: &SatSolution, prices: &SatPrices, mv: &SatFlipNeighbor) -> i64 {
        prices.gain[mv.i]
    }

    fn apply_weighted(
        &self,
        sol: &mut SatSolution,
        mv: &SatFlipNeighbor,
        weight: &[u32],
        prices: &mut SatPrices,
        unsatisfied: &mut IndexSet,
    ) -> Result<(), OptError> {
        // The clauses holding the variable leave both gains, the solution's
        // and the weighted, under the old value and come back under the new
        // one, in one walk of each clause per side.
        let i = mv.i;
        let (gain, weighted) = (&mut sol.gain, &mut prices.gain);
        for &c in self.clauses_of(i) {
            let w = weight[c] as i64;
            self.clause_shares(c, &sol.x, |v, share| {
                gain[v] -= share;
                weighted[v] -= w * share;
            });
        }
        sol.x[i] = !sol.x[i];
        for &c in self.clauses_of(i) {
            let w = weight[c] as i64;
            let satisfied = self.clause_shares(c, &sol.x, |v, share| {
                gain[v] += share;
                weighted[v] += w * share;
            });
            if satisfied {
                unsatisfied.remove(c);
            } else {
                unsatisfied.insert(c);
            }
        }
        sol.n_satisfied = (sol.n_satisfied as i64 + mv.gain) as usize;
        Ok(())
    }

    #[inline]
    fn reweight(&self, sol: &SatSolution, prices: &mut SatPrices, c: usize, delta: i64) {
        self.add_clause_shares(c, &sol.x, &mut prices.gain, delta);
    }

    /// The flips of the variables of an unsatisfied clause, every literal of
    /// which is false, so each of them satisfies it. In a satisfied clause,
    /// the flips of the variables with a true literal that leave another
    /// literal true. A variable the clause holds twice is one flip.
    ///
    /// An assignment that satisfies every clause is optimal and has no escape,
    /// which the default of
    /// [`objective_escapes`](WeightedConstraints::objective_escapes) gives.
    fn tight_moves(
        &self,
        sol: &SatSolution,
        _: &SatPrices,
        c: usize,
        out: &mut Vec<SatFlipNeighbor>,
    ) {
        let clause = self.clause(c);
        let var = |lit: i64| lit.unsigned_abs() as usize - 1;
        let sat = |lit: i64| sol.x[var(lit)] == (lit > 0);
        let true_now = clause.iter().filter(|&&lit| sat(lit)).count();
        for (p, &lit) in clause.iter().enumerate() {
            let v = var(lit);
            if clause[..p].iter().any(|&l| var(l) == v) {
                continue;
            }
            let (mut s, mut u) = (0, 0);
            for &l in clause.iter().filter(|&&l| var(l) == v) {
                if sat(l) {
                    s += 1;
                } else {
                    u += 1;
                }
            }
            if true_now == 0 || (s > 0 && true_now - s + u > 0) {
                out.push(SatFlipNeighbor::new(self, sol, v));
            }
        }
    }

    /// Each variable as in `best` or as in a fresh assignment, with even odds.
    fn restart_from(&self, best: &SatSolution, rng: &mut SmallRng) -> SatSolution {
        let fresh = self.new_solution(rng);
        uniform_binary_crossover(self, best, &fresh, rng).expect("a uniform crossover cannot fail")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::building_blocks::search::ConstraintWeights;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    /// A formula with a repeated literal and a clause holding a variable and
    /// its negation, the two shapes the shares have to count whole.
    fn formula() -> Sat {
        let mut sat = Sat::new(5);
        for clause in [
            vec![1, 2, -3],
            vec![-1, 4],
            vec![2, 2, 5],
            vec![3, -3, 1],
            vec![-2, -4, -5],
            vec![-1, -2],
            vec![3, 4, 5],
            vec![-5],
        ] {
            sat.add_clause(clause);
        }
        sat
    }

    type Weights = ConstraintWeights<Sat>;

    /// The weight of the unsatisfied clauses, from scratch.
    fn cost(sat: &Sat, x: &[bool], w: &Weights) -> i64 {
        sat.all_clauses()
            .enumerate()
            .filter(|(_, c)| !Sat::is_clause_satisfied(c, x))
            .map(|(c, _)| w.weight(c) as i64)
            .sum()
    }

    /// Every cached gain is the drop in the from-scratch cost, and the
    /// unsatisfied set is the clauses the assignment fails.
    fn assert_priced(sat: &Sat, sol: &SatSolution, w: &Weights) {
        let now = cost(sat, &sol.x, w);
        for i in 0..sat.n_vars() {
            let mut x = sol.x.clone();
            x[i] = !x[i];
            assert_eq!(w.prices().gain[i], now - cost(sat, &x, w), "gain of {i}");
        }
        let mut expected: Vec<usize> = (0..sat.n_clauses())
            .filter(|&c| !Sat::is_clause_satisfied(sat.clause(c), &sol.x))
            .collect();
        let mut got = w.violated().as_slice().to_vec();
        expected.sort_unstable();
        got.sort_unstable();
        assert_eq!(got, expected);
    }

    #[test]
    fn prices_follow_flips_and_weight_changes() {
        let sat = formula();
        let mut rng = SmallRng::seed_from_u64(11);
        let mut sol = sat.new_solution(&mut rng);
        let mut w = Weights::new(&sat, &sol, 4);
        assert_eq!(
            w.cap(),
            8,
            "the cap is the clause count when that is larger"
        );
        assert_priced(&sat, &sol, &w);
        for step in 0..300 {
            match step % 7 {
                3 => w.raise_violated(&sat, &sol),
                6 => w.lower_satisfied(&sat, &sol),
                _ => {
                    let mv = SatFlipNeighbor::new(&sat, &sol, rng.random_range(0..5));
                    w.apply(&sat, &mut sol, &mv).unwrap();
                }
            }
            assert_priced(&sat, &sol, &w);
            assert!((0..w.len()).all(|c| (1..=w.cap()).contains(&w.weight(c))));
        }
        assert!(
            (0..w.len()).any(|c| w.weight(c) > 1),
            "some weight was raised"
        );
    }

    /// An empty clause is never satisfied and no flip reaches it, so it is
    /// left out of the unsatisfied and has no tight move.
    #[test]
    fn an_empty_clause_has_no_tight_move() {
        let mut sat = Sat::new(2);
        sat.add_clause([]);
        sat.add_clause([1]);
        let sol = SatSolution::new_from_assignment(&sat, vec![false, false]);
        let w = Weights::new(&sat, &sol, 1);
        assert_eq!(w.violated().as_slice(), &[1]);
        let mut out = Vec::new();
        sat.tight_moves(&sol, w.prices(), 0, &mut out);
        assert!(out.is_empty());
    }

    /// A tight move of an unsatisfied clause satisfies it, and one of a
    /// satisfied clause turns a true literal false and leaves it satisfied.
    #[test]
    fn tight_moves_satisfy_a_clause_or_leave_it_satisfied() {
        let sat = formula();
        let sol = SatSolution::new_from_assignment(&sat, vec![true, true, true, false, false]);
        let w = Weights::new(&sat, &sol, 1);
        for c in 0..sat.n_clauses() {
            let clause = sat.clause(c);
            let was = Sat::is_clause_satisfied(clause, &sol.x);
            let mut out = Vec::new();
            sat.tight_moves(&sol, w.prices(), c, &mut out);
            let mut vars: Vec<usize> = out.iter().map(|mv| mv.i).collect();
            for mv in &out {
                let mut x = sol.x.clone();
                x[mv.i] = !x[mv.i];
                assert!(
                    Sat::is_clause_satisfied(clause, &x),
                    "clause {c}, flip {}",
                    mv.i
                );
                if was {
                    assert!(
                        clause
                            .iter()
                            .any(|&l| l.unsigned_abs() as usize - 1 == mv.i
                                && sol.x[mv.i] == (l > 0))
                    );
                }
            }
            let n = vars.len();
            vars.sort_unstable();
            vars.dedup();
            assert_eq!(vars.len(), n, "no variable twice in clause {c}");
        }
        // [1, 2, -3] holds 1 and 2 true and -3 false, so 1 and 2 can turn false.
        let mut out = Vec::new();
        sat.tight_moves(&sol, w.prices(), 0, &mut out);
        assert_eq!(out.iter().map(|mv| mv.i).collect::<Vec<_>>(), vec![0, 1]);
        // [-1, 4] holds only 4 false and -1 false, so it is unsatisfied and
        // both flips satisfy it.
        out.clear();
        sat.tight_moves(&sol, w.prices(), 1, &mut out);
        assert_eq!(out.iter().map(|mv| mv.i).collect::<Vec<_>>(), vec![0, 3]);
    }
}
