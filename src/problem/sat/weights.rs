//! Clause prices, the [`WeightedConstraints`] side of [`Sat`].

use super::{Sat, SatFlipNeighbor, SatSolution};
use crate::building_blocks::search::IndexSet;
use crate::error::OptError;
use crate::trait_defs::{MoveToNeighbor, WeightedConstraints};

/// What a [`Sat`] keeps to price a flip under clause weights, the weighted
/// flip gains.
///
/// The constraints are the clauses, and the weighted cost of an assignment is
/// the total weight of the clauses it leaves unsatisfied.
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

    fn weighted_gain(&self, _: &SatSolution, prices: &SatPrices, mv: &SatFlipNeighbor) -> f64 {
        prices.gain[mv.i] as f64
    }

    fn apply_weighted(
        &self,
        sol: &mut SatSolution,
        mv: &SatFlipNeighbor,
        weight: &[u32],
        prices: &mut SatPrices,
        unsatisfied: &mut IndexSet,
    ) -> Result<(), OptError> {
        // The clauses holding the variable leave the gains at their weight
        // under the old value and come back under the new one, as the flip
        // itself does for the unweighted gains.
        for &c in self.clauses_of(mv.i) {
            let w = weight[c] as i64;
            self.add_clause_shares(c, &sol.x, &mut prices.gain, -w);
        }
        mv.apply_to_solution(self, sol)?;
        for &c in self.clauses_of(mv.i) {
            let w = weight[c] as i64;
            if self.add_clause_shares(c, &sol.x, &mut prices.gain, w) {
                unsatisfied.remove(c);
            } else {
                unsatisfied.insert(c);
            }
        }
        Ok(())
    }

    #[inline]
    fn reweight(&self, sol: &SatSolution, prices: &mut SatPrices, c: usize, delta: i64) {
        self.add_clause_shares(c, &sol.x, &mut prices.gain, delta);
    }

    /// The flips of the variables of an unsatisfied clause. Every literal of
    /// such a clause is false, so each of them satisfies it. An assignment
    /// that satisfies every clause is optimal and has no escape, which the
    /// default of [`objective_escapes`](WeightedConstraints::objective_escapes)
    /// gives.
    fn repairs(&self, sol: &SatSolution, _: &SatPrices, c: usize, out: &mut Vec<SatFlipNeighbor>) {
        let clause = self.clause(c);
        let var = |lit: i64| lit.unsigned_abs() as usize - 1;
        for (p, &lit) in clause.iter().enumerate() {
            // A variable the clause holds twice is one flip.
            if clause[..p].iter().all(|&l| var(l) != var(lit)) {
                out.push(SatFlipNeighbor::new(self, sol, var(lit)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::ConstraintWeights;
    use crate::trait_defs::ProblemTrait;
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

    /// The weighted cost, from scratch.
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

    /// An empty clause is never satisfied and no flip reaches it, so it
    /// offers no escape and does not end a search that can still satisfy
    /// other clauses.
    #[test]
    fn an_empty_clause_is_not_an_escape() {
        let mut sat = Sat::new(2);
        sat.add_clause([]);
        sat.add_clause([1]);
        let sol = SatSolution::new_from_assignment(&sat, vec![false, false]);
        let w = Weights::new(&sat, &sol, 1);
        assert_eq!(w.violated().as_slice(), &[1]);
        let mut rng = SmallRng::seed_from_u64(0);
        let mut out = Vec::new();
        for _ in 0..20 {
            out.clear();
            w.escape_moves(&sat, &sol, &mut rng, &mut out);
            assert_eq!(out.len(), 1, "the one clause a flip can satisfy is drawn");
            assert_eq!(out[0].i, 0);
        }
    }

    #[test]
    fn escape_moves_satisfy_one_unsatisfied_clause() {
        let sat = formula();
        let mut rng = SmallRng::seed_from_u64(5);
        let sol = SatSolution::new_from_assignment(&sat, vec![false, false, true, false, false]);
        let w = Weights::new(&sat, &sol, 1);
        let mut out = Vec::new();
        for _ in 0..50 {
            out.clear();
            w.escape_moves(&sat, &sol, &mut rng, &mut out);
            assert!(!out.is_empty());
            let fixed: Vec<usize> = w
                .violated()
                .as_slice()
                .iter()
                .copied()
                .filter(|&c| {
                    out.iter().all(|mv| {
                        let mut x = sol.x.clone();
                        x[mv.i] = !x[mv.i];
                        Sat::is_clause_satisfied(sat.clause(c), &x)
                    })
                })
                .collect();
            assert!(!fixed.is_empty(), "every move satisfies the drawn clause");
            let mut vars: Vec<usize> = out.iter().map(|mv| mv.i).collect();
            vars.sort_unstable();
            vars.dedup();
            assert_eq!(vars.len(), out.len(), "no variable twice");
        }
    }

    #[test]
    fn a_satisfying_assignment_has_no_escape() {
        let mut sat = Sat::new(2);
        sat.add_clause([1, 2]);
        let sol = SatSolution::new_from_assignment(&sat, vec![true, false]);
        let w = Weights::new(&sat, &sol, 1);
        let mut out = Vec::new();
        w.escape_moves(&sat, &sol, &mut SmallRng::seed_from_u64(0), &mut out);
        assert!(out.is_empty());
    }
}
