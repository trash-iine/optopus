//! Clause weights, the [`WeightedConstraints`] side of [`Sat`].

use super::{Sat, SatFlipNeighbor, SatSolution};
use crate::building_blocks::search::IndexSet;
use crate::error::OptError;
use crate::trait_defs::{MoveToNeighbor, WeightedConstraints};
use rand::Rng;

/// One weight per clause of a [`Sat`], with the flip gains under them.
///
/// The constraints are the clauses, and the weighted cost of an assignment is
/// the total weight of the clauses it leaves unsatisfied.
#[derive(Debug, Clone)]
pub struct SatWeights {
    /// `weight[c]` is the weight of clause `c`.
    weight: Vec<u32>,
    /// `gain[i]` is the weight flipping variable `i` satisfies minus the
    /// weight it leaves unsatisfied.
    gain: Vec<i64>,
    /// The clauses the assignment leaves unsatisfied.
    unsatisfied: IndexSet,
}

impl SatWeights {
    /// The weight of clause `c`.
    pub fn weight(&self, c: usize) -> u32 {
        self.weight[c]
    }

    /// How many clauses the assignment leaves unsatisfied.
    pub fn unsatisfied_count(&self) -> usize {
        self.unsatisfied.len()
    }
}

impl WeightedConstraints for Sat {
    type Move = SatFlipNeighbor;
    type Weights = SatWeights;

    fn unit_weights(&self, sol: &SatSolution) -> SatWeights {
        let mut unsatisfied = IndexSet::new(self.n_clauses());
        for (c, clause) in self.all_clauses().enumerate() {
            if !Sat::is_clause_satisfied(clause, &sol.x) {
                unsatisfied.insert(c);
            }
        }
        SatWeights {
            weight: vec![1; self.n_clauses()],
            // At weight one the weighted gain is the count the solution keeps.
            gain: sol.gain.clone(),
            unsatisfied,
        }
    }

    fn constraint_count(&self) -> usize {
        self.n_clauses()
    }

    fn weighted_gain(&self, _: &SatSolution, weights: &SatWeights, mv: &SatFlipNeighbor) -> f64 {
        weights.gain[mv.i] as f64
    }

    fn apply_weighted(
        &self,
        sol: &mut SatSolution,
        weights: &mut SatWeights,
        mv: &SatFlipNeighbor,
    ) -> Result<(), OptError> {
        // The clauses holding the variable leave the gains at their weight
        // under the old value and come back under the new one, as the flip
        // itself does for the unweighted gains.
        for &c in self.clauses_of(mv.i) {
            let w = weights.weight[c] as i64;
            self.add_clause_shares(c, &sol.x, &mut weights.gain, -w);
        }
        mv.apply_to_solution(self, sol)?;
        for &c in self.clauses_of(mv.i) {
            let w = weights.weight[c] as i64;
            if self.add_clause_shares(c, &sol.x, &mut weights.gain, w) {
                weights.unsatisfied.remove(c);
            } else {
                weights.unsatisfied.insert(c);
            }
        }
        Ok(())
    }

    fn raise_violated(&self, sol: &SatSolution, weights: &mut SatWeights, cap: u32) {
        for &c in weights.unsatisfied.as_slice() {
            if weights.weight[c] < cap {
                weights.weight[c] += 1;
                self.add_clause_shares(c, &sol.x, &mut weights.gain, 1);
            }
        }
    }

    fn lower_satisfied(&self, sol: &SatSolution, weights: &mut SatWeights) {
        for c in 0..self.n_clauses() {
            if weights.weight[c] > 1 && !weights.unsatisfied.contains(c) {
                weights.weight[c] -= 1;
                self.add_clause_shares(c, &sol.x, &mut weights.gain, -1);
            }
        }
    }

    /// The flips of the variables of one unsatisfied clause. Every literal of
    /// such a clause is false, so each of them satisfies it. An assignment
    /// that satisfies every clause is optimal and has no escape.
    fn escape_moves(
        &self,
        sol: &SatSolution,
        weights: &SatWeights,
        rng: &mut rand::rngs::SmallRng,
        out: &mut Vec<SatFlipNeighbor>,
    ) {
        let unsatisfied = weights.unsatisfied.as_slice();
        if unsatisfied.is_empty() {
            return;
        }
        let clause = self.clause(unsatisfied[rng.random_range(0..unsatisfied.len())]);
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
    use crate::trait_defs::ProblemTrait;
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

    /// The weighted cost, from scratch.
    fn cost(sat: &Sat, x: &[bool], w: &SatWeights) -> i64 {
        sat.all_clauses()
            .enumerate()
            .filter(|(_, c)| !Sat::is_clause_satisfied(c, x))
            .map(|(c, _)| w.weight[c] as i64)
            .sum()
    }

    /// Every cached gain is the drop in the from-scratch cost, and the
    /// unsatisfied set is the clauses the assignment fails.
    fn assert_priced(sat: &Sat, sol: &SatSolution, w: &SatWeights) {
        let now = cost(sat, &sol.x, w);
        for i in 0..sat.n_vars() {
            let mut x = sol.x.clone();
            x[i] = !x[i];
            assert_eq!(w.gain[i], now - cost(sat, &x, w), "gain of {i}");
        }
        let mut expected: Vec<usize> = (0..sat.n_clauses())
            .filter(|&c| !Sat::is_clause_satisfied(sat.clause(c), &sol.x))
            .collect();
        let mut got = w.unsatisfied.as_slice().to_vec();
        expected.sort_unstable();
        got.sort_unstable();
        assert_eq!(got, expected);
    }

    #[test]
    fn prices_follow_flips_and_weight_changes() {
        let sat = formula();
        let mut rng = SmallRng::seed_from_u64(11);
        let mut sol = sat.new_solution(&mut rng);
        let mut w = sat.unit_weights(&sol);
        assert_priced(&sat, &sol, &w);
        for step in 0..300 {
            match step % 7 {
                3 => sat.raise_violated(&sol, &mut w, 4),
                6 => sat.lower_satisfied(&sol, &mut w),
                _ => {
                    let mv = SatFlipNeighbor::new(&sat, &sol, rng.random_range(0..5));
                    sat.apply_weighted(&mut sol, &mut w, &mv).unwrap();
                }
            }
            assert_priced(&sat, &sol, &w);
            assert!(w.weight.iter().all(|&x| (1..=4).contains(&x)));
        }
        assert!(w.weight.iter().any(|&x| x > 1), "some weight was raised");
    }

    #[test]
    fn escape_moves_satisfy_one_unsatisfied_clause() {
        let sat = formula();
        let mut rng = SmallRng::seed_from_u64(5);
        let sol = SatSolution::new_from_assignment(&sat, vec![false, false, true, false, false]);
        let w = sat.unit_weights(&sol);
        let mut out = Vec::new();
        for _ in 0..50 {
            out.clear();
            sat.escape_moves(&sol, &w, &mut rng, &mut out);
            assert!(!out.is_empty());
            let fixed: Vec<usize> = w
                .unsatisfied
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
        let w = sat.unit_weights(&sol);
        let mut out = Vec::new();
        sat.escape_moves(&sol, &w, &mut SmallRng::seed_from_u64(0), &mut out);
        assert!(out.is_empty());
    }
}
