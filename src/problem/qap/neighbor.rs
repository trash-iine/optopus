use crate::trait_defs::Rankable;
use rand::rngs::SmallRng;

use super::problem::{Qap, QapSolution};
use crate::building_blocks::search::{TabuMemory, pair_scan, random_distinct_pair};
use crate::error::OptError;
use crate::search_state::{EnabledTabu, Evaluable, Evaluate, MoveToNeighbor};

/// Exchanges the locations of two facilities, the move every QAP local search
/// is built on.
///
/// Priced by the O(n) formula of [`Qap::swap_delta`], or read from the table
/// of every pair's delta the solution keeps once a full scan has built it.
/// The move remembers the
/// two locations it takes the facilities from, which is what the tabu list of
/// Taillard's robust tabu search is about.
#[derive(Debug, Clone, Rankable)]
pub struct QapSwapNeighbor {
    /// The first facility.
    pub i: usize,
    /// The second facility, `i < j`.
    pub j: usize,
    /// Where `i` is before the move.
    pub loc_i: usize,
    /// Where `j` is before the move.
    pub loc_j: usize,
    /// Change in cost (negative = improvement).
    pub gain: i64,
}

impl QapSwapNeighbor {
    /// Builds the exchange of facilities `i` and `j`.
    ///
    /// # Panics
    ///
    /// Panics if `i` or `j` is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// let qap = Qap::new(
    ///     "tiny",
    ///     vec![vec![0, 5, 0], vec![5, 0, 1], vec![0, 1, 0]],
    ///     vec![vec![0, 1, 9], vec![1, 0, 2], vec![9, 2, 0]],
    /// );
    /// let sol = qap.solution_from_assignment(vec![0, 2, 1])?;
    /// let m = QapSwapNeighbor::new(&qap, &sol, 1, 2);
    ///
    /// let mut after = sol.clone();
    /// m.apply_to_solution(&qap, &mut after).unwrap();
    /// assert_eq!(after.assignment, vec![0, 1, 2]);
    /// assert_eq!(after.objective, sol.objective + m.gain);
    /// # Ok::<(), optopus::error::OptError>(())
    /// ```
    pub fn new(prob: &Qap, sol: &QapSolution, i: usize, j: usize) -> Self {
        let (i, j) = (i.min(j), i.max(j));
        Self {
            i,
            j,
            loc_i: sol.assignment[i],
            loc_j: sol.assignment[j],
            gain: prob.swap_delta(&sol.assignment, i, j),
        }
    }
}

impl Evaluate for QapSwapNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain as f64)
    }
}

impl EnabledTabu for QapSwapNeighbor {
    /// The move is tabu only when it would put both facilities back where
    /// they recently were, Taillard's rule.
    fn is_move_enabled(&self, tabu: &TabuMemory, iteration: u64) -> bool {
        tabu.is_enabled((self.i, self.loc_j), iteration)
            || tabu.is_enabled((self.j, self.loc_i), iteration)
    }

    /// Applying it forbids each facility its old location.
    fn add_to_tabu_map(&self, tabu: &mut TabuMemory, iteration: u64, rng: &mut SmallRng) {
        tabu.forbid((self.i, self.loc_i), iteration, rng);
        tabu.forbid((self.j, self.loc_j), iteration, rng);
    }
}

impl MoveToNeighbor<Qap> for QapSwapNeighbor {
    /// Hands this move's [`EnabledTabu`] policy to the search state, which is
    /// what holds the tabu map.
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

    /// Compares through the cached `gain`, exact since the costs are
    /// integers, relative to the solution the move was built from (`src`).
    fn move_to_be_better_than(&self, _prob: &Qap, src: &QapSolution, other: &QapSolution) -> bool {
        self.evaluate()
            .improves_over(src.evaluate(), other.evaluate())
    }

    fn apply_to_solution(&self, prob: &Qap, sol: &mut QapSolution) -> Result<(), OptError> {
        prob.apply_swap(sol, self.i, self.j, self.gain);
        Ok(())
    }

    /// Reads every pair's delta from the table the solution keeps, building
    /// it on the first scan, so a scan after the first is O(n²).
    fn iter(prob: &Qap, sol: &QapSolution) -> impl Iterator<Item = Self> + Send {
        let n = sol.assignment.len();
        let deltas = prob.swap_deltas(sol);
        pair_scan(
            n,
            move |i| ((), i + 1..n),
            move |_, i, j| {
                Some(match deltas {
                    Some(d) => Self {
                        i,
                        j,
                        loc_i: sol.assignment[i],
                        loc_j: sol.assignment[j],
                        gain: d[i * n + j],
                    },
                    None => Self::new(prob, sol, i, j),
                })
            },
        )
    }

    fn random_neighbor(prob: &Qap, sol: &QapSolution, rng: &mut SmallRng) -> Option<Self> {
        let (i, j) = random_distinct_pair(sol.assignment.len(), rng)?;
        Some(Self::new(prob, sol, i, j))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_state::ProblemTrait;
    use rand::{Rng, SeedableRng};

    #[test]
    fn moves_land_on_the_cost_they_promise() {
        let mut rng = SmallRng::seed_from_u64(3);
        let mut matrix = |n: usize| -> Vec<Vec<i64>> {
            (0..n)
                .map(|_| (0..n).map(|_| rng.random_range(0..10)).collect())
                .collect()
        };
        let qap = Qap::new("random", matrix(7), matrix(7));
        let sol = qap.new_solution(&mut SmallRng::seed_from_u64(4));
        let moves: Vec<_> = QapSwapNeighbor::iter(&qap, &sol).collect();
        assert_eq!(moves.len(), 7 * 6 / 2);
        for m in moves {
            let mut after = sol.clone();
            m.apply_to_solution(&qap, &mut after).unwrap();
            assert_eq!(after.objective, qap.cost(&after.assignment));
        }
    }

    /// The table a scan builds stays equal to pricing every pair anew across
    /// a run of applied swaps, on asymmetric matrices with non-zero
    /// diagonals, and a change made around the moves is never read from it.
    #[test]
    fn the_delta_table_follows_the_applied_swaps() {
        let mut rng = SmallRng::seed_from_u64(5);
        let mut matrix = |n: usize| -> Vec<Vec<i64>> {
            (0..n)
                .map(|_| (0..n).map(|_| rng.random_range(-5..10)).collect())
                .collect()
        };
        let qap = Qap::new("random", matrix(9), matrix(9));
        let mut sol = qap.new_solution(&mut SmallRng::seed_from_u64(6));
        let mut rng = SmallRng::seed_from_u64(7);
        let priced_anew = |sol: &QapSolution| -> Vec<i64> {
            QapSwapNeighbor::iter(&qap, sol)
                .map(|m| qap.swap_delta(&sol.assignment, m.i, m.j))
                .collect()
        };
        for step in 0..60 {
            let scanned: Vec<i64> = QapSwapNeighbor::iter(&qap, &sol).map(|m| m.gain).collect();
            assert_eq!(scanned, priced_anew(&sol), "step {step}");
            let m = QapSwapNeighbor::random_neighbor(&qap, &sol, &mut rng).unwrap();
            m.apply_to_solution(&qap, &mut sol).unwrap();
            assert_eq!(sol.objective, qap.cost(&sol.assignment));
            if step % 20 == 19 {
                sol.assignment.swap(0, 1);
                sol.objective = qap.cost(&sol.assignment);
            }
        }
    }
}
