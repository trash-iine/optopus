use rand::rngs::SmallRng;

use super::problem::{Qap, QapSolution};
use crate::building_blocks::search::{TabuMemory, random_distinct_pair};
use crate::error::OptError;
use crate::search_state::{EnabledTabu, Evaluable, Evaluate, MoveToNeighbor};

/// Exchanges the locations of two facilities, the move every QAP local search
/// is built on.
///
/// Priced by the O(n) formula of [`Qap::swap_delta`]. The move remembers the
/// two locations it takes the facilities from, which is what the tabu list of
/// Taillard's robust tabu search is about.
#[derive(Debug, Clone)]
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
    /// let sol = qap.solution_from_assignment(vec![0, 2, 1]);
    /// let m = QapSwapNeighbor::new(&qap, &sol, 1, 2);
    ///
    /// let mut after = sol.clone();
    /// m.apply_to_solution(&qap, &mut after).unwrap();
    /// assert_eq!(after.assignment, vec![0, 1, 2]);
    /// assert_eq!(after.objective, sol.objective + m.gain);
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

    fn apply_to_solution(&self, _prob: &Qap, sol: &mut QapSolution) -> Result<(), OptError> {
        sol.assignment.swap(self.i, self.j);
        sol.objective += self.gain;
        Ok(())
    }

    fn iter(prob: &Qap, sol: &QapSolution) -> impl Iterator<Item = Self> + Send {
        let n = sol.assignment.len();
        (0..n)
            .flat_map(move |i| (i + 1..n).map(move |j| (i, j)))
            .map(move |(i, j)| Self::new(prob, sol, i, j))
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
}
