//! Neighborhood move types for the [`Qubo`] problem.
//!
//! Two move types are provided:
//!
//! - [`QuboFlipNeighbor`] — flip a single variable (O(degree) update)
//! - [`QuboSwapNeighbor`] — swap two variables with different values (two sequential flips)
//!
//! Both implement [`MoveToNeighbor`], [`Evaluate`], and [`EnabledTabu`], so they
//! work with all heuristics ([`LocalSearch`], [`TabuSearch`], [`SimulatedAnnealing`], etc.).
//!
//! [`LocalSearch`]: crate::heuristic::LocalSearch
//! [`TabuSearch`]: crate::heuristic::TabuSearch
//! [`SimulatedAnnealing`]: crate::heuristic::SimulatedAnnealing
//! [`MoveToNeighbor`]: crate::search_state::MoveToNeighbor
//! [`Evaluate`]: crate::search_state::Evaluate
//! [`EnabledTabu`]: crate::search_state::EnabledTabu

use super::problem::{Coefficient, Qubo};
use crate::{
    common::{VarTabuMap, add_var_to_tabu, is_var_enabled},
    error::OptError,
    problem::qubo::problem::QuboSolution,
    search_state::{EnabledTabu, Evaluable, Evaluate, MoveToNeighbor, Rankable},
};
use rand::Rng;

/// A flip move that toggles a single variable `i`.
///
/// `gain` is the change in energy after the flip (negative = improvement, since QUBO is minimized).
///
/// # Usage
///
/// ```
/// use optopus::prelude::*;
///
/// let qubo = Qubo::from_entries([(0, 1, 1), (1, 2, 1)]);
/// let mut state = SearchState::new(&qubo);
///
/// // Use with any heuristic:
/// LocalSearch::<QuboFlipNeighbor>::new(StopCondition::iterations(1000))
///     .run(&mut state).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct QuboFlipNeighbor {
    /// Index of the variable to flip.
    pub i: usize,
    /// Change in objective value when this variable is flipped (negative = improvement).
    pub gain: Coefficient,
}

impl Rankable for QuboFlipNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain < other.gain
    }
}

impl EnabledTabu for QuboFlipNeighbor {
    type TabuMap = VarTabuMap;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        is_var_enabled(tabu_map, self.i, iteration)
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
        rng: &mut rand::rngs::SmallRng,
    ) {
        add_var_to_tabu(tabu_map, self.i, iteration, tabu_tenure, rng);
    }
}

// QUBO is a minimization problem: gain = change in energy (positive = worsening).
impl Evaluate for QuboFlipNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain as f64)
    }
}

impl Evaluate<Coefficient> for QuboFlipNeighbor {
    fn evaluate(&self) -> Evaluable<Coefficient> {
        Evaluable::Minimize(self.gain)
    }
}

impl MoveToNeighbor<Qubo> for QuboFlipNeighbor {
    /// Applies the flip move: toggles variable `self.i`.
    ///
    /// Updates the solution in-place in O(degree(i)):
    /// 1. Flips `solution.x[i]`
    /// 2. Inverts `solution.gain[i]`
    /// 3. Updates `gain[j]` for each neighbor `j` of `i`
    /// 4. Adds `self.gain` to `solution.objective`
    ///
    /// If the `negative_gain` index is enabled, it is maintained incrementally.
    fn apply_to_solution(&self, prob: &Qubo, sol: &mut QuboSolution) -> Result<(), OptError> {
        let bi = sol.x[self.i];

        // Flip the variable
        sol.x[self.i] = !bi;

        // Update gain for flipped variable
        let new_gain_i = -self.gain;
        sol.update_negative_gain_membership(self.i, new_gain_i);
        sol.gain[self.i] = new_gain_i;

        for &(j, q) in prob.iter_on_adjacency(self.i) {
            if j == self.i {
                continue;
            }
            let bj = sol.x[j];
            let delta = if bi == bj { q } else { -q };
            let new_g = sol.gain[j] + delta;
            sol.update_negative_gain_membership(j, new_g);
            sol.gain[j] = new_g;
        }

        // Update objective
        sol.objective += self.gain;

        Ok(())
    }

    /// Returns a lazy iterator over all possible flip moves (one per variable).
    fn iter(prob: &Qubo, sol: &QuboSolution) -> impl Iterator<Item = Self> + Send {
        prob.iter_on_variables().map(move |&i| QuboFlipNeighbor {
            i,
            gain: sol.gain[i],
        })
    }

    fn move_to_be_better_than(&self, _: &Qubo, src: &QuboSolution, other: &QuboSolution) -> bool {
        self.gain + src.objective < other.objective
    }
}

impl QuboFlipNeighbor {
    /// Generates a random flip neighbor by uniformly selecting a variable from the problem.
    ///
    /// Useful as a perturbation step (e.g., in [`RandomWalk`](crate::heuristic::RandomWalk)).
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// let qubo = Qubo::from_entries([(0, 1, 1), (1, 2, 1)]);
    /// let mut state = SearchState::new(&qubo);
    /// let flip = QuboFlipNeighbor::random_neighbor(&qubo, &state.solution, &mut state.rng);
    /// println!("random flip: variable {}, gain {}", flip.i, flip.gain);
    /// ```
    pub fn random_neighbor(
        prob: &Qubo,
        sol: &QuboSolution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Self {
        let i = prob.variables[rng.random_range(0..prob.variables.len())];
        Self {
            i,
            gain: sol.gain[i],
        }
    }
}

/// A swap move that simultaneously flips variables `i` and `j`.
///
/// Only pairs where `i` and `j` have different values are generated.
/// Each swap counts as **2 iterations** (see [`apply_to_iteration`](MoveToNeighbor::apply_to_iteration)).
///
/// `gain` is the combined change in energy (negative = improvement).
///
/// # Usage
///
/// ```
/// use optopus::prelude::*;
///
/// let qubo = Qubo::from_entries([(0, 1, 1), (1, 2, 1), (0, 2, 1)]);
/// let mut state = SearchState::new(&qubo);
///
/// TabuSearch::<QuboSwapNeighbor>::new(
///     StopCondition::iterations(10_000),
///     (5, 10),
///     None,
/// ).run(&mut state).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct QuboSwapNeighbor {
    /// Index of the first variable to flip.
    pub i: usize,
    /// Index of the second variable to flip.
    pub j: usize,
    /// Combined change in objective value when both variables are flipped (negative = improvement).
    pub gain: Coefficient,
}

impl Rankable for QuboSwapNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain < other.gain
    }
}

impl Evaluate for QuboSwapNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain as f64)
    }
}

impl EnabledTabu for QuboSwapNeighbor {
    type TabuMap = VarTabuMap;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        is_var_enabled(tabu_map, self.i, iteration) && is_var_enabled(tabu_map, self.j, iteration)
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
        rng: &mut rand::rngs::SmallRng,
    ) {
        add_var_to_tabu(tabu_map, self.i, iteration, tabu_tenure, rng);
        add_var_to_tabu(tabu_map, self.j, iteration, tabu_tenure, rng);
    }
}

impl MoveToNeighbor<Qubo> for QuboSwapNeighbor {
    /// A swap counts as 2 iterations (one for each variable flip).
    fn apply_to_iteration(&self, iter: u64) -> u64 {
        iter + 2
    }

    /// Applies the swap by performing two sequential flips: first `i`, then `j`.
    fn apply_to_solution(&self, prob: &Qubo, sol: &mut QuboSolution) -> Result<(), OptError> {
        crate::common::apply_swap_as_two_flips(prob, sol, self.i, self.j)
    }

    /// Returns a lazy iterator over all valid swap pairs `(i, j)` where
    /// `i` and `j` have different values.
    fn iter(prob: &Qubo, sol: &QuboSolution) -> impl Iterator<Item = Self> + Send {
        prob.iter_on_variables().flat_map(move |&i| {
            prob.iter_on_variables()
                .filter(move |&&j| j < i && (sol.x[i] ^ sol.x[j]))
                .map(move |&j| Self {
                    i,
                    j,
                    gain: sol.gain[i] + sol.gain[j] - prob.get_q(i, j),
                })
        })
    }

    fn move_to_be_better_than(&self, _: &Qubo, src: &QuboSolution, other: &QuboSolution) -> bool {
        self.gain + src.objective < other.objective
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_state::SearchState;

    fn make_qubo() -> Qubo {
        let mut q = Qubo::new();
        // 3-variable QUBO: Q[0][1]=1, Q[1][2]=2, Q[0][2]=3, Q[2][2]=1
        q.set_q(0, 1, 1);
        q.set_q(1, 2, 2);
        q.set_q(0, 2, 3);
        q.set_q(2, 2, 1);
        q
    }

    fn make_solution(qubo: &Qubo, assignments: &[(usize, bool)]) -> QuboSolution {
        let n = qubo
            .iter_on_variables()
            .copied()
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        let mut x = vec![false; n];
        for &(i, v) in assignments {
            x[i] = v;
        }
        QuboSolution::new_from_assignment(qubo, x)
    }

    #[test]
    fn test_flip_apply_consistency() {
        let qubo = make_qubo();
        let sol = make_solution(&qubo, &[(0, true), (1, false), (2, true)]);

        for neighbor in QuboFlipNeighbor::iter(&qubo, &sol) {
            let mut s = sol.clone();
            neighbor.apply_to_solution(&qubo, &mut s).unwrap();

            // objective should be updated correctly
            let expected_obj = qubo.calculate_energy(&s.x);
            assert_eq!(
                s.objective, expected_obj,
                "flip {}: objective={} expected={}",
                neighbor.i, s.objective, expected_obj
            );

            // gain should be updated correctly
            for &i in qubo.iter_on_variables() {
                let expected_gain = qubo.calculate_gain(&s.x, i);
                assert_eq!(
                    s.gain[i], expected_gain,
                    "flip {}: gain[{}]={} expected={}",
                    neighbor.i, i, s.gain[i], expected_gain
                );
            }
        }
    }

    #[test]
    fn test_flip_gain_matches_energy_delta() {
        let qubo = make_qubo();
        let sol = make_solution(&qubo, &[(0, true), (1, false), (2, true)]);

        for neighbor in QuboFlipNeighbor::iter(&qubo, &sol) {
            let mut flipped_x = sol.x.clone();
            flipped_x[neighbor.i] = !flipped_x[neighbor.i];
            let expected_delta = qubo.calculate_energy(&flipped_x) - sol.objective;
            assert_eq!(
                neighbor.gain, expected_delta,
                "flip {}: gain={} expected delta={}",
                neighbor.i, neighbor.gain, expected_delta
            );
        }
    }

    #[test]
    fn test_swap_apply_consistency() {
        let qubo = make_qubo();
        let sol = make_solution(&qubo, &[(0, true), (1, false), (2, true)]);

        for neighbor in QuboSwapNeighbor::iter(&qubo, &sol) {
            let mut s = sol.clone();
            neighbor.apply_to_solution(&qubo, &mut s).unwrap();

            let expected_obj = qubo.calculate_energy(&s.x);
            assert_eq!(
                s.objective, expected_obj,
                "swap ({},{}): objective={} expected={}",
                neighbor.i, neighbor.j, s.objective, expected_obj
            );
        }
    }

    #[test]
    fn test_swap_gain_matches_energy_delta() {
        let qubo = make_qubo();
        let sol = make_solution(&qubo, &[(0, true), (1, false), (2, true)]);

        for neighbor in QuboSwapNeighbor::iter(&qubo, &sol) {
            let mut flipped_x = sol.x.clone();
            flipped_x[neighbor.i] = !flipped_x[neighbor.i];
            flipped_x[neighbor.j] = !flipped_x[neighbor.j];
            let expected_delta = qubo.calculate_energy(&flipped_x) - sol.objective;
            assert_eq!(
                neighbor.gain, expected_delta,
                "swap ({},{}): gain={} expected delta={}",
                neighbor.i, neighbor.j, neighbor.gain, expected_delta
            );
        }
    }

    #[test]
    fn test_search_state_new() {
        let qubo = make_qubo();
        let _state = SearchState::new(&qubo);
    }

    #[test]
    fn test_random_neighbor() {
        let qubo = Qubo::from_entries([(0, 1, 1), (1, 2, 2), (0, 2, 3)]);
        let mut state = SearchState::new(&qubo);
        for _ in 0..20 {
            let flip = QuboFlipNeighbor::random_neighbor(&qubo, &state.solution, &mut state.rng);
            assert!(flip.i < qubo.len(), "random neighbor index out of bounds");
            assert_eq!(flip.gain, state.solution.gain[flip.i]);
        }
    }
}
