use super::problem::{Coefficient, Qubo};
use crate::{
    error::OptError,
    problem::qubo::problem::QuboSolution,
    search_state::{EnabledTabu, Evaluable, MoveToNeighbor, Rankable},
};

/// A flip move that toggles a single variable `i`.
///
/// `gain` is the change in energy after the flip (negative = improvement, since QUBO is minimized).
#[derive(Debug, Clone)]
pub struct QuboFlipNeighbour {
    /// Index of the variable to flip.
    pub i: usize,
    /// Change in objective value when this variable is flipped (negative = improvement).
    pub gain: Coefficient,
}

impl Rankable for QuboFlipNeighbour {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain < other.gain
    }
}

impl EnabledTabu for QuboFlipNeighbour {
    type TabuMap = std::collections::HashMap<usize, u64>;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        tabu_map
            .get(&self.i)
            .map_or(true, |&tabu_tenure| iteration > tabu_tenure)
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
    ) {
        let tabu_duration = rand::random_range(tabu_tenure.0..=tabu_tenure.1);
        tabu_map.insert(self.i, iteration + tabu_duration);
    }
}

impl Evaluable<Coefficient> for QuboFlipNeighbour {
    fn evaluate(&self) -> Coefficient {
        self.gain
    }
}

// QUBO is a minimization problem: gain = change in energy (positive = worsening).
// Passing gain directly to SA yields the correct acceptance probability exp(-gain/T).
impl Evaluable<f64> for QuboFlipNeighbour {
    fn evaluate(&self) -> f64 {
        self.gain as f64
    }
}

impl MoveToNeighbor<Qubo> for QuboFlipNeighbour {
    fn apply_to_solution(&self, prob: &Qubo, sol: &mut QuboSolution) -> Result<(), OptError> {
        let bi = *sol.x.get(&self.i).ok_or_else(|| {
            OptError::InvalidState(format!("{} is not found in solution", self.i))
        })?;

        // Flip the variable
        sol.x.insert(self.i, !bi);

        // Update gain
        sol.gain.insert(self.i, -self.gain);

        for (&j, &q) in prob.iter_on_adjacency(self.i) {
            if j == self.i {
                continue;
            }

            let bj = *sol
                .x
                .get(&j)
                .ok_or_else(|| OptError::InvalidState(format!("{} is not found in solution", j)))?;
            let delta = if bi == bj { q } else { -q };
            *sol.gain.entry(j).or_insert(0) += delta;
        }

        // Update objective
        sol.objective += self.gain;

        Ok(())
    }

    fn iter(prob: &Qubo, sol: &QuboSolution) -> impl Iterator<Item = Self> + Send {
        prob.iter_on_variables().map(move |&i| QuboFlipNeighbour {
            i,
            gain: sol.gain[&i],
        })
    }

    fn move_to_be_better_than(&self, _: &Qubo, src: &QuboSolution, other: &QuboSolution) -> bool {
        self.gain + src.objective < other.objective
    }
}

/// A swap move that simultaneously flips variables `i` and `j`.
///
/// `gain` is the combined change in energy (negative = improvement).
#[derive(Debug, Clone)]
pub struct QuboSwapNeighbour {
    /// Index of the first variable to flip.
    pub i: usize,
    /// Index of the second variable to flip.
    pub j: usize,
    /// Combined change in objective value when both variables are flipped (negative = improvement).
    pub gain: Coefficient,
}

impl Rankable for QuboSwapNeighbour {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain < other.gain
    }
}

impl Evaluable<f64> for QuboSwapNeighbour {
    fn evaluate(&self) -> f64 {
        self.gain as f64
    }
}

impl EnabledTabu for QuboSwapNeighbour {
    type TabuMap = std::collections::HashMap<usize, u64>;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        let enabled_i = tabu_map.get(&self.i).map_or(true, |&t| iteration > t);
        let enabled_j = tabu_map.get(&self.j).map_or(true, |&t| iteration > t);
        enabled_i && enabled_j
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
    ) {
        let d = rand::random_range(tabu_tenure.0..=tabu_tenure.1);
        tabu_map.insert(self.i, iteration + d);
        let d = rand::random_range(tabu_tenure.0..=tabu_tenure.1);
        tabu_map.insert(self.j, iteration + d);
    }
}

impl MoveToNeighbor<Qubo> for QuboSwapNeighbour {
    fn apply_to_iteration(&self, iter: u64) -> u64 {
        iter + 2
    }

    fn apply_to_solution(&self, prob: &Qubo, sol: &mut QuboSolution) -> Result<(), OptError> {
        let flip_i = QuboFlipNeighbour {
            i: self.i,
            gain: sol.gain[&self.i],
        };
        flip_i.apply_to_solution(prob, sol)?;

        let flip_j = QuboFlipNeighbour {
            i: self.j,
            gain: sol.gain[&self.j],
        };
        flip_j.apply_to_solution(prob, sol)?;

        Ok(())
    }

    fn iter(prob: &Qubo, sol: &QuboSolution) -> impl Iterator<Item = Self> + Send {
        prob.iter_on_variables().flat_map(move |&i| {
            prob.iter_on_variables()
                .filter(move |&&j| j < i && (sol.x[&i] ^ sol.x[&j]))
                .map(move |&j| Self {
                    i,
                    j,
                    gain: sol.gain[&i] + sol.gain[&j]
                        - if let Some(q) = prob.get_q(i, j) { q } else { 0 },
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
    use std::collections::HashMap;

    fn make_qubo() -> Qubo {
        let mut q = Qubo::new();
        // 3-variable QUBO: Q[0][1]=1, Q[1][2]=2, Q[0][2]=3, Q[2][2]=1
        q.set_q(0, 1, 1);
        q.set_q(1, 2, 2);
        q.set_q(0, 2, 3);
        q.set_q(2, 2, 1);
        q
    }

    fn make_solution(qubo: &Qubo, x: HashMap<usize, bool>) -> QuboSolution {
        let gain: HashMap<_, _> = x.keys().map(|&i| (i, qubo.calculate_gain(&x, i))).collect();
        let objective = qubo.calculate_energy(&x);
        QuboSolution { x, gain, objective }
    }

    #[test]
    fn test_flip_apply_consistency() {
        let qubo = make_qubo();
        let sol = make_solution(&qubo, HashMap::from([(0, true), (1, false), (2, true)]));

        for neighbor in QuboFlipNeighbour::iter(&qubo, &sol) {
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
                    s.gain[&i], expected_gain,
                    "flip {}: gain[{}]={} expected={}",
                    neighbor.i, i, s.gain[&i], expected_gain
                );
            }
        }
    }

    #[test]
    fn test_flip_gain_matches_energy_delta() {
        let qubo = make_qubo();
        let sol = make_solution(&qubo, HashMap::from([(0, true), (1, false), (2, true)]));

        for neighbor in QuboFlipNeighbour::iter(&qubo, &sol) {
            let mut flipped_x = sol.x.clone();
            flipped_x.insert(neighbor.i, !flipped_x[&neighbor.i]);
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
        let sol = make_solution(&qubo, HashMap::from([(0, true), (1, false), (2, true)]));

        for neighbor in QuboSwapNeighbour::iter(&qubo, &sol) {
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
        let sol = make_solution(&qubo, HashMap::from([(0, true), (1, false), (2, true)]));

        for neighbor in QuboSwapNeighbour::iter(&qubo, &sol) {
            let mut flipped_x = sol.x.clone();
            flipped_x.insert(neighbor.i, !flipped_x[&neighbor.i]);
            flipped_x.insert(neighbor.j, !flipped_x[&neighbor.j]);
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
}
