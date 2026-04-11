//! Neighborhood move types for the [`MaxCut`] problem.

use super::MaxCut;
use crate::{
    error::OptError,
    problem::max_cut::problem::MaxCutSolution,
    search_state::{EnabledTabu, Evaluable, Evaluate, MoveToNeighbor, Rankable},
};

/// A flip move that transfers vertex `i` to the opposite partition side.
///
/// `gain` holds the change in cut weight after the flip (positive = improvement).
#[derive(Debug, Clone, Copy)]
pub struct MaxCutFlipNeighbor {
    /// Index of the vertex to flip.
    pub i: usize,
    /// Change in cut weight after the flip.
    pub gain: f32,
}
impl Rankable for MaxCutFlipNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain > other.gain
    }
}

impl EnabledTabu for MaxCutFlipNeighbor {
    type TabuMap = std::collections::HashMap<usize, u64>;

    /// A flip move is tabu if the vertex `i` is in the tabu map with a tenure greater than the current iteration.
    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        tabu_map
            .get(&self.i)
            .map_or(true, |&tabu_tenure| iteration > tabu_tenure)
    }

    /// When a flip move is applied,
    /// the vertex `i` is added to the tabu map with a tenure
    /// randomly chosen between `tabu_tenure.0` and `tabu_tenure.1`.
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

impl MoveToNeighbor<MaxCut> for MaxCutFlipNeighbor {
    fn apply_to_solution(
        &self,
        prob: &MaxCut,
        solution: &mut MaxCutSolution,
    ) -> Result<(), OptError> {
        let bi = solution.cut[self.i];

        // Flip
        solution.cut[self.i] = !bi;

        // Update the gain for the flipped vertex (its sign always inverts).
        let new_gain_i = -self.gain;
        solution.update_positive_gain_membership(self.i, new_gain_i);
        solution.gain[self.i] = new_gain_i;

        // Update neighbour gains. After `self.cut[self.i]` has been flipped,
        // `bi` still holds the pre-flip side, so `bi ^ bj` reflects whether the
        // edge was crossing before the flip (and is now not crossing, hence
        // `+2w`), and vice versa.
        for &(j, w) in prob.iter_on_adjacency(self.i) {
            let bj = solution.cut[j];
            let delta = if bi ^ bj { w * 2.0 } else { -w * 2.0 };
            let new_g = solution.gain[j] + delta;
            solution.update_positive_gain_membership(j, new_g);
            solution.gain[j] = new_g;
        }

        // Update the objective value
        solution.objective += self.gain;

        Ok(())
    }

    fn iter(prob: &MaxCut, sol: &MaxCutSolution) -> impl Iterator<Item = Self> + Send {
        prob.iter_on_vertices().map(|&i| MaxCutFlipNeighbor {
            i,
            gain: sol.gain[i],
        })
    }

    fn move_to_be_better_than(
        &self,
        _: &MaxCut,
        src: &MaxCutSolution,
        other: &MaxCutSolution,
    ) -> bool {
        self.gain + src.objective > other.objective
    }
}

impl Evaluate for MaxCutFlipNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Maximize(self.gain as f64)
    }
}

impl MaxCutFlipNeighbor {
    /// Generates a random flip neighbor by randomly selecting a vertex `i` and using its current gain.
    pub fn random_neighbor(prob: &MaxCut, sol: &MaxCutSolution) -> Self {
        let i = prob.vertices[rand::random_range(0..prob.vertices.len())];
        Self {
            i,
            gain: sol.gain[i],
        }
    }
}

/// A swap move that simultaneously flips vertices `i` and `j` to opposite sides.
///
/// Only pairs where `i` and `j` are currently on different sides are generated.
/// `gain` is the combined change in cut weight (positive = improvement).
#[derive(Debug, Clone)]
pub struct MaxCutSwapNeighbor {
    pub i: usize,
    pub j: usize,
    pub gain: f32,
}

impl Rankable for MaxCutSwapNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain > other.gain
    }
}

impl Evaluate for MaxCutSwapNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Maximize(self.gain as f64)
    }
}

impl EnabledTabu for MaxCutSwapNeighbor {
    type TabuMap = std::collections::HashMap<usize, u64>;

    /// A swap move is tabu if either vertex `i` or `j` is in the tabu map with a tenure
    /// greater than the current iteration.
    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        let enabled_i = tabu_map
            .get(&self.i)
            .map_or(true, |&tabu_tenure| iteration > tabu_tenure);
        let enabled_j = tabu_map
            .get(&self.j)
            .map_or(true, |&tabu_tenure| iteration > tabu_tenure);
        enabled_i && enabled_j
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
    ) {
        let tabu_duration = rand::random_range(tabu_tenure.0..=tabu_tenure.1);
        tabu_map.insert(self.i, iteration + tabu_duration);

        let tabu_duration = rand::random_range(tabu_tenure.0..=tabu_tenure.1);
        tabu_map.insert(self.j, iteration + tabu_duration);
    }
}

impl MoveToNeighbor<MaxCut> for MaxCutSwapNeighbor {
    fn apply_to_iteration(&self, iter: u64) -> u64 {
        iter + 2
    }

    fn apply_to_solution(&self, prob: &MaxCut, sol: &mut MaxCutSolution) -> Result<(), OptError> {
        let flip_i = MaxCutFlipNeighbor {
            i: self.i,
            gain: sol.gain[self.i],
        };
        flip_i.apply_to_solution(prob, sol)?;
        let flip_j = MaxCutFlipNeighbor {
            i: self.j,
            gain: sol.gain[self.j],
        };
        flip_j.apply_to_solution(prob, sol)?;
        Ok(())
    }

    fn iter(prob: &MaxCut, sol: &MaxCutSolution) -> impl Iterator<Item = Self> + Send {
        prob.iter_on_vertices().flat_map(move |&i| {
            prob.iter_on_vertices()
                .filter(move |&&j| j < i && (sol.cut[i] ^ sol.cut[j]))
                .map(move |&j| Self {
                    i,
                    j,
                    gain: sol.gain[i]
                        + sol.gain[j]
                        + if prob.has_edge(i, j) {
                            2.0 * prob.get_weight(i, j)
                        } else {
                            0.0
                        },
                })
        })
    }

    fn move_to_be_better_than(
        &self,
        _: &MaxCut,
        src: &MaxCutSolution,
        other: &MaxCutSolution,
    ) -> bool {
        self.gain + src.objective > other.objective
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::max_cut::MaxCut;
    use crate::search_state::SearchState;

    #[test]
    fn test_new() {
        let mut mc = MaxCut::new();
        mc.add_weight(0, 1, 1.0);
        mc.add_weight(0, 2, 1.0);
        mc.add_weight(1, 2, 1.0);

        let _ = SearchState::new(&mc);
    }

    #[test]
    fn test_flip_neighbor() {
        let mut mc = MaxCut::new();
        mc.add_weight(0, 1, 1.0);
        mc.add_weight(0, 2, 1.0);
        mc.add_weight(1, 2, 1.0);

        let mut state = SearchState::new(&mc);
        state.solution.cut[0] = true;
        state.solution.cut[1] = false;
        state.solution.cut[2] = true;

        let neighbor = MaxCutFlipNeighbor { i: 1, gain: -2.0 };
        state.apply(&neighbor).unwrap();

        assert_eq!(state.solution.cut[1], true);
    }
}
