//! Neighborhood move types for the [`MaxCut`] problem.

use super::MaxCut;
use crate::{
    error::OptError,
    problem::max_cut::problem::MaxCutSolution,
    search_state::{EnabledTabu, Evaluate, Evaluable, MoveToNeighbor, Rankable},
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
        // cut side of the vertex
        let bi = *solution.cut.get(&self.i).ok_or_else(|| {
            OptError::InvalidState(format!("vertex {} is not found in solution.", self.i))
        })?;

        // Flip
        solution.cut.insert(self.i, !bi);

        // Update the gain for the flipped vertex
        solution.gain.insert(self.i, -self.gain);
        for (&j, &w) in prob.iter_on_adjacency(&self.i) {
            let bj = *solution.cut.get(&j).ok_or_else(|| {
                OptError::InvalidState(format!("vertex {} is not found in the solution.", j))
            })?;
            if bi ^ bj {
                *solution.gain.entry(j).or_insert(0.0) += w * 2.0;
            } else {
                *solution.gain.entry(j).or_insert(0.0) -= w * 2.0;
            }
        }

        // Update the objective value
        solution.objective += self.gain;

        Ok(())
    }

    fn iter(_: &MaxCut, sol: &MaxCutSolution) -> impl Iterator<Item = Self> + Send {
        sol.cut.keys().map(move |&i| MaxCutFlipNeighbor {
            i,
            gain: *sol
                .gain
                .get(&i)
                .expect("gain entry must exist for every vertex in cut"),
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
            gain: sol.gain[&self.i],
        };
        flip_i.apply_to_solution(prob, sol)?;
        let flip_j = MaxCutFlipNeighbor {
            i: self.j,
            gain: sol.gain[&self.j],
        };
        flip_j.apply_to_solution(prob, sol)?;
        Ok(())
    }

    fn iter(prob: &MaxCut, sol: &MaxCutSolution) -> impl Iterator<Item = Self> + Send {
        prob.iter_on_vertices().flat_map(move |&i| {
            prob.iter_on_vertices()
                .filter(move |&&j| j < i && (sol.cut[&i] ^ sol.cut[&j]))
                .map(move |&j| Self {
                    i,
                    j,
                    gain: sol.gain[&i]
                        + sol.gain[&j]
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
        state.solution.cut.insert(0, true);
        state.solution.cut.insert(1, false);
        state.solution.cut.insert(2, true);

        let neighbor = MaxCutFlipNeighbor { i: 1, gain: -2.0 };
        state.apply(&neighbor).unwrap();

        assert_eq!(state.solution.cut[&1], true);
    }
}
