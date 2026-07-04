//! Neighborhood move types for the [`MaxCut`] problem.
//!
//! Two move types are provided:
//!
//! - [`MaxCutFlipNeighbor`] — flip a single vertex (O(degree) update)
//! - [`MaxCutSwapNeighbor`] — swap two vertices on opposite sides (two sequential flips)
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

use super::{MaxCut, MaxCutSolution};
use crate::{
    common::{VarTabuMap, add_var_to_tabu, is_var_enabled},
    error::OptError,
    search_state::{EnabledTabu, Evaluable, Evaluate, MoveToNeighbor, Rankable},
};
use rand::Rng;

/// A flip move that transfers vertex `i` to the opposite partition side.
///
/// This is the most common move type for MaxCut. The neighborhood size is O(n)
/// and each move application takes O(degree(i)) time.
///
/// `gain` holds the change in cut weight after the flip (positive = improvement).
///
/// # Usage
///
/// ```
/// use optopus::prelude::*;
///
/// let mc = MaxCut::from_edges([(0, 1, 1.0), (1, 2, 1.0)]);
/// let mut state = SearchState::new(&mc);
///
/// // Use with any heuristic:
/// LocalSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(1000))
///     .run(&mut state).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MaxCutFlipNeighbor {
    /// Index of the vertex to flip.
    pub i: usize,
    /// Change in cut weight after the flip (positive = improvement).
    pub gain: f32,
}
impl Rankable for MaxCutFlipNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain > other.gain
    }
}

impl EnabledTabu for MaxCutFlipNeighbor {
    type TabuMap = VarTabuMap;

    /// A flip move is tabu if the vertex `i` is in the tabu map with a tenure greater than the current iteration.
    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        is_var_enabled(tabu_map, self.i, iteration)
    }

    /// When a flip move is applied,
    /// the vertex `i` is added to the tabu map with a tenure
    /// randomly chosen between `tabu_tenure.0` and `tabu_tenure.1`.
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

impl MoveToNeighbor<MaxCut> for MaxCutFlipNeighbor {
    /// Applies the flip move: transfers vertex `self.i` to the opposite partition side.
    ///
    /// Updates the solution in-place in O(degree(i)):
    /// 1. Flips `solution.x[i]`
    /// 2. Inverts `solution.gain[i]`
    /// 3. Updates `gain[j]` for each neighbor `j` of `i`
    /// 4. Adds `self.gain` to `solution.objective`
    ///
    /// If the `positive_gain` index is enabled, it is maintained incrementally.
    fn apply_to_solution(
        &self,
        prob: &MaxCut,
        solution: &mut MaxCutSolution,
    ) -> Result<(), OptError> {
        let bi = solution.x[self.i];

        // Flip
        solution.x[self.i] = !bi;

        // Update the gain for the flipped vertex (its sign always inverts).
        let new_gain_i = -self.gain;
        solution.update_positive_gain_membership(self.i, new_gain_i);
        solution.gain[self.i] = new_gain_i;

        // Update neighbor gains. After `self.x[self.i]` has been flipped,
        // `bi` still holds the pre-flip side, so `bi ^ bj` reflects whether the
        // edge was crossing before the flip (and is now not crossing, hence
        // `+2w`), and vice versa.
        for &(j, w) in prob.graph.iter_on_adjacency(self.i) {
            let bj = solution.x[j];
            let delta = if bi ^ bj { w * 2.0 } else { -w * 2.0 };
            let new_g = solution.gain[j] + delta;
            solution.update_positive_gain_membership(j, new_g);
            solution.gain[j] = new_g;
        }

        // Update the objective value
        solution.objective += self.gain;

        Ok(())
    }

    /// Returns a lazy iterator over all possible flip moves (one per vertex).
    ///
    /// The iterator yields `n` moves where `n` is the number of vertices with edges.
    fn iter(prob: &MaxCut, sol: &MaxCutSolution) -> impl Iterator<Item = Self> + Send {
        prob.graph.iter_on_vertices().map(|&i| MaxCutFlipNeighbor {
            i,
            gain: sol.gain[i],
        })
    }

    /// Returns `true` if applying this move to `src` would produce a solution
    /// better than `other`.
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
    /// Returns the gain as `Evaluable::Maximize`, since MaxCut is a maximization problem.
    ///
    /// This is used by [`SimulatedAnnealing`](crate::heuristic::SimulatedAnnealing) and
    /// [`LateAcceptanceHillClimbing`](crate::heuristic::LateAcceptanceHillClimbing)
    /// for acceptance decisions.
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Maximize(self.gain as f64)
    }
}

impl MaxCutFlipNeighbor {
    /// Generates a random flip neighbor by uniformly selecting a vertex from the graph.
    ///
    /// Useful as a perturbation step (e.g., in [`RandomWalk`](crate::heuristic::RandomWalk)).
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// let mc = MaxCut::from_edges([(0, 1, 1.0), (1, 2, 1.0)]);
    /// let mut state = SearchState::new(&mc);
    /// let solution = state.solution.clone();
    /// let flip = MaxCutFlipNeighbor::random_neighbor(&mc, &solution, &mut state.rng);
    /// println!("random flip: vertex {}, gain {}", flip.i, flip.gain);
    /// ```
    pub fn random_neighbor(
        prob: &MaxCut,
        sol: &MaxCutSolution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Self {
        let i = prob.graph.vertices[rng.random_range(0..prob.graph.vertices.len())];
        Self {
            i,
            gain: sol.gain[i],
        }
    }
}

/// A swap move that simultaneously flips vertices `i` and `j` to opposite sides.
///
/// Only pairs where `i` and `j` are currently on different sides are generated.
/// Each swap counts as **2 iterations** (see [`apply_to_iteration`](MoveToNeighbor::apply_to_iteration)).
/// The neighborhood size is O(n^2), so it is slower per iteration than [`MaxCutFlipNeighbor`]
/// but can escape local optima that flips cannot.
///
/// `gain` is the combined change in cut weight (positive = improvement).
///
/// # Usage
///
/// ```
/// use optopus::prelude::*;
///
/// let mc = MaxCut::from_edges([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]);
/// let mut state = SearchState::new(&mc);
///
/// TabuSearch::<MaxCutSwapNeighbor>::new(
///     StopCondition::iterations(10_000),
///     (5, 10),
///     None,
/// ).run(&mut state).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MaxCutSwapNeighbor {
    /// First vertex to swap (currently on one side).
    pub i: usize,
    /// Second vertex to swap (currently on the opposite side from `i`).
    pub j: usize,
    /// Combined change in cut weight (positive = improvement).
    pub gain: f32,
}

impl Rankable for MaxCutSwapNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain > other.gain
    }
}

impl Evaluate for MaxCutSwapNeighbor {
    /// Returns the combined gain as `Evaluable::Maximize`.
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Maximize(self.gain as f64)
    }
}

impl EnabledTabu for MaxCutSwapNeighbor {
    type TabuMap = VarTabuMap;

    /// A swap move is tabu if either vertex `i` or `j` is in the tabu map with a tenure
    /// greater than the current iteration.
    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        is_var_enabled(tabu_map, self.i, iteration) && is_var_enabled(tabu_map, self.j, iteration)
    }

    /// Adds both vertices `i` and `j` to the tabu map, each with an independently
    /// randomised tenure from `tabu_tenure`.
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

impl MoveToNeighbor<MaxCut> for MaxCutSwapNeighbor {
    /// A swap counts as 2 iterations (one for each vertex flip).
    fn apply_to_iteration(&self, iter: u64) -> u64 {
        iter + 2
    }

    /// Applies the swap by performing two sequential flips: first `i`, then `j`.
    ///
    /// The second flip uses the updated gain after the first flip, so the combined
    /// effect accounts for the interaction between the two vertices.
    fn apply_to_solution(&self, prob: &MaxCut, sol: &mut MaxCutSolution) -> Result<(), OptError> {
        crate::common::apply_swap_as_two_flips(prob, sol, self.i, self.j)
    }

    /// Returns a lazy iterator over all valid swap pairs `(i, j)` where
    /// `i` and `j` are on different sides.
    ///
    /// The gain is computed as `gain[i] + gain[j] + 2*w(i,j)` to account for
    /// the interaction when both vertices are flipped simultaneously.
    fn iter(prob: &MaxCut, sol: &MaxCutSolution) -> impl Iterator<Item = Self> + Send {
        prob.graph.iter_on_vertices().flat_map(move |&i| {
            prob.graph
                .iter_on_vertices()
                .filter(move |&&j| j < i && (sol.x[i] ^ sol.x[j]))
                .map(move |&j| Self {
                    i,
                    j,
                    gain: sol.gain[i]
                        + sol.gain[j]
                        + if prob.graph.has_edge(i, j) {
                            2.0 * prob.graph.get_weight(i, j)
                        } else {
                            0.0
                        },
                })
        })
    }

    /// Returns `true` if applying this swap to `src` would produce a solution
    /// better than `other`.
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
        let mc = MaxCut::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]);
        let _ = SearchState::new(&mc);
    }

    #[test]
    fn test_flip_neighbor() {
        let mc = MaxCut::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]);

        let mut state = SearchState::new(&mc);
        state.solution.x[0] = true;
        state.solution.x[1] = false;
        state.solution.x[2] = true;

        let neighbor = MaxCutFlipNeighbor { i: 1, gain: -2.0 };
        state.apply(&neighbor).unwrap();

        assert!(state.solution.x[1]);
    }
}
