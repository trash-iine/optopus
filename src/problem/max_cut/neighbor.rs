//! Neighborhood move types for the [`MaxCut`] problem.
//!
//! Two move types are provided:
//!
//! - [`MaxCutFlipNeighbor`], flip a single vertex (O(degree) update)
//! - [`MaxCutSwapNeighbor`], swap two vertices on opposite sides (two sequential flips)
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
    common::{
        TabuMemory,
        binary::{differing_pairs, random_differing_pair},
    },
    error::OptError,
    search_state::{EnabledTabu, Evaluable, Evaluate, MoveToNeighbor},
};
use rand::Rng;
use rand::rngs::SmallRng;

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
impl EnabledTabu for MaxCutFlipNeighbor {
    /// The move is tabu while vertex `i` is still blocked at the current iteration.
    fn is_move_enabled(&self, tabu: &TabuMemory, iteration: u64) -> bool {
        tabu.is_enabled(self.i, iteration)
    }

    /// Applying the move forbids vertex `i` for a tenure the memory draws.
    fn add_to_tabu_map(&self, tabu: &mut TabuMemory, iteration: u64, rng: &mut SmallRng) {
        tabu.forbid(self.i, iteration, rng);
    }
}

impl MoveToNeighbor<MaxCut> for MaxCutFlipNeighbor {
    /// Hands this move's [`EnabledTabu`] policy to the search state, which is
    /// what holds the tabu map.
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

    /// Applies the flip move: transfers vertex `self.i` to the opposite partition side.
    ///
    /// Updates the solution in-place in O(degree(i)):
    /// 1. Flips `solution.x[i]`
    /// 2. Inverts `solution.gain[i]`
    /// 3. Updates `gain[j]` for each neighbor `j` of `i`
    /// 4. Adds `self.gain` to `solution.objective`
    ///
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
        solution.gain[self.i] = new_gain_i;

        // Update neighbor gains. After `self.x[self.i]` has been flipped,
        // `bi` still holds the pre-flip side, so `bi ^ bj` reflects whether the
        // edge was crossing before the flip (and is now not crossing, hence
        // `+2w`), and vice versa.
        for &(j, w) in prob.graph.iter_on_adjacency(self.i) {
            let bj = solution.x[j];
            let delta = if bi ^ bj { w * 2.0 } else { -w * 2.0 };
            let new_g = solution.gain[j] + delta;
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
        prob.graph
            .iter_on_vertices()
            .map(|&i| MaxCutFlipNeighbor::new(prob, sol, i))
    }

    /// Returns `true` if applying this move to `src` would produce a solution
    /// better than `other`.
    fn move_to_be_better_than(
        &self,
        _: &MaxCut,
        src: &MaxCutSolution,
        other: &MaxCutSolution,
    ) -> bool {
        self.evaluate()
            .improves_over(src.evaluate(), other.evaluate())
    }

    /// O(1): picks a uniformly random vertex.
    fn random_neighbor(
        prob: &MaxCut,
        sol: &MaxCutSolution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Option<Self> {
        if prob.graph.vertices.is_empty() {
            return None;
        }
        let i = prob.graph.vertices[rng.random_range(0..prob.graph.vertices.len())];
        Some(Self::new(prob, sol, i))
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
    /// Builds the flip of vertex `i`, reading its cached gain.
    ///
    /// A flip's gain needs no correction. It is exactly the value the solution
    /// already maintains, so this is [`BinaryProblem::flip_move`](crate::trait_defs::BinaryProblem::flip_move)
    /// spelled like the other constructors, keeping every construction site
    /// on one path, the way [`MaxCutSwapNeighbor::new`] does. `prob` is unused
    /// for that reason and taken only so the two constructors read alike at the
    /// call site.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (1, 2, 1.0)]));
    /// let sol = MaxCutSolution::new_from_assignment(&mc, vec![false, true, false]);
    /// let flip = MaxCutFlipNeighbor::new(&mc, &sol, 1);
    /// assert_eq!(flip.gain, sol.gain[1]);
    /// ```
    pub fn new(_prob: &MaxCut, sol: &MaxCutSolution, i: usize) -> Self {
        <MaxCut as crate::trait_defs::BinaryProblem>::flip_move(sol, i)
    }
}

/// A swap move that simultaneously flips vertices `i` and `j` to opposite sides.
///
/// Only pairs where `i` and `j` are currently on different sides are generated.
/// Each swap counts as 2 iterations (see [`apply_to_iteration`](MoveToNeighbor::apply_to_iteration)).
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

impl MaxCutSwapNeighbor {
    /// Builds the swap of `i` and `j`, computing the combined gain.
    ///
    /// The gain is `gain[i] + gain[j] + 2·w(i, j)`: the two flip gains plus a
    /// correction for the edge between the vertices, which each flip alone
    /// would count with the wrong sign. Every construction site goes through
    /// here so the correction cannot be forgotten at one of them.
    ///
    /// `i` and `j` are expected to sit on opposite sides, which is what makes
    /// the move a swap, but nothing here depends on it, so a caller that
    /// deliberately builds a same-side pair still gets a correctly evaluated
    /// move.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (1, 2, 1.0)]));
    /// let sol = MaxCutSolution::new_from_assignment(&mc, vec![false, true, false]);
    /// let swap = MaxCutSwapNeighbor::new(&mc, &sol, 1, 0);
    /// assert_eq!(swap.gain, sol.gain[1] + sol.gain[0] + 2.0 * mc.graph.get_weight(1, 0));
    /// ```
    /// Non-adjacent pairs need no correction, and
    /// [`get_weight`](crate::common::Graph::get_weight) already returns `0.0`
    /// for them, so this does not pay for a separate `has_edge` lookup, the
    /// perturbation operators call it once per move.
    pub fn new(prob: &MaxCut, sol: &MaxCutSolution, i: usize, j: usize) -> Self {
        Self {
            i,
            j,
            gain: sol.gain[i] + sol.gain[j] + 2.0 * prob.graph.get_weight(i, j),
        }
    }
}

impl Evaluate for MaxCutSwapNeighbor {
    /// Returns the combined gain as `Evaluable::Maximize`.
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Maximize(self.gain as f64)
    }
}

impl EnabledTabu for MaxCutSwapNeighbor {
    /// A swap is tabu unless both vertexs it moves are free.
    fn is_move_enabled(&self, tabu: &TabuMemory, iteration: u64) -> bool {
        tabu.is_enabled(self.i, iteration) && tabu.is_enabled(self.j, iteration)
    }

    /// Applying the swap forbids both vertexs, each for its own drawn tenure.
    fn add_to_tabu_map(&self, tabu: &mut TabuMemory, iteration: u64, rng: &mut SmallRng) {
        tabu.forbid(self.i, iteration, rng);
        tabu.forbid(self.j, iteration, rng);
    }
}

impl MoveToNeighbor<MaxCut> for MaxCutSwapNeighbor {
    /// Hands this move's [`EnabledTabu`] policy to the search state, which is
    /// what holds the tabu map.
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

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
        differing_pairs(prob, sol).map(move |(i, j)| Self::new(prob, sol, i, j))
    }

    /// Returns `true` if applying this swap to `src` would produce a solution
    /// better than `other`.
    fn move_to_be_better_than(
        &self,
        _: &MaxCut,
        src: &MaxCutSolution,
        other: &MaxCutSolution,
    ) -> bool {
        self.evaluate()
            .improves_over(src.evaluate(), other.evaluate())
    }

    /// O(n): one vertex from each side, uniformly over all cross-side pairs.
    fn random_neighbor(
        prob: &MaxCut,
        sol: &MaxCutSolution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Option<Self> {
        random_differing_pair(prob, sol, rng).map(|(i, j)| Self::new(prob, sol, i, j))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::max_cut::{MaxCut, MaxCutSolution};
    use crate::search_state::SearchState;

    /// A weighted instance with an odd cycle and a pendant, so no two vertices
    /// have the same neighborhood and a gain update that touches the wrong
    /// neighbor cannot cancel out.
    fn weighted_instance() -> MaxCut {
        MaxCut::from_edges([
            (0, 1, 1.0),
            (0, 2, 2.0),
            (1, 2, -3.0),
            (1, 3, 4.0),
            (2, 3, 0.5),
            (3, 4, -1.5),
        ])
    }

    /// `apply_to_solution` updates the objective and every touched gain in
    /// place rather than re-deriving them, and a wrong sign or a missed
    /// neighbor there does not show up in the solution's assignment, only in
    /// the caches every heuristic then reads. So the caches are compared
    /// against a full recompute after each move, for flips and swaps both.
    #[test]
    fn apply_leaves_the_objective_and_every_gain_exact() {
        let mc = weighted_instance();
        let sol = MaxCutSolution::new_from_assignment(&mc, vec![true, false, true, false, true]);

        let check = |s: &MaxCutSolution, what: &str| {
            assert_eq!(
                s.objective,
                mc.calculate_cut_size(&s.x),
                "{what}: objective drifted"
            );
            for &i in mc.graph.iter_on_vertices() {
                assert_eq!(
                    s.gain[i],
                    mc.calculate_gain(&s.x, i),
                    "{what}: gain[{i}] drifted"
                );
            }
        };

        check(&sol, "the starting solution");

        for m in MaxCutFlipNeighbor::iter(&mc, &sol) {
            let mut s = sol.clone();
            m.apply_to_solution(&mc, &mut s).unwrap();
            assert_eq!(s.x[m.i], !sol.x[m.i], "flip {} did not move", m.i);
            check(&s, &format!("flip {}", m.i));
        }

        for m in MaxCutSwapNeighbor::iter(&mc, &sol) {
            let mut s = sol.clone();
            m.apply_to_solution(&mc, &mut s).unwrap();
            assert_eq!(
                s.x[m.i], !sol.x[m.i],
                "swap ({}, {}) did not move i",
                m.i, m.j
            );
            assert_eq!(
                s.x[m.j], !sol.x[m.j],
                "swap ({}, {}) did not move j",
                m.i, m.j
            );
            check(&s, &format!("swap ({}, {})", m.i, m.j));
        }
    }

    /// The gain a move advertises is what every heuristic ranks it by, so it
    /// has to be the objective difference the move actually makes.
    #[test]
    fn an_advertised_gain_is_the_objective_difference_it_makes() {
        let mc = weighted_instance();
        let sol = MaxCutSolution::new_from_assignment(&mc, vec![true, false, true, false, true]);

        for m in MaxCutFlipNeighbor::iter(&mc, &sol) {
            let mut moved = sol.x.clone();
            moved[m.i] = !moved[m.i];
            assert_eq!(
                m.gain,
                mc.calculate_cut_size(&moved) - sol.objective,
                "flip {}",
                m.i
            );
        }

        for m in MaxCutSwapNeighbor::iter(&mc, &sol) {
            let mut moved = sol.x.clone();
            moved[m.i] = !moved[m.i];
            moved[m.j] = !moved[m.j];
            assert_eq!(
                m.gain,
                mc.calculate_cut_size(&moved) - sol.objective,
                "swap ({}, {})",
                m.i,
                m.j
            );
        }
    }

    #[test]
    fn test_random_neighbor_samples_member_of_iter() {
        use rand::SeedableRng;
        let mc = MaxCut::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]);
        let mut state = SearchState::new_with_seed(&mc, 42);
        state.solution.x = vec![true, false, true];
        let sol = state.solution.clone();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);

        let flips: Vec<_> = MaxCutFlipNeighbor::iter(&mc, &sol).collect();
        for _ in 0..20 {
            let m = <MaxCutFlipNeighbor as MoveToNeighbor<MaxCut>>::random_neighbor(
                &mc, &sol, &mut rng,
            )
            .unwrap();
            assert!(flips.iter().any(|f| f.i == m.i && f.gain == m.gain));
        }

        let swaps: Vec<_> = MaxCutSwapNeighbor::iter(&mc, &sol).collect();
        for _ in 0..20 {
            let m = <MaxCutSwapNeighbor as MoveToNeighbor<MaxCut>>::random_neighbor(
                &mc, &sol, &mut rng,
            )
            .unwrap();
            assert!(
                swaps
                    .iter()
                    .any(|s| s.i == m.i && s.j == m.j && s.gain == m.gain)
            );
        }
    }

    #[test]
    fn test_random_swap_neighbor_none_when_one_sided() {
        use rand::SeedableRng;
        let mc = MaxCut::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]);
        let mut state = SearchState::new(&mc);
        state.solution.x = vec![false, false, false];
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
        assert!(
            <MaxCutSwapNeighbor as MoveToNeighbor<MaxCut>>::random_neighbor(
                &mc,
                &state.solution,
                &mut rng
            )
            .is_none()
        );
    }
}
