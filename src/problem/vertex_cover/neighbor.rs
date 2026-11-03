//! Neighborhood move types for the [`VertexCover`] problem.

use super::VertexCover;
use crate::trait_defs::{Rankable, better_after_move};
use crate::{
    building_blocks::representation::{differing_pairs_by_row, random_differing_pair},
    error::OptError,
    problem::vertex_cover::problem::VertexCoverSolution,
    search_state::{EnabledTabu, Evaluate, MoveToNeighbor},
};
use rand::Rng;

/// A flip move that toggles whether vertex `i` is in the cover.
///
/// It ranks as the solution does, by the change in uncovered edges first and
/// the change in cover size second. `gain` is the same change as one
/// penalized number (negative = improvement), the value
/// [`Evaluate`] reports for acceptance probabilities. Applying the move
/// refreshes the gain of every neighbor of `i` and the uncovered-edge count,
/// i.e. O(degree(i)) work.
#[derive(Debug, Clone, Copy, Rankable, Evaluate, EnabledTabu)]
#[tabu(dense(i))]
pub struct VertexCoverFlipNeighbor {
    /// Index of the vertex to flip.
    pub i: usize,
    /// Change in the number of uncovered edges.
    #[rank(min)]
    pub uncovered_delta: i32,
    /// Change in the cover size, `+1` or `-1`.
    #[rank(min)]
    pub cover_delta: i32,
    /// Change in the penalized objective,
    /// `cover_delta + penalty_weight * uncovered_delta`.
    #[objective(minimize)]
    pub gain: i32,
}

impl VertexCoverFlipNeighbor {
    /// Builds the flip of vertex `i`, reading its cached gain.
    ///
    /// A flip's gain needs no correction. It is exactly the value the solution
    /// already maintains, so this is [`BinaryProblem::flip_move`](crate::trait_defs::BinaryProblem::flip_move)
    /// spelled like the other constructors, keeping every construction site
    /// on one path, the way [`VertexCoverSwapNeighbor::new`] does.
    pub fn new(prob: &VertexCover, sol: &VertexCoverSolution, i: usize) -> Self {
        crate::trait_defs::BinaryProblem::flip_move(prob, sol, i)
    }
}

impl MoveToNeighbor<VertexCover> for VertexCoverFlipNeighbor {
    /// Hands this move's [`EnabledTabu`] policy to the search state, which is
    /// what holds the tabu map.
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

    fn apply_to_solution(
        &self,
        prob: &VertexCover,
        sol: &mut VertexCoverSolution,
    ) -> Result<(), OptError> {
        let was_in = sol.x[self.i];
        sol.x[self.i] = !was_in;

        // Self gain inverts on a flip (since flipping back exactly undoes the move).
        sol.gain[self.i] = -self.gain;

        let pw = prob.penalty_weight();

        if was_in {
            sol.cover_size -= 1;
        } else {
            sol.cover_size += 1;
        }

        // Update neighbor gains and edge coverage counters.
        // For each neighbor `j`:
        //   - If !cover[j]: edge (i, j) flips between covered ↔ uncovered.
        //   - gain[j] shifts by ±pw depending on (cover[j], was_in).
        for &(j, _w) in prob.graph.iter_on_adjacency(self.i) {
            let cj = sol.x[j];
            if !cj {
                if was_in {
                    sol.uncovered_edges += 1;
                } else {
                    sol.uncovered_edges -= 1;
                }
            }
            // Delta on gain[j]:
            //   cover[j] = true  → gain[j] = -1 + pw * out_count(j)
            //   cover[j] = false → gain[j] =  1 - pw * out_count(j)
            // Flipping i changes out_count(j) by +1 (was_in true → i is now out) or
            // -1 (was_in false → i is now in).
            let delta = match (cj, was_in) {
                (true, true) => pw,
                (true, false) => -pw,
                (false, true) => -pw,
                (false, false) => pw,
            };
            sol.gain[j] += delta;
        }

        sol.objective += self.gain;

        Ok(())
    }

    fn iter(prob: &VertexCover, sol: &VertexCoverSolution) -> impl Iterator<Item = Self> + Send {
        prob.graph
            .iter_on_vertices()
            .map(|&i| VertexCoverFlipNeighbor::new(prob, sol, i))
    }

    fn move_to_be_better_than(
        &self,
        _: &VertexCover,
        src: &VertexCoverSolution,
        other: &VertexCoverSolution,
    ) -> bool {
        better_after_move(self, src, other)
    }

    /// O(1): picks a uniformly random vertex.
    fn random_neighbor(
        prob: &VertexCover,
        sol: &VertexCoverSolution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Option<Self> {
        if prob.graph.vertices.is_empty() {
            return None;
        }
        let i = prob.graph.vertices[rng.random_range(0..prob.graph.vertices.len())];
        Some(Self::new(prob, sol, i))
    }
}

/// A swap move that simultaneously flips an in-cover vertex `i` and an out-of-cover vertex `j`
/// (or vice versa), so the cover size is unchanged but coverage may improve.
///
/// Only pairs `(i, j)` with `cover[i] != cover[j]` and `i < j` are enumerated.
/// Applying the move performs two flips in sequence (`apply_to_iteration`
/// returns `iter + 2`), each doing O(degree) work.
///
/// It ranks like the flip, by the change in uncovered edges and then in cover
/// size, which a swap leaves at `0`.
#[derive(Debug, Clone, Copy, Rankable, Evaluate, EnabledTabu)]
#[tabu(dense(i), dense(j))]
pub struct VertexCoverSwapNeighbor {
    pub i: usize,
    pub j: usize,
    /// Change in the number of uncovered edges.
    #[rank(min)]
    pub uncovered_delta: i32,
    /// Change in the cover size, always `0`, kept so the move has the
    /// solution's two keys.
    #[rank(min)]
    pub cover_delta: i32,
    /// Change in the penalized objective, `penalty_weight * uncovered_delta`.
    #[objective(minimize)]
    pub gain: i32,
}

impl VertexCoverSwapNeighbor {
    /// Builds the swap of `i` and `j`, computing the combined gain.
    ///
    /// The gain is `gain[i] + gain[j]`, minus one `penalty_weight` when `(i, j)`
    /// is an edge: flipping both ends toggles that edge's coverage twice, so
    /// each flip gain counts it once and one of the two has to be cancelled.
    /// Every construction site goes through here so the correction cannot be
    /// forgotten at one of them.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// let vc = VertexCover::new(Graph::from_edges([(0, 1, 1.0), (1, 2, 1.0)]));
    /// let sol = vc.solution_from_assignment(&[true, false, false])?;
    /// let swap = VertexCoverSwapNeighbor::new(&vc, &sol, 1, 0);
    /// assert_eq!(swap.gain, sol.gain[1] + sol.gain[0] - vc.penalty_weight());
    /// # Ok::<(), optopus::error::OptError>(())
    /// ```
    pub fn new(prob: &VertexCover, sol: &VertexCoverSolution, i: usize, j: usize) -> Self {
        Self::with_edge(prob, sol, i, j, prob.graph.has_edge(i, j))
    }

    /// The swap of `i` and `j`, adjacent or not, the one place the
    /// correction is written.
    fn with_edge(
        prob: &VertexCover,
        sol: &VertexCoverSolution,
        i: usize,
        j: usize,
        adjacent: bool,
    ) -> Self {
        let edge_correction = if adjacent { prob.penalty_weight() } else { 0 };
        let gain = sol.gain[i] + sol.gain[j] - edge_correction;
        Self {
            i,
            j,
            uncovered_delta: prob.uncovered_delta(gain, 0),
            cover_delta: 0,
            gain,
        }
    }
}

impl MoveToNeighbor<VertexCover> for VertexCoverSwapNeighbor {
    /// Hands this move's [`EnabledTabu`] policy to the search state, which is
    /// what holds the tabu map.
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

    fn apply_to_iteration(&self, iter: u64) -> u64 {
        iter + 2
    }

    fn apply_to_solution(
        &self,
        prob: &VertexCover,
        sol: &mut VertexCoverSolution,
    ) -> Result<(), OptError> {
        crate::building_blocks::representation::apply_swap_as_two_flips(prob, sol, self.i, self.j)
    }

    fn iter(prob: &VertexCover, sol: &VertexCoverSolution) -> impl Iterator<Item = Self> + Send {
        differing_pairs_by_row(prob, sol, |i| {
            prob.graph.neighbors(i).map(|&(j, _)| (j, true))
        })
        .map(move |(i, j, adjacent)| Self::with_edge(prob, sol, i, j, adjacent))
    }

    fn move_to_be_better_than(
        &self,
        _: &VertexCover,
        src: &VertexCoverSolution,
        other: &VertexCoverSolution,
    ) -> bool {
        better_after_move(self, src, other)
    }

    /// O(n): one vertex in the cover and one outside it, uniformly over all
    /// mixed pairs.
    fn random_neighbor(
        prob: &VertexCover,
        sol: &VertexCoverSolution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Option<Self> {
        random_differing_pair(prob, sol, rng).map(|(i, j)| Self::new(prob, sol, i, j))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_state::SearchState;

    fn make_triangle() -> VertexCover {
        let mut g = crate::building_blocks::instance::Graph::new();
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.add_edge(1, 2);
        VertexCover::new(g)
    }

    #[test]
    fn test_flip_neighbor_apply() {
        let vc = make_triangle();
        let mut state = SearchState::new(&vc);
        // Force solution to all-false to make assertions deterministic.
        state.solution = vc.solution_from_assignment(&[false, false, false]).unwrap();
        state.best_solution = state.solution.clone();

        let n = VertexCoverFlipNeighbor::new(&vc, &state.solution, 0);
        state.apply(&n).unwrap();

        // After inserting vertex 0: cover_size = 1, uncovered edges = 1 (only (1,2)).
        assert!(state.solution.x[0]);
        assert_eq!(state.solution.cover_size, 1);
        assert_eq!(state.solution.uncovered_edges, 1);

        // Verify against from-scratch recomputation.
        let (gain, obj, cs, ue) = vc.calculate_state(&state.solution.x);
        assert_eq!(state.solution.gain, gain);
        assert_eq!(state.solution.objective, obj);
        assert_eq!(state.solution.cover_size, cs);
        assert_eq!(state.solution.uncovered_edges, ue);
    }

    #[test]
    fn test_swap_neighbor_apply_and_gain() {
        let vc = make_triangle();
        let mut state = SearchState::new(&vc);
        state.solution = vc.solution_from_assignment(&[true, false, false]).unwrap();
        state.best_solution = state.solution.clone();

        // Swap vertex 0 (in cover) with vertex 1 (out of cover).
        // Both cover sizes are 1; uncovered edges before = {(1,2)}; after = {(0,2)}.
        // Δobjective = 0.
        let swap = VertexCoverSwapNeighbor::new(&vc, &state.solution, 1, 0);
        let predicted_gain = swap.gain;
        let uncovered_before = state.solution.uncovered_edges as i32;
        state.apply(&swap).unwrap();
        assert_eq!(swap.cover_delta, 0);
        assert_eq!(
            state.solution.uncovered_edges as i32 - uncovered_before,
            swap.uncovered_delta
        );

        let (gain, obj, cs, ue) = vc.calculate_state(&state.solution.x);
        assert_eq!(state.solution.gain, gain);
        assert_eq!(state.solution.objective, obj);
        assert_eq!(state.solution.cover_size, cs);
        assert_eq!(state.solution.uncovered_edges, ue);
        assert_eq!(predicted_gain, 0);
    }

    #[test]
    fn test_swap_iter_only_mixed_pairs() {
        let vc = make_triangle();
        let sol = vc.solution_from_assignment(&[true, false, false]).unwrap();
        let pairs: Vec<_> = VertexCoverSwapNeighbor::iter(&vc, &sol).collect();
        // Mixed (i, j) with j < i: (1, 0) and (2, 0). No (2, 1) since both are out.
        assert_eq!(pairs.len(), 2);
        for p in &pairs {
            assert_ne!(sol.x[p.i], sol.x[p.j]);
            assert!(p.j < p.i);
        }
    }

    #[test]
    fn test_random_neighbor_samples_member_of_iter() {
        use rand::SeedableRng;
        let vc = make_triangle();
        let sol = vc.solution_from_assignment(&[true, false, false]).unwrap();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);

        let flips: Vec<_> = VertexCoverFlipNeighbor::iter(&vc, &sol).collect();
        for _ in 0..20 {
            let m = <VertexCoverFlipNeighbor as MoveToNeighbor<VertexCover>>::random_neighbor(
                &vc, &sol, &mut rng,
            )
            .unwrap();
            assert!(flips.iter().any(|f| f.i == m.i && f.gain == m.gain));
        }

        let swaps: Vec<_> = VertexCoverSwapNeighbor::iter(&vc, &sol).collect();
        for _ in 0..20 {
            let m = <VertexCoverSwapNeighbor as MoveToNeighbor<VertexCover>>::random_neighbor(
                &vc, &sol, &mut rng,
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
    fn test_random_swap_neighbor_none_when_uniform() {
        use rand::SeedableRng;
        let vc = make_triangle();
        let sol = vc.solution_from_assignment(&[false, false, false]).unwrap();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
        assert!(
            <VertexCoverSwapNeighbor as MoveToNeighbor<VertexCover>>::random_neighbor(
                &vc, &sol, &mut rng
            )
            .is_none()
        );
    }
}
