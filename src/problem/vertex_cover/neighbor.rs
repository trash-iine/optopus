//! Neighborhood move types for the [`VertexCover`] problem.

use super::VertexCover;
use crate::{
    error::OptError,
    problem::vertex_cover::problem::VertexCoverSolution,
    search_state::{EnabledTabu, Evaluable, Evaluate, MoveToNeighbor, Rankable},
};
use rand::Rng;

/// A flip move that toggles whether vertex `i` is in the cover.
///
/// `gain` holds the change in (penalty-augmented) objective after the flip
/// (negative = improvement, since Vertex Cover is minimization).
#[derive(Debug, Clone, Copy)]
pub struct VertexCoverFlipNeighbor {
    /// Index of the vertex to flip.
    pub i: usize,
    /// Change in objective after the flip.
    pub gain: i32,
}

impl Rankable for VertexCoverFlipNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain < other.gain
    }
}

impl Evaluate for VertexCoverFlipNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain as f64)
    }
}

impl EnabledTabu for VertexCoverFlipNeighbor {
    /// Vec indexed by vertex ID. Value = expiry iteration (0 = never tabu).
    type TabuMap = Vec<u64>;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        tabu_map
            .get(self.i)
            .is_none_or(|&tabu_tenure| iteration > tabu_tenure)
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
        rng: &mut rand::rngs::SmallRng,
    ) {
        let tabu_duration = rng.random_range(tabu_tenure.0..=tabu_tenure.1);
        if self.i >= tabu_map.len() {
            tabu_map.resize(self.i + 1, 0);
        }
        tabu_map[self.i] = iteration + tabu_duration;
    }
}

impl MoveToNeighbor<VertexCover> for VertexCoverFlipNeighbor {
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
        // For each neighbour `j`:
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
            .map(|&i| VertexCoverFlipNeighbor {
                i,
                gain: sol.gain[i],
            })
    }

    fn move_to_be_better_than(
        &self,
        _: &VertexCover,
        src: &VertexCoverSolution,
        other: &VertexCoverSolution,
    ) -> bool {
        self.gain + src.objective < other.objective
    }
}

/// A swap move that simultaneously flips an in-cover vertex `i` and an out-of-cover vertex `j`
/// (or vice versa), so the cover size is unchanged but coverage may improve.
///
/// Only pairs `(i, j)` with `cover[i] != cover[j]` and `i < j` are enumerated.
#[derive(Debug, Clone, Copy)]
pub struct VertexCoverSwapNeighbor {
    pub i: usize,
    pub j: usize,
    pub gain: i32,
}

impl Rankable for VertexCoverSwapNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain < other.gain
    }
}

impl Evaluate for VertexCoverSwapNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain as f64)
    }
}

impl EnabledTabu for VertexCoverSwapNeighbor {
    type TabuMap = Vec<u64>;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        let enabled_i = tabu_map
            .get(self.i)
            .is_none_or(|&tenure| iteration > tenure);
        let enabled_j = tabu_map
            .get(self.j)
            .is_none_or(|&tenure| iteration > tenure);
        enabled_i && enabled_j
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
        rng: &mut rand::rngs::SmallRng,
    ) {
        let max_v = self.i.max(self.j);
        if max_v >= tabu_map.len() {
            tabu_map.resize(max_v + 1, 0);
        }
        let d = rng.random_range(tabu_tenure.0..=tabu_tenure.1);
        tabu_map[self.i] = iteration + d;
        let d = rng.random_range(tabu_tenure.0..=tabu_tenure.1);
        tabu_map[self.j] = iteration + d;
    }
}

impl MoveToNeighbor<VertexCover> for VertexCoverSwapNeighbor {
    fn apply_to_iteration(&self, iter: u64) -> u64 {
        iter + 2
    }

    fn apply_to_solution(
        &self,
        prob: &VertexCover,
        sol: &mut VertexCoverSolution,
    ) -> Result<(), OptError> {
        crate::common::apply_swap_as_two_flips(prob, sol, self.i, self.j)
    }

    fn iter(prob: &VertexCover, sol: &VertexCoverSolution) -> impl Iterator<Item = Self> + Send {
        let pw = prob.penalty_weight();
        prob.graph.iter_on_vertices().flat_map(move |&i| {
            prob.graph
                .iter_on_vertices()
                .filter(move |&&j| j < i && (sol.x[i] != sol.x[j]))
                .map(move |&j| {
                    // Combined Δobjective for flipping i then j (cover_size unchanged):
                    //   gain[i] + gain[j] - pw if (i, j) is an edge, else gain[i] + gain[j].
                    // The -pw term cancels the double-counted toggle of edge (i, j).
                    let edge_correction = if prob.graph.has_edge(i, j) { pw } else { 0 };
                    Self {
                        i,
                        j,
                        gain: sol.gain[i] + sol.gain[j] - edge_correction,
                    }
                })
        })
    }

    fn move_to_be_better_than(
        &self,
        _: &VertexCover,
        src: &VertexCoverSolution,
        other: &VertexCoverSolution,
    ) -> bool {
        self.gain + src.objective < other.objective
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_state::SearchState;

    fn make_triangle() -> VertexCover {
        let mut g = crate::common::Graph::new();
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
        state.solution = vc.solution_from_assignment(&[false, false, false]);
        state.best_solution = state.solution.clone();

        let n = VertexCoverFlipNeighbor {
            i: 0,
            gain: state.solution.gain[0],
        };
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
        state.solution = vc.solution_from_assignment(&[true, false, false]);
        state.best_solution = state.solution.clone();

        // Swap vertex 0 (in cover) with vertex 1 (out of cover).
        // Both cover sizes are 1; uncovered edges before = {(1,2)}; after = {(0,2)}.
        // Δobjective = 0.
        let swap = VertexCoverSwapNeighbor {
            i: 1,
            j: 0,
            gain: state.solution.gain[1] + state.solution.gain[0] - vc.penalty_weight(),
        };
        let predicted_gain = swap.gain;
        state.apply(&swap).unwrap();

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
        let sol = vc.solution_from_assignment(&[true, false, false]);
        let pairs: Vec<_> = VertexCoverSwapNeighbor::iter(&vc, &sol).collect();
        // Mixed (i, j) with j < i: (1, 0) and (2, 0). No (2, 1) since both are out.
        assert_eq!(pairs.len(), 2);
        for p in &pairs {
            assert_ne!(sol.x[p.i], sol.x[p.j]);
            assert!(p.j < p.i);
        }
    }
}
