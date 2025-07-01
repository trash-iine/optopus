// ! Defines the neighbor structure and the methods to enumerate neighbors for the MaxCut problem.

use super::MaxCut;
use crate::search_state::{EnumerateMoveToNeighbor, Evaluable, SearchState};

/// Represents a neighbor in the MaxCut problem where a vertex is flipped.
#[derive(Debug, Clone)]
pub struct MaxCutFlipNeighbor {
    i: usize,
    gain: f32,
}

impl std::hash::Hash for MaxCutFlipNeighbor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.i.hash(state);
    }
}

impl std::cmp::PartialEq for MaxCutFlipNeighbor {
    fn eq(&self, other: &Self) -> bool {
        self.i == other.i
    }
}
impl std::cmp::Eq for MaxCutFlipNeighbor {}

impl<'a> EnumerateMoveToNeighbor<MaxCutFlipNeighbor> for SearchState<'a, MaxCut> {
    fn apply_to_iteration(&mut self, _: &MaxCutFlipNeighbor) {
        self.iteration += 1;
    }
    fn apply_to_solution(&mut self, neighbor: &MaxCutFlipNeighbor) {
        // cut side of the vertex
        let bi = *self
            .solution
            .cut
            .get(&neighbor.i)
            .expect(format!("vertex {} is not found in solution.", neighbor.i).as_str());

        self.solution.cut.insert(neighbor.i, !bi);

        // Update the gain for the flipped vertex
        self.solution.gain.insert(neighbor.i, -neighbor.gain);
        for (&j, &w) in self.instance.iter_on_adjacency(&neighbor.i) {
            let bj = *self
                .solution
                .cut
                .get(&j)
                .expect(format!("vertex {} is not found in the solution.", j).as_str());
            if bi ^ bj {
                *self.solution.gain.entry(j).or_insert(0.0) += w * 2.0;
            } else {
                *self.solution.gain.entry(j).or_insert(0.0) -= w * 2.0;
            }
        }
    }

    fn apply_to_objective(&mut self, neighbor: &MaxCutFlipNeighbor) {
        self.objective += neighbor.gain;
    }

    fn iter_on_move_to_neighbor(&self) -> impl Iterator<Item = MaxCutFlipNeighbor> {
        (0..self.instance.len()).map(move |i| MaxCutFlipNeighbor {
            i,
            gain: *self
                .solution
                .gain
                .get(&i)
                .expect(format!("vertex {} is not found in the solution.", i).as_str()),
        })
    }

    fn is_move_to_be_better_than_currernt(&self, neighbor: &MaxCutFlipNeighbor) -> bool {
        neighbor.gain > 0.0
    }

    fn is_move_to_be_better_than_best(&self, neighbor: &MaxCutFlipNeighbor) -> bool {
        neighbor.gain + self.objective > self.best_objective
    }

    fn is_first_move_better_than_second(
        &self,
        first: &MaxCutFlipNeighbor,
        second: &MaxCutFlipNeighbor,
    ) -> bool {
        first.gain > second.gain
    }
}

impl Evaluable<f64> for MaxCutFlipNeighbor {
    fn evaluate(&self) -> f64 {
        self.gain as f64
    }
}

pub struct MaxCutSwapNeighbor {
    i: usize,
    j: usize,
    gain: f32,
}

impl std::hash::Hash for MaxCutSwapNeighbor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.i.hash(state);
        self.j.hash(state);
    }
}

impl<'a> EnumerateMoveToNeighbor<MaxCutSwapNeighbor> for SearchState<'a, MaxCut> {
    fn apply_to_iteration(&mut self, _: &MaxCutSwapNeighbor) {
        self.iteration += 2;
    }

    fn apply_to_solution(&mut self, neighbor: &MaxCutSwapNeighbor) {
        // cut side of the vertex
        let bi = *self
            .solution
            .cut
            .get(&neighbor.i)
            .expect(format!("vertex {} is not found in solution.", neighbor.i).as_str());
        let bj = *self
            .solution
            .cut
            .get(&neighbor.j)
            .expect(format!("vertex {} is not found in solution.", neighbor.j).as_str());

        self.solution.cut.insert(neighbor.i, bj);
        self.solution.cut.insert(neighbor.j, bi);

        // Update the gain for the swapped vertices
    }

    fn apply_to_objective(&mut self, neighbor: &MaxCutSwapNeighbor) {
        self.objective += neighbor.gain;
    }

    fn iter_on_move_to_neighbor(&self) -> impl Iterator<Item = MaxCutSwapNeighbor> {
        (0..self.instance.len())
            .flat_map(move |i| (0..i).map(move |j| MaxCutSwapNeighbor { i, j, gain: 0.0 }))
    }

    fn is_move_to_be_better_than_currernt(&self, neighbor: &MaxCutSwapNeighbor) -> bool {
        neighbor.gain > 0.0
    }

    fn is_move_to_be_better_than_best(&self, neighbor: &MaxCutSwapNeighbor) -> bool {
        neighbor.gain + self.objective > self.best_objective
    }

    fn is_first_move_better_than_second(
        &self,
        first: &MaxCutSwapNeighbor,
        second: &MaxCutSwapNeighbor,
    ) -> bool {
        first.gain > second.gain
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

        let _ = SearchState::new(&mc, rand::rng());
    }

    #[test]
    fn test_flip_neighbor() {
        let mut mc = MaxCut::new();
        mc.add_weight(0, 1, 1.0);
        mc.add_weight(0, 2, 1.0);
        mc.add_weight(1, 2, 1.0);

        let mut state = SearchState::new(&mc, rand::rng());
        state.solution.cut.insert(0, true);
        state.solution.cut.insert(1, false);
        state.solution.cut.insert(2, true);

        let neighbor = MaxCutFlipNeighbor { i: 1, gain: -2.0 };
        state.apply(&neighbor);

        assert_eq!(state.solution.cut[&1], true);
    }
}
