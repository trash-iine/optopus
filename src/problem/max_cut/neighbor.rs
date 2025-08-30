// ! Defines the neighbor structure and the methods to enumerate neighbors for the MaxCut problem.
use super::MaxCut;
use crate::search_state::{EnabledTabu, EnumerateMoveToNeighbor, Evaluable, SearchState};

/// Represents a neighbor in the MaxCut problem where a vertex is flipped.
#[derive(Debug, Clone, Copy)]
pub struct MaxCutFlipNeighbor {
    pub i: usize,
    pub gain: f32,
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

impl EnabledTabu for MaxCutFlipNeighbor {
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
        self.solution.cut.keys().map(move |&i| MaxCutFlipNeighbor {
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

#[derive(Debug, Clone)]
pub struct MaxCutSwapNeighbor {
    pub i: usize,
    pub j: usize,
    pub gain: f32,
}

impl std::hash::Hash for MaxCutSwapNeighbor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.i.hash(state);
        self.j.hash(state);
    }
}

impl std::cmp::PartialEq for MaxCutSwapNeighbor {
    fn eq(&self, other: &Self) -> bool {
        (self.i == other.i && self.j == other.j) || (self.i == other.j && self.j == other.i)
    }
}
impl std::cmp::Eq for MaxCutSwapNeighbor {}

impl EnabledTabu for MaxCutSwapNeighbor {
    type TabuMap = std::collections::HashMap<usize, u64>;

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

impl<'a> EnumerateMoveToNeighbor<MaxCutSwapNeighbor> for SearchState<'a, MaxCut> {
    fn apply_to_iteration(&mut self, _: &MaxCutSwapNeighbor) {
        self.iteration += 2;
    }

    fn apply_to_solution(&mut self, neighbor: &MaxCutSwapNeighbor) {
        let flip_i = MaxCutFlipNeighbor {
            i: neighbor.i,
            gain: self.solution.gain[&neighbor.i],
        };
        self.apply_to_solution(&flip_i);
        let flip_j = MaxCutFlipNeighbor {
            i: neighbor.j,
            gain: self.solution.gain[&neighbor.j],
        };
        self.apply_to_solution(&flip_j);
    }

    fn apply_to_objective(&mut self, neighbor: &MaxCutSwapNeighbor) {
        self.objective += neighbor.gain;
    }

    fn iter_on_move_to_neighbor(&self) -> impl Iterator<Item = MaxCutSwapNeighbor> {
        self.instance.iter_on_vertices().flat_map(move |&i| {
            self.instance
                .iter_on_vertices()
                .filter(move |&&j| j < i && (self.solution.cut[&i] ^ self.solution.cut[&j]))
                .map(move |&j| MaxCutSwapNeighbor {
                    i,
                    j,
                    gain: self.solution.gain[&i]
                        + self.solution.gain[&j]
                        + if self.instance.has_edge(i, j) {
                            2.0 * self.instance.get_weight(i, j)
                        } else {
                            0.0
                        },
                })
        })
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
