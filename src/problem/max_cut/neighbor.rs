// ! Defines the neighbor structure and the methods to enumerate neighbors for the MaxCut problem.
use super::MaxCut;
use crate::{
    problem::max_cut::problem::MaxCutSolution,
    search_state::{EnabledTabu, Evaluable, MoveToNeigbor, Rankable},
};

/// Represents a neighbor in the MaxCut problem where a vertex is flipped.
#[derive(Debug, Clone, Copy)]
pub struct MaxCutFlipNeighbor {
    pub i: usize,
    pub gain: f32,
}
impl Rankable for MaxCutFlipNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain > other.gain
    }
}

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

impl MoveToNeigbor<MaxCut> for MaxCutFlipNeighbor {
    fn apply_to_solution(&self, prob: &MaxCut, solution: &mut MaxCutSolution) {
        // cut side of the vertex
        let bi = *solution
            .cut
            .get(&self.i)
            .expect(format!("vertex {} is not found in solution.", self.i).as_str());

        solution.cut.insert(self.i, !bi);

        // Update the gain for the flipped vertex
        solution.gain.insert(self.i, -self.gain);
        for (&j, &w) in prob.iter_on_adjacency(&self.i) {
            let bj = *solution
                .cut
                .get(&j)
                .expect(format!("vertex {} is not found in the solution.", j).as_str());
            if bi ^ bj {
                *solution.gain.entry(j).or_insert(0.0) += w * 2.0;
            } else {
                *solution.gain.entry(j).or_insert(0.0) -= w * 2.0;
            }
        }

        solution.objective += self.gain;
    }

    fn iter(_: &MaxCut, sol: &MaxCutSolution) -> impl Iterator<Item = Self> + Send {
        sol.cut.keys().map(move |&i| MaxCutFlipNeighbor {
            i,
            gain: *sol
                .gain
                .get(&i)
                .expect(format!("vertex {} is not found in the solution.", i).as_str()),
        })
    }

    fn move_to_be_better_than(
        &self,
        _: &MaxCut,
        src: &<MaxCut as crate::search_state::ProblemTrait>::Solution,
        other: &<MaxCut as crate::search_state::ProblemTrait>::Solution,
    ) -> bool {
        self.gain + src.objective > other.objective
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

impl Rankable for MaxCutSwapNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain > other.gain
    }
}

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

impl MoveToNeigbor<MaxCut> for MaxCutSwapNeighbor {
    fn apply_to_iteration(&self, iter: u64) -> u64 {
        iter + 2
    }
    fn apply_to_solution(
        &self,
        prob: &MaxCut,
        sol: &mut <MaxCut as crate::search_state::ProblemTrait>::Solution,
    ) {
        let flip_i = MaxCutFlipNeighbor {
            i: self.i,
            gain: sol.gain[&self.i],
        };
        flip_i.apply_to_solution(prob, sol);
        let flip_j = MaxCutFlipNeighbor {
            i: self.j,
            gain: sol.gain[&self.j],
        };
        flip_j.apply_to_solution(prob, sol);
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
        src: &<MaxCut as crate::search_state::ProblemTrait>::Solution,
        other: &<MaxCut as crate::search_state::ProblemTrait>::Solution,
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
