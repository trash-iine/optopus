use super::problem::{Coefficient, Qubo};
use crate::search_state::{EnumerateMoveToNeighbor, Evaluable, SearchState};

#[derive(Debug, Clone)]
pub struct QuboFlipNeighbour {
    i: usize,
    gain: Coefficient,
}

impl std::hash::Hash for QuboFlipNeighbour {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.i.hash(state);
    }
}

impl std::cmp::PartialEq for QuboFlipNeighbour {
    fn eq(&self, other: &Self) -> bool {
        self.i == other.i
    }
}
impl std::cmp::Eq for QuboFlipNeighbour {}

impl Evaluable<Coefficient> for QuboFlipNeighbour {
    fn evaluate(&self) -> Coefficient {
        self.gain
    }
}

impl<'a> EnumerateMoveToNeighbor<QuboFlipNeighbour> for SearchState<'a, Qubo> {
    fn apply_to_iteration(&mut self, _: &QuboFlipNeighbour) {
        self.iteration += 1;
    }

    fn apply_to_solution(&mut self, neighbor: &QuboFlipNeighbour) {
        let bi = *self
            .solution
            .x
            .get(&neighbor.i)
            .expect(format!("{} is not found in solution", neighbor.i).as_str());

        self.solution.x.insert(neighbor.i, !bi);

        // update gain_list
        self.solution.gain.insert(neighbor.i, -neighbor.gain);
        for (&j, &q) in self.instance.iter_on_adjacency(neighbor.i) {
            if let Some(&bj) = self.solution.x.get(&j) {
                if bi ^ bj {
                    *self.solution.gain.entry(j).or_insert(0) += q * 2;
                } else {
                    *self.solution.gain.entry(j).or_insert(0) -= q * 2;
                }
            }
        }
        for (&j, &q) in self.instance.iter_on_adjacency(neighbor.i) {
            if self.solution.x[&j] ^ self.solution.x[&neighbor.i] {
                self.solution.gain.insert(j, self.solution.gain[&j] + q);
            } else {
                self.solution.gain.insert(j, self.solution.gain[&j] - q);
            }
        }
    }

    fn apply_to_objective(&mut self, neighbor: &QuboFlipNeighbour) {
        self.objective += neighbor.gain;
    }

    fn iter_on_move_to_neighbor(&self) -> impl Iterator<Item = QuboFlipNeighbour> {
        (0..self.solution.x.len()).map(move |i| QuboFlipNeighbour {
            i,
            gain: self.solution.gain[&i],
        })
    }

    fn is_move_to_be_better_than_currernt(&self, neighbor: &QuboFlipNeighbour) -> bool {
        neighbor.gain < 0
    }

    fn is_move_to_be_better_than_best(&self, neighbor: &QuboFlipNeighbour) -> bool {
        neighbor.gain + self.objective < self.best_objective
    }

    fn is_first_move_better_than_second(
        &self,
        first: &QuboFlipNeighbour,
        second: &QuboFlipNeighbour,
    ) -> bool {
        first.gain < second.gain
    }
}

#[cfg(test)]
mod search_state_tests {
    use super::*;

    #[test]
    fn test_new() {
        let qubo = Qubo::new();
        let state = SearchState::new(&qubo, rand::rng());
        assert_eq!(state.iteration, 0);
        assert_eq!(
            state.solution.x,
            std::collections::HashMap::from([(0, false), (1, false), (2, false)])
        );
        assert_eq!(state.objective, 0);
        assert_eq!(
            state.best_solution.x,
            std::collections::HashMap::from([(0, false), (1, false), (2, false)])
        );
        assert_eq!(state.best_objective, 0);
        assert_eq!(
            state.solution.gain,
            std::collections::HashMap::from([(0, 0), (1, 0), (2, 0)])
        );
    }
}
