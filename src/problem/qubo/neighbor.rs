use super::problem::{Coefficient, Qubo};
use crate::{
    problem::qubo::problem::QuboSolution,
    search_state::{EnabledTabu, Evaluable, MoveToNeigbor, Rankable},
};

#[derive(Debug, Clone)]
pub struct QuboFlipNeighbour {
    i: usize,
    gain: Coefficient,
}

impl Rankable for QuboFlipNeighbour {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain < other.gain
    }
}

impl EnabledTabu for QuboFlipNeighbour {
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

impl Evaluable<Coefficient> for QuboFlipNeighbour {
    fn evaluate(&self) -> Coefficient {
        self.gain
    }
}

impl MoveToNeigbor<Qubo> for QuboFlipNeighbour {
    fn apply_to_solution(
        &self,
        prob: &Qubo,
        sol: &mut <Qubo as crate::search_state::ProblemTrait>::Solution,
    ) {
        let bi = *sol
            .x
            .get(&self.i)
            .expect(format!("{} is not found in solution", self.i).as_str());

        sol.x.insert(self.i, !bi);

        // update gain_list
        sol.gain.insert(self.i, -self.gain);
        for (&j, &q) in prob.iter_on_adjacency(self.i) {
            if let Some(&bj) = sol.x.get(&j) {
                if bi ^ bj {
                    *sol.gain.entry(j).or_insert(0) += q * 2;
                } else {
                    *sol.gain.entry(j).or_insert(0) -= q * 2;
                }
            }
        }
        for (&j, &q) in prob.iter_on_adjacency(self.i) {
            if sol.x[&j] ^ sol.x[&self.i] {
                sol.gain.insert(j, sol.gain[&j] + q);
            } else {
                sol.gain.insert(j, sol.gain[&j] - q);
            }
        }
    }

    fn iter(_: &Qubo, sol: &QuboSolution) -> impl Iterator<Item = Self> + Send {
        (0..sol.x.len()).map(move |i| QuboFlipNeighbour {
            i,
            gain: sol.gain[&i],
        })
    }
    fn move_to_be_better_than(
        &self,
        _: &Qubo,
        src: &<Qubo as crate::search_state::ProblemTrait>::Solution,
        other: &<Qubo as crate::search_state::ProblemTrait>::Solution,
    ) -> bool {
        self.gain + src.objective < other.objective
    }
}
