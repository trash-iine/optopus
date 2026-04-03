use super::problem::{TspSolution, TspWithCoordinates, normalize_edge_pair};
use crate::{
    error::OptError,
    search_state::{EnabledTabu, Evaluate, Evaluable, MoveToNeighbor, Rankable},
};

/// A 2-opt move that removes edges `(tour[i], tour[i+1])` and `(tour[j], tour[(j+1)%n])`,
/// then reconnects as `(tour[i], tour[j])` and `(tour[i+1], tour[(j+1)%n])`.
///
/// Implemented by reversing the sub-segment `tour[i+1..=j]`.
/// `gain` is the change in tour length after the move (negative = improvement).
#[derive(Debug, Clone)]
pub struct TspTwoOptNeighbor {
    /// Smallest index of the two removed edges.
    pub i: usize,
    /// Largest index of the two removed edges.
    pub j: usize,
    /// Change in tour length (negative = improvement).
    pub gain: f64,
}

impl Rankable for TspTwoOptNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain < other.gain
    }
}

impl Evaluate for TspTwoOptNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain)
    }
}

impl EnabledTabu for TspTwoOptNeighbor {
    // Keyed by the (i, j) pair (normalized so i < j)
    type TabuMap = std::collections::HashMap<(usize, usize), u64>;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        let key = (self.i.min(self.j), self.i.max(self.j));
        tabu_map.get(&key).map_or(true, |&t| iteration > t)
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
    ) {
        let d = rand::random_range(tabu_tenure.0..=tabu_tenure.1);
        let key = (self.i.min(self.j), self.i.max(self.j));
        tabu_map.insert(key, iteration + d);
    }
}

impl MoveToNeighbor<TspWithCoordinates> for TspTwoOptNeighbor {
    fn apply_to_solution(
        &self,
        prob: &TspWithCoordinates,
        sol: &mut TspSolution,
    ) -> Result<(), OptError> {
        let n = sol.tour.len();
        // All edges at positions [i..=j] change: boundaries swap endpoints, internals reverse.
        for k in self.i..=self.j {
            let e = (sol.tour[k], sol.tour[(k + 1) % n]);
            prob.update_gains_for_removed_edge(sol, e);
        }
        sol.tour[self.i + 1..=self.j].reverse();
        sol.objective += self.gain;
        for k in self.i..=self.j {
            let e = (sol.tour[k], sol.tour[(k + 1) % n]);
            prob.update_gains_for_added_edge(sol, e);
        }
        Ok(())
    }

    fn iter(prob: &TspWithCoordinates, sol: &TspSolution) -> impl Iterator<Item = Self> + Send {
        let n = sol.tour.len();
        (0..n - 1).flat_map(move |i| {
            // When i=0, j=n-1 would reverse the entire tour (trivially equivalent for undirected),
            // so exclude that case to avoid redundant moves
            let max_j = if i == 0 { n - 1 } else { n };
            (i + 2..max_j).map(move |j| {
                let e1 = prob.get_edge_from(&sol.tour, i);
                let e2 = prob.get_edge_from(&sol.tour, j);
                let key = normalize_edge_pair(e1, e2);

                TspTwoOptNeighbor {
                    i,
                    j,
                    gain: sol.gain[&key],
                }
            })
        })
    }

    fn move_to_be_better_than(
        &self,
        _: &TspWithCoordinates,
        src: &TspSolution,
        other: &TspSolution,
    ) -> bool {
        self.gain + src.objective < other.objective
    }
}

/// A relocate move that removes city `tour[pos]` from its current position
/// and inserts it immediately after `tour[ins]`.
///
/// `pos` and `ins` are indices in the original tour before the move is applied.
/// `gain` is the change in tour length (negative = improvement).
#[derive(Debug, Clone)]
pub struct TspRelocateNeighbor {
    /// Index of the city to be relocated.
    pub pos: usize,
    /// Index of the city after which the relocated city will be inserted.
    pub ins: usize,
    /// Change in tour length (negative = improvement).
    pub gain: f64,
}

impl Rankable for TspRelocateNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain < other.gain
    }
}

impl Evaluate for TspRelocateNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain)
    }
}

impl EnabledTabu for TspRelocateNeighbor {
    // Keyed by the (pos, ins) pair
    type TabuMap = std::collections::HashMap<(usize, usize), u64>;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        tabu_map
            .get(&(self.pos, self.ins))
            .map_or(true, |&t| iteration > t)
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
    ) {
        let d = rand::random_range(tabu_tenure.0..=tabu_tenure.1);
        tabu_map.insert((self.pos, self.ins), iteration + d);
    }
}

impl MoveToNeighbor<TspWithCoordinates> for TspRelocateNeighbor {
    fn apply_to_solution(
        &self,
        prob: &TspWithCoordinates,
        sol: &mut TspSolution,
    ) -> Result<(), OptError> {
        let n = sol.tour.len();
        let prev = (self.pos + n - 1) % n;
        let next = (self.pos + 1) % n;
        let ins_next = (self.ins + 1) % n;
        let c_prev = sol.tour[prev];
        let c_pos = sol.tour[self.pos];
        let c_next = sol.tour[next];
        let c_ins = sol.tour[self.ins];
        let c_ins_next = sol.tour[ins_next];
        prob.update_gains_for_removed_edge(sol, (c_prev, c_pos));
        prob.update_gains_for_removed_edge(sol, (c_pos, c_next));
        prob.update_gains_for_removed_edge(sol, (c_ins, c_ins_next));
        let city = sol.tour.remove(self.pos);
        let insert_at = if self.ins < self.pos {
            self.ins + 1
        } else {
            self.ins
        };
        sol.tour.insert(insert_at, city);
        sol.objective += self.gain;
        prob.update_gains_for_added_edge(sol, (c_prev, c_next));
        prob.update_gains_for_added_edge(sol, (c_ins, c_pos));
        prob.update_gains_for_added_edge(sol, (c_pos, c_ins_next));
        Ok(())
    }

    fn iter(prob: &TspWithCoordinates, sol: &TspSolution) -> impl Iterator<Item = Self> + Send {
        let n = prob.get_n();
        // Eager collection: no Arc needed since all items are built before returning.
        let mut items: Vec<TspRelocateNeighbor> = Vec::with_capacity(n * n.saturating_sub(2));

        for pos in 0..n {
            let prev = (pos + n - 1) % n;
            let next = (pos + 1) % n;

            for ins in 0..n {
                // Inserting after pos itself or after its predecessor (prev) would
                // leave the tour unchanged, so skip these cases
                if ins == pos || ins == prev {
                    continue;
                }

                let ins_next = (ins + 1) % n;

                // Cost saved by removing city pos from prev-pos-next
                let removal_gain = prob.distance(sol.tour[prev], sol.tour[pos])
                    + prob.distance(sol.tour[pos], sol.tour[next])
                    - prob.distance(sol.tour[prev], sol.tour[next]);

                // Additional cost of inserting city pos between ins and ins_next
                let insertion_cost = prob.distance(sol.tour[ins], sol.tour[pos])
                    + prob.distance(sol.tour[pos], sol.tour[ins_next])
                    - prob.distance(sol.tour[ins], sol.tour[ins_next]);

                let gain = insertion_cost - removal_gain;
                items.push(TspRelocateNeighbor { pos, ins, gain });
            }
        }

        items.into_iter()
    }

    fn move_to_be_better_than(
        &self,
        _: &TspWithCoordinates,
        src: &TspSolution,
        other: &TspSolution,
    ) -> bool {
        self.gain + src.objective < other.objective
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_square_tsp() -> TspWithCoordinates {
        // 4 cities forming a unit square: (0,0), (1,0), (1,1), (0,1)
        // Edge length = 1, diagonal = sqrt(2)
        TspWithCoordinates::new(
            "square".to_string(),
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        )
    }

    #[test]
    fn test_two_opt_gain_calculation() {
        let tsp = make_square_tsp();
        // Initial tour [0,1,3,2] has a crossing (length = 1 + sqrt(2) + 1 + sqrt(2))
        let tour = vec![0, 1, 3, 2];
        let objective = tsp.calculate_tour_length(&tour).unwrap();
        let gain = tsp.compute_all_gains(&tour);
        let sol = TspSolution {
            tour,
            objective,
            gain,
        };

        // 2-opt (i=1, j=2): replace edges (1,3) and (2,0) with (1,2) and (3,0)
        // → tour [0,1,2,3] (length = 4.0)
        let neighbors: Vec<_> = TspTwoOptNeighbor::iter(&tsp, &sol).collect();
        let best = neighbors
            .iter()
            .min_by(|a, b| a.gain.partial_cmp(&b.gain).unwrap())
            .unwrap();
        assert!(
            best.gain < 0.0,
            "the best 2-opt move should be an improvement"
        );

        let mut sol2 = sol.clone();
        best.apply_to_solution(&tsp, &mut sol2).unwrap();
        let expected = tsp.calculate_tour_length(&sol2.tour).unwrap();
        assert!((sol2.objective - expected).abs() < 1e-9);
    }

    #[test]
    fn test_two_opt_apply_consistency() {
        let tsp = make_square_tsp();
        let tour = vec![0, 2, 1, 3];
        let objective = tsp.calculate_tour_length(&tour).unwrap();
        let gain = tsp.compute_all_gains(&tour);
        let sol = TspSolution {
            tour,
            objective,
            gain,
        };

        for neighbor in TspTwoOptNeighbor::iter(&tsp, &sol) {
            let mut s = sol.clone();
            neighbor.apply_to_solution(&tsp, &mut s).unwrap();
            let expected = tsp.calculate_tour_length(&s.tour).unwrap();
            assert!(
                (s.objective - expected).abs() < 1e-9,
                "2-opt ({},{}) after: objective={} expected={}",
                neighbor.i,
                neighbor.j,
                s.objective,
                expected
            );
            let expected_gains = tsp.compute_all_gains(&s.tour);
            for (key, &val) in &s.gain {
                let exp = expected_gains[key];
                assert!(
                    (val - exp).abs() < 1e-9,
                    "2-opt ({},{}) gain[{:?}]: {} expected {}",
                    neighbor.i,
                    neighbor.j,
                    key,
                    val,
                    exp
                );
            }
        }
    }

    #[test]
    fn test_relocate_apply_consistency() {
        let tsp = make_square_tsp();
        let tour = vec![0, 1, 2, 3];
        let objective = tsp.calculate_tour_length(&tour).unwrap();
        let gain = tsp.compute_all_gains(&tour);
        let sol = TspSolution {
            tour,
            objective,
            gain,
        };

        for neighbor in TspRelocateNeighbor::iter(&tsp, &sol) {
            let mut s = sol.clone();
            neighbor.apply_to_solution(&tsp, &mut s).unwrap();
            let expected = tsp.calculate_tour_length(&s.tour).unwrap();
            assert!(
                (s.objective - expected).abs() < 1e-9,
                "relocate (pos={}, ins={}) after: objective={} expected={}",
                neighbor.pos,
                neighbor.ins,
                s.objective,
                expected
            );
            let expected_gains = tsp.compute_all_gains(&s.tour);
            for (key, &val) in &s.gain {
                let exp = expected_gains[key];
                assert!(
                    (val - exp).abs() < 1e-9,
                    "relocate (pos={}, ins={}) gain[{:?}]: {} expected {}",
                    neighbor.pos,
                    neighbor.ins,
                    key,
                    val,
                    exp
                );
            }
        }
    }

    #[test]
    fn test_relocate_tour_is_valid_permutation() {
        let tsp = make_square_tsp();
        let tour = vec![0, 1, 2, 3];
        let objective = tsp.calculate_tour_length(&tour).unwrap();
        let gain = tsp.compute_all_gains(&tour);
        let sol = TspSolution {
            tour,
            objective,
            gain,
        };

        for neighbor in TspRelocateNeighbor::iter(&tsp, &sol) {
            let mut s = sol.clone();
            neighbor.apply_to_solution(&tsp, &mut s).unwrap();
            let mut sorted = s.tour.clone();
            sorted.sort();
            assert_eq!(
                sorted,
                vec![0, 1, 2, 3],
                "tour should remain a valid permutation"
            );
        }
    }
}
