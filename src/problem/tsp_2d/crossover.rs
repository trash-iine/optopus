use std::collections::{HashMap, HashSet};

use rand::Rng;

use crate::search_state::{Crossover, SubProblemExtractable};

use super::problem::{TspSolution, TspTour, TspWithCoordinates};

/// Order Crossover (OX) for TSP.
///
/// Copies a random contiguous segment from `sol1`, then fills the
/// remaining positions in order from `sol2`, skipping cities already
/// present in the copied segment.
pub struct TspOrderCrossover;

impl Crossover<TspWithCoordinates> for TspOrderCrossover {
    fn crossover(
        &mut self,
        prob: &TspWithCoordinates,
        sol1: &TspSolution,
        sol2: &TspSolution,
    ) -> TspSolution {
        let mut rng = rand::rng();
        let n = prob.get_n();

        if n == 0 {
            return sol1.clone();
        }

        // Choose a random contiguous segment from sol1
        let s1 = rng.random_range(0..n);
        let s2 = rng.random_range(0..n);
        let (start, end) = if s1 <= s2 { (s1, s2) } else { (s2, s1) };

        let mut child = vec![usize::MAX; n];
        let mut in_segment = HashSet::with_capacity(end - start + 1);

        for (slot, &city) in child[start..=end].iter_mut().zip(&sol1.tour[start..=end]) {
            *slot = city;
            in_segment.insert(city);
        }

        // Fill remaining positions from sol2, preserving relative order
        let mut pos = (end + 1) % n;
        let mut b_idx = (end + 1) % n;
        let segment_len = end - start + 1;
        let mut filled = segment_len;

        while filled < n {
            let city = sol2.tour[b_idx];
            if !in_segment.contains(&city) {
                child[pos] = city;
                pos = (pos + 1) % n;
                filled += 1;
            }
            b_idx = (b_idx + 1) % n;
        }

        let objective =
            prob.calculate_tour_length(&child).expect("OX crossover should produce a valid tour");
        let gain = prob.compute_all_gains(&child);
        TspSolution { tour: child, objective, gain }
    }
}

/// Returns the set of cities whose incident edges differ between the two parent tours.
///
/// A city is "free" if at least one of its two adjacent edges in `sol1`'s tour
/// is not common to both parent tours (considering edges as undirected).
/// "Fixed" cities have both incident edges shared across both parents and inherit
/// `sol1`'s position in [`TspWithCoordinates::lift_solution`].
fn free_cities(
    prob: &TspWithCoordinates,
    sol1: &TspSolution,
    sol2: &TspSolution,
) -> Vec<usize> {
    let n = prob.get_n();

    let make_edge_set = |tour: &TspTour| -> HashSet<(usize, usize)> {
        let len = tour.len();
        (0..len)
            .map(|k| {
                let u = tour[k];
                let v = tour[(k + 1) % len];
                if u < v { (u, v) } else { (v, u) }
            })
            .collect()
    };

    let edges1 = make_edge_set(&sol1.tour);
    let edges2 = make_edge_set(&sol2.tour);
    let common: HashSet<(usize, usize)> = edges1.intersection(&edges2).copied().collect();

    // position lookup for sol1
    let pos_a: HashMap<usize, usize> =
        sol1.tour.iter().enumerate().map(|(k, &c)| (c, k)).collect();

    (0..n)
        .filter(|&c| {
            let k = pos_a[&c];
            let pred = sol1.tour[(k + n - 1) % n];
            let succ = sol1.tour[(k + 1) % n];
            let e1 = if pred < c { (pred, c) } else { (c, pred) };
            let e2 = if c < succ { (c, succ) } else { (succ, c) };
            !common.contains(&e1) || !common.contains(&e2)
        })
        .collect()
}

impl SubProblemExtractable for TspWithCoordinates {
    /// Creates a sub-TSP containing only the "free" cities — those whose incident
    /// edges differ between the two parent tours.
    ///
    /// Sub-problem city `i` corresponds to `free_cities(...)[i]` in the original problem.
    fn extract_sub_problem(
        &self,
        sol1: &TspSolution,
        sol2: &TspSolution,
    ) -> TspWithCoordinates {
        let free = free_cities(self, sol1, sol2);
        let sub_coords: Vec<(f64, f64)> = free.iter().map(|&c| self.coordinates[c]).collect();
        TspWithCoordinates::new(format!("{}_sub", self.name), sub_coords)
    }

    /// Lifts the sub-problem solution back to the full solution space.
    ///
    /// - Fixed cities (same incident edges in both parents): inherit positions from `sol1`.
    /// - Free cities: replaced in the same relative positions by the sub-problem tour order.
    fn lift_solution(
        &self,
        sol1: &TspSolution,
        sol2: &TspSolution,
        sub_solution: &TspSolution,
    ) -> TspSolution {
        let free = free_cities(self, sol1, sol2);
        let free_set: HashSet<usize> = free.iter().copied().collect();

        // Map sub-problem indices back to original city indices
        let lifted_order: Vec<usize> = sub_solution.tour.iter().map(|&i| free[i]).collect();

        // Positions of free cities in sol1's tour (in tour order)
        let free_positions: Vec<usize> = (0..sol1.tour.len())
            .filter(|&k| free_set.contains(&sol1.tour[k]))
            .collect();

        let mut tour = sol1.tour.clone();
        for (&pos, &city) in free_positions.iter().zip(lifted_order.iter()) {
            tour[pos] = city;
        }

        let objective =
            self.calculate_tour_length(&tour).expect("lifted TSP tour should be valid");
        let gain = self.compute_all_gains(&tour);
        TspSolution { tour, objective, gain }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::problem::tsp_2d::{TspSolution, TspWithCoordinates};
    use crate::search_state::{Crossover, SubProblemExtractable};

    use super::TspOrderCrossover;

    /// 4-city square: (0,0), (1,0), (1,1), (0,1)
    fn make_tsp() -> TspWithCoordinates {
        TspWithCoordinates::new(
            "test".to_string(),
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        )
    }

    fn make_sol(tsp: &TspWithCoordinates, tour: Vec<usize>) -> TspSolution {
        let objective = tsp.calculate_tour_length(&tour).unwrap();
        let gain = tsp.compute_all_gains(&tour);
        TspSolution { tour, objective, gain }
    }

    #[test]
    fn test_order_crossover_valid_tour() {
        let tsp = make_tsp();
        let a = make_sol(&tsp, vec![0, 1, 2, 3]);
        let b = make_sol(&tsp, vec![2, 0, 3, 1]);
        let mut cx = TspOrderCrossover;
        let offspring = cx.crossover(&tsp, &a, &b);
        let cities: HashSet<usize> = offspring.tour.iter().copied().collect();
        assert_eq!(offspring.tour.len(), 4);
        assert_eq!(cities, (0..4).collect::<HashSet<usize>>(), "offspring must visit all 4 cities exactly once");
    }

    #[test]
    fn test_order_crossover_identical_parents() {
        let tsp = make_tsp();
        let s = make_sol(&tsp, vec![0, 1, 2, 3]);
        let mut cx = TspOrderCrossover;
        let offspring = cx.crossover(&tsp, &s, &s);
        assert_eq!(offspring.tour, s.tour);
    }

    #[test]
    fn test_extract_sub_problem_identical_tours() {
        let tsp = make_tsp();
        let s = make_sol(&tsp, vec![0, 1, 2, 3]);
        let sub = tsp.extract_sub_problem(&s, &s);
        assert_eq!(sub.get_n(), 0, "identical tours → 0 free cities");
    }

    #[test]
    fn test_lift_solution_valid_tour() {
        let tsp = make_tsp();
        // tour_a and tour_b share only 1 common edge → all 4 cities are free
        let parent_a = make_sol(&tsp, vec![0, 1, 2, 3]);
        let parent_b = make_sol(&tsp, vec![0, 2, 1, 3]);
        let sub = tsp.extract_sub_problem(&parent_a, &parent_b);

        // Sub-problem contains all 4 free cities (remapped to 0-3)
        let sub_sol = make_sol(&sub, vec![1, 3, 0, 2]);
        let lifted = tsp.lift_solution(&parent_a, &parent_b, &sub_sol);

        let cities: HashSet<usize> = lifted.tour.iter().copied().collect();
        assert_eq!(lifted.tour.len(), 4, "lifted tour must have 4 cities");
        assert_eq!(cities, (0..4).collect::<HashSet<usize>>(), "lifted tour must visit all cities");
    }
}
