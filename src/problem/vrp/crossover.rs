use super::problem::{Vrp, VrpSolution};
use crate::common::order_crossover;
use crate::search_state::Crossover;

/// Order-crossover (OX) for VRP over the "giant tour" encoding.
///
/// Both parents are flattened into a single customer sequence (routes
/// concatenated in order). A random contiguous segment is copied from parent 1,
/// the remaining customers are filled in parent 2's relative order (classic OX,
/// mirroring [`crate::problem::TspOrderCrossover`]), and the resulting giant tour
/// is greedily split into exactly `num_vehicles` capacity-respecting routes.
pub struct VrpOrderCrossover;

/// Concatenates all routes into a single customer sequence.
fn flatten(sol: &VrpSolution) -> Vec<usize> {
    sol.routes.iter().flatten().copied().collect()
}

/// Greedily splits a giant customer sequence into exactly `num_vehicles` routes,
/// starting a new route whenever the current one would exceed capacity (the last
/// route absorbs any remaining customers, incurring penalty if overloaded).
pub(super) fn split_into_routes(prob: &Vrp, giant: &[usize]) -> Vec<Vec<usize>> {
    let v = prob.num_vehicles;
    let mut routes: Vec<Vec<usize>> = vec![Vec::new(); v];
    let mut load = 0i64;
    let mut r = 0usize;
    for &c in giant {
        let dc = prob.demands[c];
        if r < v - 1 && !routes[r].is_empty() && load + dc > prob.capacity {
            r += 1;
            load = 0;
        }
        routes[r].push(c);
        load += dc;
    }
    routes
}

impl Crossover<Vrp> for VrpOrderCrossover {
    fn crossover(
        &mut self,
        prob: &Vrp,
        sol1: &VrpSolution,
        sol2: &VrpSolution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Result<VrpSolution, crate::error::OptError> {
        let n = prob.get_n();
        if n == 0 {
            return Ok(sol1.clone());
        }

        let child = order_crossover(&flatten(sol1), &flatten(sol2), rng);
        let routes = split_into_routes(prob, &child);
        Ok(prob.solution_from_routes(routes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn vrp() -> Vrp {
        Vrp::new(
            "t",
            vec![
                (0.0, 0.0),
                (1.0, 0.0),
                (2.0, 0.0),
                (0.0, 1.0),
                (0.0, 2.0),
                (-1.0, 0.0),
            ],
            vec![0, 1, 1, 1, 1, 1],
            2,
            3,
        )
    }

    #[test]
    fn offspring_is_valid_permutation() {
        let prob = vrp();
        let a = prob.solution_from_routes(vec![vec![1, 2], vec![3, 4], vec![5]]);
        let b = prob.solution_from_routes(vec![vec![5, 3], vec![1, 4], vec![2]]);
        let mut cx = VrpOrderCrossover;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        for _ in 0..20 {
            let child = cx.crossover(&prob, &a, &b, &mut rng).unwrap();
            assert_eq!(child.routes.len(), prob.num_vehicles);
            prob.validate_routes(&child.routes).unwrap();
        }
    }

    #[test]
    fn identical_parents_preserve_customers() {
        let prob = vrp();
        let s = prob.solution_from_routes(vec![vec![1, 2], vec![3, 4], vec![5]]);
        let mut cx = VrpOrderCrossover;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
        let child = cx.crossover(&prob, &s, &s, &mut rng).unwrap();
        prob.validate_routes(&child.routes).unwrap();
    }
}
