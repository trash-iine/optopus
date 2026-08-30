use super::problem::{Vrp, VrpSolution};
use super::split::split_giant_tour;
use crate::common::order_crossover;
use crate::search_state::Crossover;

/// Order-crossover (OX) for VRP over the "giant tour" encoding.
///
/// Both parents are flattened into a single customer sequence (routes
/// concatenated in order). A random contiguous segment is copied from parent 1,
/// the remaining customers are filled in parent 2's relative order (classic OX,
/// mirroring [`crate::problem::TspOrderCrossover`]), and the resulting giant tour
/// is decoded into exactly `num_vehicles` routes by [`split_giant_tour`], which
/// picks the cut positions optimally for that order.
///
/// The decode runs under [`Vrp::penalty_weight`], so the quantity Split minimizes
/// is exactly the offspring's [`VrpSolution::objective`]. That fixed penalty is
/// the only thing separating this operator from the recombination step of
/// [`crate::heuristic::HybridGeneticSearchForVrp`], which drives the same decoder
/// with a penalty it retunes as the search runs.
pub struct VrpOrderCrossover;

/// Concatenates all routes into a single customer sequence.
fn flatten(sol: &VrpSolution) -> Vec<usize> {
    sol.routes.iter().flatten().copied().collect()
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
        let routes = split_giant_tour(prob, &child, prob.penalty_weight());
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

    /// From identical parents OX reproduces the parent's own customer order, and
    /// the parent's routes are one way of cutting that order — so the optimal
    /// decoder cannot come back with anything worse. (The greedy splitter this
    /// operator used to end with could, by re-cutting a good partition.)
    #[test]
    fn identical_parents_never_yield_a_worse_child() {
        let prob = vrp();
        // A deliberately mediocre but feasible partition: the tour order is
        // 1,3,5,2,4 and the cuts are not the ones Split would choose.
        let parent = prob.solution_from_routes(vec![vec![1, 3], vec![5, 2], vec![4]]);
        let mut cx = VrpOrderCrossover;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(11);
        for _ in 0..20 {
            let child = cx.crossover(&prob, &parent, &parent, &mut rng).unwrap();
            prob.validate_routes(&child.routes).unwrap();
            assert!(
                child.objective <= parent.objective + 1e-9,
                "child ({}) is worse than the parent it decodes ({})",
                child.objective,
                parent.objective
            );
        }
    }
}
