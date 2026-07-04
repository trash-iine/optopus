use std::collections::HashSet;

use crate::common::{Graph, lift_binary_solution, uniform_binary_crossover};
use crate::search_state::{Crossover, SubProblemExtractable};

use super::problem::{MaxCut, MaxCutSolution};

/// Uniform crossover for MaxCut.
///
/// For each vertex whose side differs between the two parents, the offspring
/// inherits the side from `sol1` or `sol2` with equal probability.
/// Vertices with the same side in both parents are inherited unchanged.
///
/// # Usage
///
/// ```
/// use optopus::prelude::*;
/// use optopus::problem::MaxCutUniformCrossover;
///
/// let mc = MaxCut::from_edges([
///     (0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0),
///     (2, 3, 1.0), (3, 4, 1.0),
/// ]);
///
/// let mut ga = GeneticAlgorithm::new(
///     StopCondition::iterations(10_000),
///     20,  // population size
///     MaxCutUniformCrossover,
///     Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
///         StopCondition::failed_updates(100),
///     )),
/// );
/// let mut state = SearchState::new(&mc);
/// ga.run(&mut state).unwrap();
/// ```
pub struct MaxCutUniformCrossover;

impl Crossover<MaxCut> for MaxCutUniformCrossover {
    /// Produces an offspring by cloning `sol1` and then, for each vertex where
    /// `sol1` and `sol2` disagree, randomly choosing one parent's assignment
    /// with 50/50 probability.
    ///
    /// The resulting solution has correct `gain` and `objective` values
    /// (maintained incrementally via flip moves).
    fn crossover(
        &mut self,
        prob: &MaxCut,
        sol1: &MaxCutSolution,
        sol2: &MaxCutSolution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Result<MaxCutSolution, crate::error::OptError> {
        uniform_binary_crossover(prob, sol1, sol2, rng)
    }
}

impl SubProblemExtractable for MaxCut {
    /// Creates a sub-MaxCut containing only vertices whose side assignment
    /// differs between the two parent solutions.
    ///
    /// Only edges *between* free (disagreeing) vertices are included.
    /// Vertices that are isolated in the sub-problem (no edges to other free vertices)
    /// do not appear in the sub-solution and will inherit `sol1`'s assignment
    /// in [`Self::lift_solution`].
    ///
    /// This is the key building block for [`SubProblemBasedCrossover`](crate::heuristic::SubProblemBasedCrossover),
    /// which solves the sub-problem with a local heuristic before lifting.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// let mc = MaxCut::from_edges([(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)]);
    ///
    /// // Parent A: all false, Parent B: all true → all vertices disagree
    /// let sol_a = MaxCutSolution::new_from_assignment(&mc,vec![false; 3]);
    /// let sol_b = MaxCutSolution::new_from_assignment(&mc,vec![true; 3]);
    /// let sub = mc.extract_sub_problem(&sol_a, &sol_b);
    /// assert_eq!(sub.graph.num_vertices(), 3);  // all vertices are free
    /// assert_eq!(sub.graph.num_edges(), 3);
    ///
    /// // Same parents → no disagreement → empty sub-problem
    /// let sub_same = mc.extract_sub_problem(&sol_a, &sol_a);
    /// assert!(sub_same.graph.is_empty());
    /// ```
    fn extract_sub_problem(&self, sol1: &MaxCutSolution, sol2: &MaxCutSolution) -> MaxCut {
        let free: HashSet<usize> = self
            .graph
            .iter_on_vertices()
            .filter(|&&v| sol1.x[v] != sol2.x[v])
            .copied()
            .collect();

        let mut sub_graph = Graph::new();
        for &u in &free {
            for &(v, w) in self.graph.iter_on_adjacency(u) {
                if free.contains(&v) && u < v {
                    sub_graph.add_weight(u, v, w);
                }
            }
        }
        MaxCut::new(sub_graph)
    }

    /// Lifts the sub-problem solution back into the full solution space.
    ///
    /// - **Fixed vertices** (same side in both parents): inherit from `sol1`.
    /// - **Free vertices** (different side): take from `sub_solution`.
    ///
    /// The returned solution has correct `gain` and `objective` values.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// let mc = MaxCut::from_edges([(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)]);
    ///
    /// // Vertex 0 agrees (false), vertices 1 and 2 disagree
    /// let sol_a = MaxCutSolution::new_from_assignment(&mc,vec![false, false, false]);
    /// let sol_b = MaxCutSolution::new_from_assignment(&mc,vec![false, true, true]);
    ///
    /// let sub = mc.extract_sub_problem(&sol_a, &sol_b);
    ///
    /// // Solve sub-problem: assign vertex 1=true, 2=false
    /// let sub_sol = MaxCutSolution::new_from_assignment(&sub,vec![false, true, false]);
    ///
    /// let lifted = mc.lift_solution(&sol_a, &sol_b, &sub_sol);
    /// assert_eq!(lifted.x[0], false);  // fixed from sol_a
    /// assert_eq!(lifted.x[1], true);   // from sub_solution
    /// assert_eq!(lifted.x[2], false);  // from sub_solution
    /// ```
    fn lift_solution(
        &self,
        sol1: &MaxCutSolution,
        sol2: &MaxCutSolution,
        sub_solution: &MaxCutSolution,
    ) -> MaxCutSolution {
        lift_binary_solution(
            self,
            sol1,
            sol2,
            sub_solution,
            sub_solution.iter_on_vertices(),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::problem::max_cut::{MaxCut, MaxCutSolution};
    use crate::search_state::{Crossover, SubProblemExtractable};
    use rand::SeedableRng;

    use super::MaxCutUniformCrossover;

    fn make_mc() -> MaxCut {
        MaxCut::from_edges([(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)])
    }

    fn make_sol(mc: &MaxCut, assignments: &[(usize, bool)]) -> MaxCutSolution {
        let mut cut = vec![false; mc.graph.len()];
        for &(v, side) in assignments {
            cut[v] = side;
        }
        MaxCutSolution::new_from_assignment(mc, cut)
    }

    #[test]
    fn test_uniform_crossover_identical_parents() {
        let mc = make_mc();
        let s = make_sol(&mc, &[(0, false), (1, true), (2, false)]);
        let mut cx = MaxCutUniformCrossover;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let offspring = cx.crossover(&mc, &s, &s, &mut rng).unwrap();
        assert_eq!(offspring.x, s.x);
        assert_eq!(offspring.objective, s.objective);
    }

    #[test]
    fn test_uniform_crossover_gain_consistency() {
        let mc = make_mc();
        let a = make_sol(&mc, &[(0, false), (1, true), (2, false)]);
        let b = make_sol(&mc, &[(0, true), (1, false), (2, true)]);
        let mut cx = MaxCutUniformCrossover;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let offspring = cx.crossover(&mc, &a, &b, &mut rng).unwrap();
        for &v in mc.graph.iter_on_vertices() {
            let g = offspring.gain[v];
            let mut flipped = offspring.x.clone();
            flipped[v] = !flipped[v];
            let expected = mc.calculate_cut_size(&flipped) - offspring.objective;
            assert!(
                (g - expected).abs() < 1e-5,
                "gain[{v}] mismatch: {g} vs {expected}"
            );
        }
    }

    #[test]
    fn test_extract_sub_problem_size() {
        let mc = make_mc();
        let s = make_sol(&mc, &[(0, false), (1, true), (2, false)]);
        let sub_same = mc.extract_sub_problem(&s, &s);
        assert_eq!(
            sub_same.graph.len(),
            0,
            "identical parents → 0 free vertices"
        );

        let all_f = make_sol(&mc, &[(0, false), (1, false), (2, false)]);
        let all_t = make_sol(&mc, &[(0, true), (1, true), (2, true)]);
        let sub_diff = mc.extract_sub_problem(&all_f, &all_t);
        assert_eq!(
            sub_diff.graph.len(),
            3,
            "all-different parents → 3 free vertices"
        );
    }

    #[test]
    fn test_lift_solution() {
        let mc = make_mc();
        // Free: vertices 1 and 2 (differ); Fixed: vertex 0 (same: false)
        let parent_a = make_sol(&mc, &[(0, false), (1, false), (2, false)]);
        let parent_b = make_sol(&mc, &[(0, false), (1, true), (2, true)]);
        let sub = mc.extract_sub_problem(&parent_a, &parent_b);

        // Sub-problem keeps original vertex IDs: vertices 1 and 2 with edge weight 2.0
        let sub_sol = make_sol(&sub, &[(1, true), (2, false)]);
        let lifted = mc.lift_solution(&parent_a, &parent_b, &sub_sol);

        assert_eq!(
            lifted.x[0], parent_a.x[0],
            "fixed vertex 0 inherits from parent_a"
        );
        assert_eq!(
            lifted.x[1], sub_sol.x[1],
            "free vertex 1 comes from sub_solution"
        );
        assert_eq!(
            lifted.x[2], sub_sol.x[2],
            "free vertex 2 comes from sub_solution"
        );

        for &v in mc.graph.iter_on_vertices() {
            let g = lifted.gain[v];
            let mut flipped = lifted.x.clone();
            flipped[v] = !flipped[v];
            let expected = mc.calculate_cut_size(&flipped) - lifted.objective;
            assert!((g - expected).abs() < 1e-5, "lifted gain[{v}] mismatch");
        }
    }
}
