use std::collections::HashSet;

use crate::common::{Graph, lift_binary_solution, uniform_binary_crossover};
use crate::search_state::{Crossover, SubProblemExtractable};

use super::problem::{VertexCover, VertexCoverSolution};

/// Uniform crossover for Vertex Cover.
///
/// For each vertex, the membership is taken from `sol1` or `sol2`
/// with equal probability.
pub struct VertexCoverUniformCrossover;

impl Crossover<VertexCover> for VertexCoverUniformCrossover {
    fn crossover(
        &mut self,
        prob: &VertexCover,
        sol1: &VertexCoverSolution,
        sol2: &VertexCoverSolution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Result<VertexCoverSolution, crate::error::OptError> {
        uniform_binary_crossover(prob, sol1, sol2, rng)
    }
}

impl SubProblemExtractable for VertexCover {
    /// Builds a sub-VertexCover containing only the vertices whose membership
    /// differs between the two parents. Only edges *between* free vertices are kept.
    fn extract_sub_problem(
        &self,
        sol1: &VertexCoverSolution,
        sol2: &VertexCoverSolution,
    ) -> VertexCover {
        let free: HashSet<usize> = self
            .graph
            .iter_on_vertices()
            .filter(|&&v| sol1.x[v] != sol2.x[v])
            .copied()
            .collect();

        let mut sub_graph = Graph::new();
        for &u in &free {
            for &(v, _w) in self.graph.iter_on_adjacency(u) {
                if free.contains(&v) && u < v {
                    sub_graph.add_edge(u, v);
                }
            }
        }
        VertexCover::new(sub_graph)
    }

    /// Lifts the sub-solution back into the full solution space.
    ///
    /// - Fixed vertices (same membership in both parents): inherit from `sol1`.
    /// - Free vertices: take from `sub_solution`.
    fn lift_solution(
        &self,
        sol1: &VertexCoverSolution,
        sol2: &VertexCoverSolution,
        sub_solution: &VertexCoverSolution,
    ) -> VertexCoverSolution {
        lift_binary_solution(self, sol1, sol2, sub_solution, 0..sub_solution.x.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_vc() -> VertexCover {
        let mut g = Graph::new();
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(0, 2);
        VertexCover::new(g)
    }

    #[test]
    fn test_uniform_crossover_identical_parents() {
        let vc = make_vc();
        let s = vc.solution_from_assignment(&[true, false, true]);
        let mut cx = VertexCoverUniformCrossover;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let offspring = cx.crossover(&vc, &s, &s, &mut rng).unwrap();
        assert_eq!(offspring.x, s.x);
        assert_eq!(offspring.objective, s.objective);
    }

    #[test]
    fn test_uniform_crossover_gain_consistency() {
        let vc = make_vc();
        let a = vc.solution_from_assignment(&[true, false, true]);
        let b = vc.solution_from_assignment(&[false, true, false]);
        let mut cx = VertexCoverUniformCrossover;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let offspring = cx.crossover(&vc, &a, &b, &mut rng).unwrap();

        let (gain, obj, cs, ue) = vc.calculate_state(&offspring.x);
        assert_eq!(offspring.gain, gain);
        assert_eq!(offspring.objective, obj);
        assert_eq!(offspring.cover_size, cs);
        assert_eq!(offspring.uncovered_edges, ue);
    }

    #[test]
    fn test_extract_sub_problem_size() {
        let vc = make_vc();
        let s = vc.solution_from_assignment(&[true, false, true]);
        let sub_same = vc.extract_sub_problem(&s, &s);
        assert_eq!(sub_same.graph.len(), 0);

        let all_f = vc.solution_from_assignment(&[false, false, false]);
        let all_t = vc.solution_from_assignment(&[true, true, true]);
        let sub_diff = vc.extract_sub_problem(&all_f, &all_t);
        assert_eq!(sub_diff.graph.len(), 3);
    }

    #[test]
    fn test_lift_solution() {
        let vc = make_vc();
        // Free: vertices 1 and 2 (differ); Fixed: vertex 0 (same: false).
        let parent_a = vc.solution_from_assignment(&[false, false, false]);
        let parent_b = vc.solution_from_assignment(&[false, true, true]);
        let sub = vc.extract_sub_problem(&parent_a, &parent_b);

        let sub_sol = sub.solution_from_assignment(&[false, true, false]);
        let lifted = vc.lift_solution(&parent_a, &parent_b, &sub_sol);

        // Fixed vertex 0 inherits from parent_a.
        assert_eq!(lifted.x[0], parent_a.x[0]);
        assert_eq!(lifted.x[1], sub_sol.x[1]);
        assert_eq!(lifted.x[2], sub_sol.x[2]);

        // Verify gain consistency.
        let (gain, obj, cs, ue) = vc.calculate_state(&lifted.x);
        assert_eq!(lifted.gain, gain);
        assert_eq!(lifted.objective, obj);
        assert_eq!(lifted.cover_size, cs);
        assert_eq!(lifted.uncovered_edges, ue);
    }
}
