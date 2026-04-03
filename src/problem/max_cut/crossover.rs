use std::collections::HashSet;

use rand::Rng;

use crate::search_state::{Crossover, MoveToNeighbor, SubProblemExtractable};

use super::neighbor::MaxCutFlipNeighbor;
use super::problem::{MaxCut, MaxCutSolution};

/// Uniform crossover for MaxCut.
///
/// For each vertex, the side assignment is taken from `sol1` or `sol2`
/// with equal probability.
pub struct MaxCutUniformCrossover;

impl Crossover<MaxCut> for MaxCutUniformCrossover {
    fn crossover(
        &mut self,
        prob: &MaxCut,
        sol1: &MaxCutSolution,
        sol2: &MaxCutSolution,
    ) -> MaxCutSolution {
        let mut rng = rand::rng();
        let mut sol = sol1.clone();
        for &i in prob.iter_on_vertices() {
            if sol.cut[i] != sol2.cut[i] && rng.random::<bool>() {
                let neighbor = MaxCutFlipNeighbor {
                    i,
                    gain: sol.gain[i],
                };
                neighbor
                    .apply_to_solution(prob, &mut sol)
                    .expect("flipping a vertex should never fail");
            }
        }

        sol
    }
}

impl SubProblemExtractable for MaxCut {
    /// Creates a sub-MaxCut containing only vertices whose side assignment
    /// differs between the two parent solutions.
    ///
    /// Only edges *between* free vertices are included in the sub-problem.
    /// Vertices that are isolated in the sub-problem (no edges to other free vertices)
    /// do not appear in the sub-solution and will inherit `sol1`'s assignment
    /// in [`Self::lift_solution`].
    fn extract_sub_problem(&self, sol1: &MaxCutSolution, sol2: &MaxCutSolution) -> MaxCut {
        let free: HashSet<usize> = self
            .iter_on_vertices()
            .filter(|&&v| sol1.cut[v] != sol2.cut[v])
            .copied()
            .collect();

        let mut sub = MaxCut::new();
        for &u in &free {
            for &(v, w) in self.iter_on_adjacency(u) {
                if free.contains(&v) && u < v {
                    sub.add_weight(u, v, w);
                }
            }
        }
        sub
    }

    /// Lifts the sub-problem solution back to the full solution space.
    ///
    /// - Fixed vertices (same side in both parents): inherit from `sol1`.
    /// - Free vertices (different side): take from `sub_solution`.
    fn lift_solution(
        &self,
        sol1: &MaxCutSolution,
        _sol2: &MaxCutSolution,
        sub_solution: &MaxCutSolution,
    ) -> MaxCutSolution {
        let mut sol = sol1.clone();
        for v in sub_solution.iter_on_vertices() {
            if sol.cut[v] == sub_solution.cut[v] {
                continue;
            }
            let neighbor = MaxCutFlipNeighbor {
                i: v,
                gain: sol.gain[v],
            };
            neighbor
                .apply_to_solution(self, &mut sol)
                .expect("flipping should never fail");
        }

        sol
    }
}

#[cfg(test)]
mod tests {
    use crate::problem::max_cut::{MaxCut, MaxCutSolution};
    use crate::search_state::{Crossover, SubProblemExtractable};

    use super::MaxCutUniformCrossover;

    fn make_mc() -> MaxCut {
        let mut mc = MaxCut::new();
        mc.add_weight(0, 1, 1.0);
        mc.add_weight(1, 2, 2.0);
        mc.add_weight(0, 2, 3.0);
        mc
    }

    fn make_sol(mc: &MaxCut, assignments: &[(usize, bool)]) -> MaxCutSolution {
        let n = mc.len();
        let mut cut = vec![false; n];
        for &(v, side) in assignments {
            cut[v] = side;
        }
        let gain = mc
            .iter_on_vertices()
            .map(|&v| (v, mc.calculate_gain(&cut, v)))
            .fold(vec![0.0f32; n], |mut g, (v, gv)| {
                g[v] = gv;
                g
            });
        let objective = mc.calculate_cut_size(&cut);
        MaxCutSolution {
            cut,
            gain,
            objective,
        }
    }

    #[test]
    fn test_uniform_crossover_identical_parents() {
        let mc = make_mc();
        let s = make_sol(&mc, &[(0, false), (1, true), (2, false)]);
        let mut cx = MaxCutUniformCrossover;
        let offspring = cx.crossover(&mc, &s, &s);
        assert_eq!(offspring.cut, s.cut);
        assert_eq!(offspring.objective, s.objective);
    }

    #[test]
    fn test_uniform_crossover_gain_consistency() {
        let mc = make_mc();
        let a = make_sol(&mc, &[(0, false), (1, true), (2, false)]);
        let b = make_sol(&mc, &[(0, true), (1, false), (2, true)]);
        let mut cx = MaxCutUniformCrossover;
        let offspring = cx.crossover(&mc, &a, &b);
        for &v in mc.iter_on_vertices() {
            let g = offspring.gain[v];
            let mut flipped = offspring.cut.clone();
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
        assert_eq!(sub_same.len(), 0, "identical parents → 0 free vertices");

        let all_f = make_sol(&mc, &[(0, false), (1, false), (2, false)]);
        let all_t = make_sol(&mc, &[(0, true), (1, true), (2, true)]);
        let sub_diff = mc.extract_sub_problem(&all_f, &all_t);
        assert_eq!(sub_diff.len(), 3, "all-different parents → 3 free vertices");
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
            lifted.cut[0], parent_a.cut[0],
            "fixed vertex 0 inherits from parent_a"
        );
        assert_eq!(
            lifted.cut[1], sub_sol.cut[1],
            "free vertex 1 comes from sub_solution"
        );
        assert_eq!(
            lifted.cut[2], sub_sol.cut[2],
            "free vertex 2 comes from sub_solution"
        );

        for &v in mc.iter_on_vertices() {
            let g = lifted.gain[v];
            let mut flipped = lifted.cut.clone();
            flipped[v] = !flipped[v];
            let expected = mc.calculate_cut_size(&flipped) - lifted.objective;
            assert!((g - expected).abs() < 1e-5, "lifted gain[{v}] mismatch");
        }
    }
}
