use super::Heuristic;
use crate::error::OptError;
use crate::search_state::{Crossover, SearchState, SubProblemExtractable};
use rand::RngCore;

/// Generic crossover operator that works for any problem implementing [`SubProblemExtractable`].
///
/// For each crossover call it:
/// 1. Extracts a sub-problem containing only the variables that differ between the two parents.
/// 2. Solves the sub-problem with `sub_heuristic`.
/// 3. Lifts the sub-solution back to the full solution space.
///
/// # MaxCut example
///
/// - Vertices with the same side in both parents are fixed; their edges become bias terms.
/// - Vertices with different sides form the sub-MaxCut instance.
/// - `lift_solution` merges the fixed sides with the sub-problem result.
pub struct SubProblemBasedCrossover<P: SubProblemExtractable> {
    /// Heuristic used to solve the sub-problem (e.g. [`crate::heuristic::LocalSearch`]).
    pub sub_heuristic: Box<dyn Heuristic<P>>,
}

impl<P: SubProblemExtractable> Crossover<P> for SubProblemBasedCrossover<P> {
    fn crossover(
        &mut self,
        prob: &P,
        sol1: &P::Solution,
        sol2: &P::Solution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Result<P::Solution, OptError> {
        let sub_prob = prob.extract_sub_problem(sol1, sol2);
        let sub_seed = rng.next_u64();
        let mut sub_state = SearchState::new_with_seed(&sub_prob, sub_seed);
        self.sub_heuristic.run(&mut sub_state)?;
        Ok(prob.lift_solution(sol1, sol2, &sub_state.best_solution))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::{LocalSearch, StopCondition};
    use crate::problem::{MaxCut, MaxCutFlipNeighbor, MaxCutSolution};
    use rand::SeedableRng;

    fn instance() -> MaxCut {
        MaxCut::from_edges([
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
            (3, 4, 1.0),
            (4, 5, 1.0),
            (5, 0, 1.0),
            (0, 3, 1.0),
        ])
    }

    fn crossover() -> SubProblemBasedCrossover<MaxCut> {
        SubProblemBasedCrossover {
            sub_heuristic: Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
                StopCondition::iterations(200),
            )),
        }
    }

    /// The round trip is the whole operator: what the parents agree on has to
    /// come back untouched, and what they disagree on has to come back as the
    /// sub-search left it. A lift that reads the sub-solution against the
    /// wrong index map produces a valid-looking child of the wrong instance,
    /// which is why the child's caches are checked against a recompute.
    #[test]
    fn the_child_keeps_what_the_parents_agree_on_and_is_exactly_evaluated() {
        let mc = instance();
        let a =
            MaxCutSolution::new_from_assignment(&mc, vec![true, true, false, false, true, false]);
        let b =
            MaxCutSolution::new_from_assignment(&mc, vec![true, false, true, false, false, false]);
        let agreed: Vec<usize> = (0..6).filter(|&i| a.x[i] == b.x[i]).collect();
        assert!(!agreed.is_empty() && agreed.len() < 6, "need a mixed pair");

        let mut rng = rand::rngs::SmallRng::seed_from_u64(9);
        let child = crossover().crossover(&mc, &a, &b, &mut rng).unwrap();

        for &i in &agreed {
            assert_eq!(child.x[i], a.x[i], "vertex {i} was fixed and moved anyway");
        }
        assert_eq!(
            child.objective,
            mc.calculate_cut_size(&child.x),
            "the lifted child's objective is not its cut"
        );
        for &i in mc.graph.iter_on_vertices() {
            assert_eq!(child.gain[i], mc.calculate_gain(&child.x, i), "gain[{i}]");
        }
    }

    /// Identical parents leave nothing free, so the sub-problem is empty and
    /// the child is the parent. An operator that mishandled the empty
    /// sub-problem would error or return something else here.
    #[test]
    fn identical_parents_reproduce_themselves() {
        let mc = instance();
        let a =
            MaxCutSolution::new_from_assignment(&mc, vec![true, false, true, false, true, false]);

        let mut rng = rand::rngs::SmallRng::seed_from_u64(9);
        let child = crossover().crossover(&mc, &a, &a, &mut rng).unwrap();

        assert_eq!(child.x, a.x);
        assert_eq!(child.objective, a.objective);
    }

    /// The sub-search is seeded from the caller's RNG, so a crossover is
    /// reproducible with the stream and varies without it. Drawing no seed at
    /// all would desynchronise every later draw in the generation.
    #[test]
    fn the_sub_search_draws_one_seed_from_the_callers_stream() {
        let mc = instance();
        let a =
            MaxCutSolution::new_from_assignment(&mc, vec![true, true, false, false, true, false]);
        let b =
            MaxCutSolution::new_from_assignment(&mc, vec![false, false, true, true, false, true]);

        let child_of = |seed| {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
            crossover().crossover(&mc, &a, &b, &mut rng).unwrap().x
        };
        assert_eq!(child_of(9), child_of(9), "same stream, same child");

        let mut rng = rand::rngs::SmallRng::seed_from_u64(9);
        let mut probe = rng.clone();
        let _ = crossover().crossover(&mc, &a, &b, &mut rng).unwrap();
        probe.next_u64();
        assert_eq!(
            rng.next_u64(),
            probe.next_u64(),
            "the crossover must consume exactly one draw"
        );
    }
}
