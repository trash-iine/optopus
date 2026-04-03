use super::Heuristic;
use crate::search_state::{Crossover, SearchState, SubProblemExtractable};

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
    ) -> P::Solution {
        let sub_prob = prob.extract_sub_problem(sol1, sol2);
        let mut sub_state = SearchState::new(&sub_prob);
        self.sub_heuristic
            .run(&mut sub_state)
            .expect("sub_heuristic failed inside SubProblemBasedCrossover");
        prob.lift_solution(sol1, sol2, &sub_state.best_solution)
    }
}
