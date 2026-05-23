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
/// # References
///
/// - Whitley, D., Hains, D., and Howe, A. "A Hybrid Genetic Algorithm for the Traveling Salesman
///   Problem Using Generalized Partition Crossover." In *Parallel Problem Solving from Nature
///   (PPSN XI)*, pp. 566-575. Springer, 2010.
///   [DOI](https://doi.org/10.1007/978-3-642-15844-5_57)
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
        // Fork a deterministic seed from the parent RNG so the sub-search
        // stays reproducible while remaining independent of the outer stream.
        let sub_seed = rng.next_u64();
        let mut sub_state = SearchState::new_with_seed(&sub_prob, sub_seed);
        self.sub_heuristic.run(&mut sub_state)?;
        Ok(prob.lift_solution(sol1, sol2, &sub_state.best_solution))
    }
}
