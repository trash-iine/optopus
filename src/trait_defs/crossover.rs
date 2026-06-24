use super::problem::ProblemTrait;

/// Combines parent solutions into a single offspring solution.
///
/// Implement this for each crossover operator you want to use.
///
/// # `&mut self`
///
/// `&mut self` is required so that operators like [`crate::heuristic::SubProblemBasedCrossover`]
/// can call an inner heuristic's `run` method during crossover.
/// Stateless operators (e.g. uniform crossover) simply do not mutate `self`.
pub trait Crossover<Problem: ProblemTrait> {
    /// Combines two parent solutions into a single offspring.
    ///
    /// Returns `Err` only when the operator genuinely cannot produce an
    /// offspring (e.g. an inner sub-heuristic failed). Stateless operators
    /// such as the per-problem uniform crossovers never fail and simply
    /// return `Ok(...)`.
    fn crossover(
        &mut self,
        prob: &Problem,
        sol1: &Problem::Solution,
        sol2: &Problem::Solution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Result<Problem::Solution, crate::error::OptError>;
}

/// Forwards [`Crossover`] through a boxed trait object so `GeneticAlgorithm`
/// can use a runtime-selected operator (needed by the benchmark TOML config).
impl<P: ProblemTrait> Crossover<P> for Box<dyn Crossover<P>> {
    fn crossover(
        &mut self,
        prob: &P,
        sol1: &P::Solution,
        sol2: &P::Solution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Result<P::Solution, crate::error::OptError> {
        (**self).crossover(prob, sol1, sol2, rng)
    }
}

/// Enables a problem to create a sub-problem from multiple parent solutions and lift the result back.
///
/// Used by [`crate::heuristic::SubProblemBasedCrossover`] to implement crossover by:
/// 1. Finding variables where the parents disagree (free variables).
/// 2. Building a reduced sub-problem containing only those variables.
///    Contributions from fixed (agreeing) variables are folded into the sub-problem objective.
/// 3. Solving the sub-problem with any [`crate::heuristic::Heuristic`].
/// 4. Lifting the sub-solution back to the full solution space.
///
/// # Example (MaxCut)
///
/// - Vertices with the same side in both parents → fixed; their edges become bias terms.
/// - Vertices with different sides → free; form the sub-MaxCut instance.
/// - `lift_solution` merges fixed assignments (from `sol1`) with the sub-problem result.
///
/// # Sized bound
///
/// Required because `extract_sub_problem` returns `Self`.
pub trait SubProblemExtractable: ProblemTrait + Sized {
    /// Creates a sub-problem containing only the variables that differ between the two parents.
    ///
    /// Fixed variables' contributions (edges to free variables) must be incorporated
    /// into the sub-problem objective so the sub-problem remains self-contained.
    fn extract_sub_problem(&self, sol1: &Self::Solution, sol2: &Self::Solution) -> Self;

    /// Reconstructs a full solution from a sub-problem solution.
    ///
    /// - Variables that agreed in both parents: value taken from `sol1`.
    /// - Variables that differed (free variables): value taken from `sub_solution`.
    fn lift_solution(
        &self,
        sol1: &Self::Solution,
        sol2: &Self::Solution,
        sub_solution: &Self::Solution,
    ) -> Self::Solution;
}
