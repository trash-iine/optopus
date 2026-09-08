use super::rankable::Rankable;

/// Is a combinatorial optimization problem.
///
/// Implement this trait to define a custom problem. You must specify the associated
/// [`ProblemTrait::Solution`] type and provide a method to generate a random initial solution.
///
/// # Example
///
/// ```rust,ignore
/// struct MyProblem { /* ... */ }
/// struct MySolution { value: f64 }
///
/// // Evaluate is what a solution states; Rankable follows from it.
/// impl Evaluate for MySolution {
///     fn evaluate(&self) -> Evaluable<f64> { Evaluable::Maximize(self.value) }
/// }
/// impl ProblemTrait for MyProblem {
///     type Solution = MySolution;
///     fn new_solution(&self, _rng: &mut impl rand::Rng) -> MySolution {
///         MySolution { value: 0.0 }
///     }
/// }
/// ```
pub trait ProblemTrait {
    type Solution: Clone + Rankable;
    fn new_solution(&self, rng: &mut impl rand::Rng) -> Self::Solution;
}
