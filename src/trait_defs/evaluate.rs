/// The change in objective value resulting from a move, with explicit optimization direction.
///
/// `T` is the numeric type of the change (commonly `f64`, but any type is accepted).
///
/// Choose the variant that matches your problem's optimization direction:
/// - [`Evaluable::Maximize`]: the objective is being maximized (positive = improvement).
/// - [`Evaluable::Minimize`]: the cost is being minimized (positive = worsening).
#[derive(Clone, Copy, Debug)]
pub enum Evaluable<T = f64> {
    /// Change in a maximized objective (positive = improvement, negative = worsening).
    Maximize(T),
    /// Change in a minimized cost (positive = worsening, negative = improvement).
    Minimize(T),
}

impl Evaluable<f64> {
    /// Returns the worsening amount: positive when the move degrades the objective.
    ///
    /// Used internally by `boltzmann_accept` to compute `exp(-worsening / T)`.
    pub fn worsening_amount(self) -> f64 {
        match self {
            Evaluable::Maximize(gain) => -gain,
            Evaluable::Minimize(cost_delta) => cost_delta,
        }
    }
}

/// Implemented by neighbor types that can evaluate their objective change for SA acceptance.
///
/// `T` is the numeric type returned (default `f64`). Use `T = f64` for compatibility
/// with [`crate::heuristic::SimulatedAnnealing`] and [`crate::heuristic::boltzmann_accept`].
pub trait Evaluate<T = f64> {
    fn evaluate(&self) -> Evaluable<T>;
}
