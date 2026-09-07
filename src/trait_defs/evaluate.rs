/// An objective value, or a change in one, with the optimization direction
/// attached.
///
/// `T` is the numeric type (commonly `f64`, but any type is accepted).
///
/// Choose the variant that matches your problem's optimization direction:
/// - [`Evaluable::Maximize`]: the objective is being maximized (higher is better).
/// - [`Evaluable::Minimize`]: the cost is being minimized (lower is better).
///
/// Attaching the direction to the value is what lets a search that computes
/// with an objective stay direction-agnostic. Without it every such caller
/// would branch on a flag, and one that forgot would still produce plausible
/// numbers.
#[derive(Clone, Copy, Debug)]
pub enum Evaluable<T = f64> {
    /// Change in a maximized objective (positive = improvement, negative = worsening).
    Maximize(T),
    /// Change in a minimized cost (positive = worsening, negative = improvement).
    Minimize(T),
}

impl Evaluable<f64> {
    /// The value with the direction applied, so that lower is better whichever
    /// way the problem optimizes.
    ///
    /// For a move's delta this is how much worse the move makes things, which
    /// is what [`worsening_amount`](Self::worsening_amount) names. For a
    /// solution's objective it is what a physical analogy calls the energy, and
    /// what population annealing weights a replica by. Same arithmetic either
    /// way, and the reason both readings can share one type.
    pub fn minimized(self) -> f64 {
        match self {
            Evaluable::Maximize(value) => -value,
            Evaluable::Minimize(value) => value,
        }
    }

    /// Returns the worsening amount: positive when the move degrades the objective.
    ///
    /// The same number as [`minimized`](Self::minimized), named for the reading
    /// that suits a delta. Used by `boltzmann_accept` to compute
    /// `exp(-worsening / T)`.
    pub fn worsening_amount(self) -> f64 {
        self.minimized()
    }
}

/// Reports an objective value, or a change in one, with the direction attached.
///
/// Implemented by two kinds of type, and the difference is what the value
/// means rather than anything about the trait:
///
/// - A **move** reports the change applying it would make. This is what
///   [`SimulatedAnnealing`](crate::heuristic::SimulatedAnnealing),
///   `LateAcceptanceHillClimbing` and `RlSearch` need.
/// - A **solution** reports its own objective. This is what a search that
///   computes with the objective needs rather than merely comparing them:
///   [`PopulationAnnealing`](crate::heuristic::PopulationAnnealing) weights a
///   replica by `exp(-Δβ (E_j - E_min))`, and a difference of objectives has no
///   meaning until the direction is fixed.
///
/// [`Rankable`](super::Rankable) answers "is this better?" and is enough for a
/// search that only compares. Implement this as well where the number itself is
/// used.
///
/// `T` is the numeric type returned (default `f64`). Use `T = f64` for
/// compatibility with [`boltzmann_accept`](crate::heuristic::boltzmann_accept).
pub trait Evaluate<T = f64> {
    fn evaluate(&self) -> Evaluable<T>;
}
