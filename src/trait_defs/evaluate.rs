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
    /// Read as a move's delta this is how much worse the move makes things,
    /// positive when it degrades the objective, which is what
    /// `boltzmann_accept` puts in `exp(-minimized / T)`. Read as a solution's
    /// objective it is what a physical analogy calls the energy, and what
    /// population annealing weights a replica by. Same arithmetic either way,
    /// and the reason both readings share one type and one method.
    pub fn minimized(self) -> f64 {
        match self {
            Evaluable::Maximize(value) => -value,
            Evaluable::Minimize(value) => value,
        }
    }

    /// Whether this delta, applied to the solution `from`, lands better than
    /// the solution `over`.
    ///
    /// The three values are read through [`minimized`](Self::minimized), so the
    /// optimization direction is stated once, by the `Evaluate` impls, and the
    /// comparison itself does not restate it. This is the body every
    /// [`MoveToNeighbor::move_to_be_better_than`](crate::trait_defs::MoveToNeighbor::move_to_be_better_than)
    /// override in the crate spells out.
    ///
    /// It agrees with the solution's [`Rankable`](super::Rankable) only where
    /// the solution ranks by its objective, which `#[derive(Rankable)]` with no
    /// `rank` attribute does. An override for a solution ranked by keys
    /// compares in that order instead.
    pub fn improves_over(self, from: Evaluable<f64>, over: Evaluable<f64>) -> bool {
        from.minimized() + self.minimized() < over.minimized()
    }

    /// The number inside, without its direction.
    pub(crate) fn raw(self) -> f64 {
        match self {
            Evaluable::Maximize(v) | Evaluable::Minimize(v) => v,
        }
    }

    /// `value` with the direction of `self`.
    pub(crate) fn with_value(self, value: f64) -> Evaluable<f64> {
        match self {
            Evaluable::Maximize(_) => Evaluable::Maximize(value),
            Evaluable::Minimize(_) => Evaluable::Minimize(value),
        }
    }
}

/// Reports an objective value, or a change in one, with the direction attached.
///
/// Implemented by two kinds of type, and the difference is what the value
/// means rather than anything about the trait:
///
/// - A **move** reports the change applying it would make. This is what
///   [`SimulatedAnnealing`](crate::heuristic::SimulatedAnnealing),
///   `LateAcceptanceHillClimbing` and `ReinforcementLearningSearch` need.
/// - A **solution** reports its own objective. This is what a search that
///   computes with the objective needs rather than merely comparing them:
///   [`PopulationAnnealing`](crate::heuristic::PopulationAnnealing) weights a
///   replica by `exp(-Δβ (E_j - E_min))`, and a difference of objectives has no
///   meaning until the direction is fixed.
///
/// [`Rankable`](super::Rankable) answers "is this better?" and is enough for a
/// search that only compares. Implement this as well where the number itself is
/// used. Where the two agree, `#[derive(Rankable)]` with no `rank` attribute
/// writes the `Rankable` impl from this one.
///
/// `T` is the numeric type returned (default `f64`). Use `T = f64` for
/// compatibility with [`boltzmann_accept`](crate::building_blocks::search::boltzmann_accept).
///
/// # Deriving it
///
/// Where the number is one field, `#[derive(Evaluate)]` writes the impl, and
/// the field is marked `#[objective(minimize)]` or `#[objective(maximize)]`.
/// The field is read as an `f64`, so any numeric field works. On a solution
/// it is the objective and on a move the change the move makes, and every
/// built-in solution and move whose number is a field uses it.
///
/// ```
/// use optopus::prelude::*;
///
/// #[derive(Evaluate, Rankable)]
/// struct Schedule {
///     order: Vec<usize>,
///     #[objective(minimize)]
///     makespan: u32,
/// }
///
/// let s = Schedule { order: vec![1, 0], makespan: 12 };
/// assert!(matches!(s.evaluate(), Evaluable::Minimize(m) if m == 12.0));
/// ```
///
/// Exactly one field is marked, so the number is stated once.
///
/// ```compile_fail
/// use optopus::prelude::*;
///
/// #[derive(Evaluate)]
/// struct Two {
///     #[objective(minimize)]
///     cost: f64,
///     #[objective(minimize)]
///     time: f64,
/// }
/// ```
///
/// The direction is `minimize` or `maximize` and nothing else.
///
/// ```compile_fail
/// use optopus::prelude::*;
///
/// #[derive(Evaluate)]
/// struct Lower {
///     #[objective(min)]
///     cost: f64,
/// }
/// ```
pub trait Evaluate<T = f64> {
    fn evaluate(&self) -> Evaluable<T>;
}
