//! Mapping one problem instance onto another, with solutions crossing in both
//! directions.

use super::ProblemTrait;

/// A map from an instance of one problem to an instance of another, with a
/// solution mapping in each direction.
///
/// The exact kernelization of a MaxCut instance
/// ([`MaxCutKernel`](crate::problem::MaxCutKernel)) is the shape this
/// describes: something smaller to search, a way in for a warm start, and a way
/// back out. `Source` and `Target` are separate associated types rather than
/// one because a reduction need not stay inside its problem — a penalised
/// objective, for instance, can reduce into a
/// [`Qubo`](crate::problem::Qubo) when its penalty term is quadratic.
///
/// # What this trait is not
///
/// It is only the map. Running a heuristic on the target and folding the result
/// back is a *search-state* operation and lives there:
/// [`SearchState::open_reduction`](crate::search_state::SearchState::open_reduction)
/// draws the sub-state's seed and projects the warm start, and
/// [`close_reduction`](crate::search_state::SearchState::close_reduction)
/// merges the sub-run's counters and installs the lifted result. Doing that by
/// hand — or in the other order — is where copies drift apart, silently, in
/// `iteration` / `n_accepted` / `best_iteration` rather than in the objective,
/// so it belongs on the type that owns the state, not here. `close_reduction`
/// is where that reasoning is written down; it is not repeated at the call
/// sites.
///
/// # Exactness
///
/// The trait does not require the map to preserve the objective; an approximate
/// reduction has the same shape. A caller may require it, and the current one
/// does: `MaxCutKernel` guarantees
///
/// ```text
/// kernel_cut(y) + offset == original_cut(lift(y))    for every y
/// ```
///
/// — for *every* `y`, not only optimal ones, which is what lets a heuristic be
/// stopped at any point and lifted. Stating that as the implementation's
/// obligation rather than the trait's keeps the plumbing reusable by a map that
/// only preserves optima, or none.
pub trait ProblemReduction {
    /// The problem being mapped *from*.
    type Source: ProblemTrait;
    /// The problem being mapped *to*.
    type Target: ProblemTrait;

    /// The reduced instance — what a heuristic actually searches.
    fn target(&self) -> &Self::Target;

    /// Maps a solution of the source onto the target, for use as a warm start.
    fn project(
        &self,
        sol: &<Self::Source as ProblemTrait>::Solution,
    ) -> <Self::Target as ProblemTrait>::Solution;

    /// Maps a solution of the target back onto the source.
    ///
    /// `base` supplies whatever this map dropped. When the target does not
    /// cover the source's whole variable index space, the uncovered positions
    /// keep their value from `base` rather than defaulting.
    ///
    /// `source` is a parameter rather than a field of the reduction because a
    /// solution generally carries incremental caches that only the instance can
    /// rebuild, and holding a `&Self::Source` is not open to every
    /// implementation: `MaxCutKernel` is stored in a heuristic that outlives
    /// any one `run_once`, so it cannot borrow that call's `state.instance`.
    fn lift(
        &self,
        source: &Self::Source,
        base: &<Self::Source as ProblemTrait>::Solution,
        sol: &<Self::Target as ProblemTrait>::Solution,
    ) -> <Self::Source as ProblemTrait>::Solution;
}
