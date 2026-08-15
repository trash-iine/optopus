//! Mapping one problem instance onto another, with solutions crossing in both
//! directions.

use super::ProblemTrait;

/// The solution type of a reduction's source problem.
pub type SourceSolution<R> = <<R as ProblemReduction>::Source as ProblemTrait>::Solution;

/// The solution type of a reduction's target problem.
pub type TargetSolution<R> = <<R as ProblemReduction>::Target as ProblemTrait>::Solution;

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
    fn project(&self, sol: &SourceSolution<Self>) -> TargetSolution<Self>;

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
        base: &SourceSolution<Self>,
        sol: &TargetSolution<Self>,
    ) -> SourceSolution<Self>;
}

/// The map's own contract, exercised through the one implementation the core
/// library has: [`MaxCutKernel`](crate::problem::MaxCutKernel).
///
/// Nothing here touches a `SearchState` — that is the point of the split, and
/// the search-state half of the crossing is tested where it lives, in
/// [`crate::search_state`].
#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{MaxCut, MaxCutKernel};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    /// A sparse instance, i.e. one the kernel rules actually reduce.
    fn reducible_instance(seed: u64, n: usize) -> MaxCut {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if rng.random_bool(2.5 / n as f64) {
                    edges.push((i, j, 1.0));
                }
            }
        }
        edges.push((n - 1, n - 2, 1.0));
        MaxCut::from_edges(edges)
    }

    /// A kernel warm start keeps the incumbent's decisions on the vertices the
    /// kernel still has, and re-derives the removed ones — which the rules
    /// choose optimally, so the round trip can only improve the cut.
    #[test]
    fn a_round_trip_keeps_the_kernel_vertices_and_never_loses_cut() {
        let mut rng = SmallRng::seed_from_u64(7);
        for seed in 0..5u64 {
            let mc = reducible_instance(seed, 200);
            let kernel = MaxCutKernel::reduce(&mc);
            let original = mc.new_solution(&mut rng);

            let projected = ProblemReduction::project(&kernel, &original);
            let back = ProblemReduction::lift(&kernel, &mc, &original, &projected);

            assert_eq!(
                ProblemReduction::project(&kernel, &back).x,
                projected.x,
                "the kernel's own vertices must survive the round trip"
            );
            assert!(
                back.objective >= original.objective,
                "re-deriving removed vertices must not lose cut: {} < {}",
                back.objective,
                original.objective
            );
        }
    }
}
