//! The structure ruin-and-recreate reads: elements, containers, and what it
//! costs to put one in the other.

use super::{Evaluate, ProblemTrait};

/// A problem whose solutions assign elements to containers, where the
/// containers compete for a finite resource.
///
/// This is what makes ruin-and-recreate work, and regret-k in particular.
/// Removing part of the solution leaves a pool of unplaced elements, and
/// putting one back has a cost that depends on which container takes it.
/// Regret measures how much worse an element gets once its best container is
/// taken. That second dimension is the whole point. With a single container
/// the second-best place is the slot next door, a gap of nearly zero for every
/// element, and regret-2 becomes indistinguishable from greedy.
///
/// # Which problems fit
///
/// A large family of NP-hard problems is exactly this shape. Vehicle routing
/// (containers are vehicles, the resource is capacity), bin packing (bins), the
/// generalized assignment problem (agents), multiple knapsack, capacitated
/// facility location, parallel machine scheduling, graph colouring (colour
/// classes). What does not fit is a problem with one container, such as a
/// single TSP tour, or with two fixed values per element, such as a binary
/// problem where `regret = |cost(true) − cost(false)| = |gain|` carries nothing
/// greedy did not already have.
///
/// # Containers may come and go
///
/// [`num_buckets`](Self::num_buckets) reads the current container count from
/// the partial solution rather than being a constant of the instance, because
/// for several of the problems above it is not one. Bin packing opens a bin
/// when the open ones are full, and graph colouring adds a colour class. A
/// problem that needs somewhere new to put an element reports the empty
/// container it would use, so the repair operators never have to ask for one.
/// Vehicle routing satisfies this without doing anything special, since its
/// fleet is fixed and an unused vehicle is an empty route.
///
/// # The working representation
///
/// [`Partial`](Self::Partial) exists because destroy and repair together touch
/// the solution many times per iteration, and maintaining a `Solution`, with
/// its caches and its objective, through every one of them is wasteful. A
/// problem keeps whatever makes the edits cheap (vehicle routing keeps its
/// routes plus per-route loads and position indexes) and pays the conversion
/// twice per iteration instead.
pub trait Ruinable: ProblemTrait {
    /// What gets removed and re-inserted: a customer, an item, a job, a vertex.
    type Element: Copy + Eq;

    /// The representation destroy and repair edit.
    type Partial;

    /// Reads a solution into the working representation.
    fn to_partial(&self, sol: &Self::Solution) -> Self::Partial;

    /// Converts back, recomputing whatever the solution caches.
    fn finish(&self, partial: Self::Partial) -> Self::Solution;

    // -----------------------------------------------------------------
    // destroy
    // -----------------------------------------------------------------

    /// Collects every currently placed element into `out`, which is cleared
    /// first.
    fn elements(&self, partial: &Self::Partial, out: &mut Vec<Self::Element>);

    /// Rebuilds `partial` from `sol`, reusing whatever buffers it already
    /// holds.
    ///
    /// Ruin and recreate converts a solution into a partial once per iteration
    /// and throws it away at the end, so a working representation that owns
    /// index buffers reallocates them on every one. The default builds a fresh
    /// partial, which is correct everywhere. Override it where the
    /// representation owns anything worth keeping.
    fn refresh_partial(&self, partial: &mut Self::Partial, sol: &Self::Solution) {
        *partial = self.to_partial(sol);
    }

    /// How many elements are currently placed.
    ///
    /// Separate from [`elements`](Self::elements) because a caller often needs
    /// only the count, and listing them costs a pass and a buffer. Ruin and
    /// recreate asks for the count once per iteration to size the ruin, and the
    /// destroy operator that follows lists them anyway.
    ///
    /// The default lists them, which is correct for any implementation and
    /// wasteful for most. Override it where the count is cheaper than the list,
    /// as it is wherever the elements sit in a handful of containers.
    fn num_elements(&self, partial: &Self::Partial) -> usize {
        let mut buffer = Vec::new();
        self.elements(partial, &mut buffer);
        buffer.len()
    }

    /// Removes every element of `set` from wherever it currently sits.
    fn remove_all(&self, partial: &mut Self::Partial, set: &[Self::Element]);

    /// What is saved by taking `element` out of its current place, the detour
    /// it costs in vehicle-routing terms.
    ///
    /// Drives worst-removal. Not defaulted, since a default of `0.0` would make
    /// worst-removal silently identical to random-removal, which is the kind
    /// of degradation that does not show up as a failure anywhere.
    fn removal_gain(&self, partial: &Self::Partial, element: Self::Element) -> f64;

    /// How alike two elements are, where smaller is more alike.
    ///
    /// Drives Shaw removal, whose premise is that removing a cluster of
    /// similar elements opens up a rearrangement that removing scattered ones
    /// does not. Depends on the instance only, not on the current placement,
    /// so it can be computed without a `Partial`.
    ///
    /// Not defaulted, for the same reason as
    /// [`removal_gain`](Self::removal_gain).
    fn relatedness(&self, a: Self::Element, b: Self::Element) -> f64;

    // -----------------------------------------------------------------
    // repair
    // -----------------------------------------------------------------

    /// How many containers an element may currently be placed in.
    ///
    /// See the note above: a problem whose containers are created on demand
    /// includes the next empty one here.
    fn num_buckets(&self, partial: &Self::Partial) -> usize;

    /// How many distinct positions `bucket` offers.
    ///
    /// `1` when the container is a set rather than a sequence (a bin, an agent,
    /// a colour class); `len + 1` when the order inside it matters (a route).
    fn num_places(&self, partial: &Self::Partial, bucket: usize) -> usize;

    /// The cost of placing `element` at `(bucket, place)`.
    ///
    /// Fold any resource violation in here, weighted so that a feasible
    /// placement always wins. That is what lets the repair operators assume a
    /// placement is always available, instead of having to handle "every
    /// container is full" as a special case. Vehicle routing does it with
    /// `Vrp::penalty_weight`, the same weight its objective already charges
    /// overload at.
    fn insertion_cost(
        &self,
        partial: &Self::Partial,
        bucket: usize,
        place: usize,
        element: Self::Element,
    ) -> f64;

    /// Places `element` at `(bucket, place)`.
    fn insert(
        &self,
        partial: &mut Self::Partial,
        bucket: usize,
        place: usize,
        element: Self::Element,
    );

    /// The direction-normalized objective of `partial`, without materializing
    /// a full [`Solution`](ProblemTrait::Solution).
    ///
    /// This is what lets a search reject a candidate without ever converting
    /// it: [`Alns`](crate::heuristic::Alns) calls [`finish`](Self::finish)
    /// only for the candidate it accepts, and every other iteration reads this
    /// instead. On a problem whose acceptance rate falls as the temperature
    /// cools -- which is most of them, most of a run -- that turns the
    /// majority of iterations from a full solution rebuild into a cached
    /// read.
    ///
    /// <div class="warning">
    /// The default implementation clones `partial` and calls
    /// <code>finish</code> + <code>Evaluate</code>, which is correct
    /// but pays the exact cost this method exists to avoid -- a problem that
    /// never overrides it gets no benefit from calling it instead of
    /// <code>finish</code> directly. Override it with the running total a
    /// problem already tracks while destroy/repair edit the partial (see
    /// [`Vrp`](crate::problem::vrp::Vrp), which maintains it through
    /// `removal_gain` / `insertion_cost` deltas and the anchored descent's own
    /// running totals).
    ///
    /// When this default is invoked at runtime, a one-shot
    /// <code>tracing::warn!</code> is emitted per concrete `Self` type (via
    /// <code>OnceLock</code>).
    /// </div>
    fn partial_energy(&self, partial: &Self::Partial) -> f64
    where
        Self::Solution: Evaluate,
        Self::Partial: Clone,
    {
        static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        WARNED.get_or_init(|| {
            tracing::warn!(
                problem_type = std::any::type_name::<Self>(),
                "Using the default clone+finish implementation of \
                 Ruinable::partial_energy, which rebuilds a full Solution on \
                 every call. Override it with the running total the problem \
                 already tracks."
            );
        });
        self.finish(partial.clone()).evaluate().minimized()
    }
}

/// A local search that repairs a partial solution around the places it was
/// just disturbed.
///
/// Separate from [`Heuristic`](crate::heuristic::Heuristic) because that trait
/// has nowhere to say where to look. Ruin-and-recreate re-inserts elements
/// greedily and leaves the neighbourhood of each insertion locally poor, so
/// something has to clean up after it, but a full sweep is O(n) per iteration
/// and the search runs thousands of iterations. The caller knows exactly which
/// elements it moved, and handing that over is the difference between a descent
/// that is affordable per iteration and one that is not.
///
/// Optional, since a problem with no such local search runs plain
/// ruin-and-recreate. For vehicle routing it is not optional in practice,
/// greedy re-insertion never fixes the edges its own choices spoil.
pub trait LocalRepair<P: Ruinable> {
    /// Improves `partial` in the neighbourhood of `anchors`.
    fn repair_around(
        &mut self,
        prob: &P,
        partial: &mut P::Partial,
        anchors: &[P::Element],
        rng: &mut rand::rngs::SmallRng,
    );
}
