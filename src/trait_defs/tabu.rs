use crate::common::TabuMemory;
use rand::rngs::SmallRng;

/// Is for moves that support a tabu list mechanism.
///
/// A move is considered *enabled* if it is not currently forbidden by the
/// search's [`TabuMemory`]. After a move is applied, it records itself there.
///
/// What a move decides is its **policy**: which keys it has to find free, which
/// keys applying it forbids, and whether those two sets are the same. They are
/// not always — a VRP relocate asks whether a customer may enter its
/// destination route and forbids the route it just left, so that the customer
/// cannot be moved straight back. What a move does *not* decide is where the
/// prohibitions are kept: that is one [`TabuMemory`] on the
/// [`SearchState`](crate::search_state::SearchState), which is what lets two
/// move types over the same [`TabuKey`](crate::common::TabuKey) shape see each
/// other's entries.
///
/// The state is not generic over the move, but the methods that touch the
/// memory are: [`tabu_allows`](crate::search_state::SearchState::tabu_allows)
/// and [`record_tabu`](crate::search_state::SearchState::record_tabu) take
/// `N: EnabledTabu`, so the policy is resolved statically and inlines into a
/// neighborhood scan. Implementing this trait is the whole of opting in, and a
/// move that does not implement it simply cannot be passed to those methods —
/// the mistake is a compile error, not a silent no-op.
pub trait EnabledTabu {
    /// Returns `true` if this move is allowed under the current tabu memory at
    /// the given iteration.
    fn is_move_enabled(&self, tabu: &TabuMemory, iteration: u64) -> bool;

    /// Forbids what this move touches, for a tenure the memory draws from its
    /// own [`tenure`](TabuMemory::tenure) using `rng` (the state passes its own,
    /// so seeded runs stay reproducible).
    fn add_to_tabu_map(&self, tabu: &mut TabuMemory, iteration: u64, rng: &mut SmallRng);
}
