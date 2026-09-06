use crate::common::TabuMemory;
use rand::rngs::SmallRng;

/// Is for moves that support a tabu list mechanism.
///
/// A move is considered enabled if it is not currently forbidden by the
/// search's [`TabuMemory`]. After a move is applied, it records itself there.
///
/// What a move decides is its **policy**: which keys it has to find free, which
/// keys applying it forbids, and whether those two sets are the same. They are
/// not always — a VRP relocate asks whether a customer may enter its
/// destination route and forbids the route it just left, so that the customer
/// cannot be moved straight back. What a move does not decide is where the
/// prohibitions are kept.
///
/// The state is not generic over the move, but the methods that touch the
/// memory are: [`tabu_allows`](crate::search_state::SearchState::tabu_allows)
/// and [`record_tabu`](crate::search_state::SearchState::record_tabu) take
/// `N: EnabledTabu`, so the policy is resolved statically and inlines into a
/// neighborhood scan. Implementing this trait is the whole of opting in, and a
/// move that does not implement it simply cannot be passed to those methods,
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

#[cfg(test)]
mod tests {
    use crate::trait_defs::MoveToNeighbor;

    /// Implementing [`EnabledTabu`](super::EnabledTabu) does nothing on its own
    /// — [`SearchState::apply`](crate::search_state::SearchState::apply) reaches
    /// a move's policy through [`MoveToNeighbor::tabu_policy`], whose default is
    /// `None`. A built-in move that implements the one and forgets the other
    /// would degrade [`TabuSearch`](crate::heuristic::TabuSearch) to a plain
    /// best-move search without failing to compile, so every built-in move is
    /// checked here.
    #[test]
    fn every_built_in_move_hands_over_its_tabu_policy() {
        use crate::problem::*;

        macro_rules! assert_policy {
            ($($problem:ty => $mv:expr),+ $(,)?) => {
                $(
                    assert!(
                        MoveToNeighbor::<$problem>::tabu_policy(&$mv).is_some(),
                        "{} does not override MoveToNeighbor::tabu_policy",
                        std::any::type_name_of_val(&$mv),
                    );
                )+
            };
        }

        assert_policy! {
            MaxCut => MaxCutFlipNeighbor { i: 0, gain: 0.0 },
            MaxCut => MaxCutSwapNeighbor { i: 0, j: 1, gain: 0.0 },
            Qubo => QuboFlipNeighbor { i: 0, gain: 0 },
            Qubo => QuboSwapNeighbor { i: 0, j: 1, gain: 0 },
            Sat => SatFlipNeighbor { i: 0, gain: 0 },
            Sat => SatSwapNeighbor { i: 0, j: 1, gain: 0 },
            VertexCover => VertexCoverFlipNeighbor { i: 0, gain: 0 },
            VertexCover => VertexCoverSwapNeighbor { i: 0, j: 1, gain: 0 },
            TspWithCoordinates => TspTwoOptNeighbor { i: 0, j: 1, gain: 0.0 },
            TspWithCoordinates => TspRelocateNeighbor { pos: 0, ins: 2, gain: 0.0 },
            JobShopScheduling => JobShopSwapNeighbor { i: 0, gain: 0.0 },
            JobShopScheduling => JobShopRelocateNeighbor { from: 0, to: 1, gain: 0.0 },
            Vrp => VrpRelocateNeighbor {
                from_r: 0, from_i: 0, to_r: 1, to_i: 0,
                customer: 1, gain: 0.0, overload_delta: 0,
            },
            Vrp => VrpSwapNeighbor {
                r1: 0, i1: 0, r2: 1, i2: 0,
                c1: 1, c2: 2, gain: 0.0, overload_delta: 0,
            },
            Vrp => VrpTwoOptNeighbor { r: 0, p: 0, q: 1, gain: 0.0 },
            FormulaProblem => FormulaFlipNeighbor { i: 0, gain: 0.0 },
            FormulaProblem => FormulaSwapNeighbor { i: 0, j: 1, gain: 0.0 },
        }
    }
}
