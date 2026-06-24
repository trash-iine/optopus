/// Is for moves that support a tabu list mechanism.
///
/// A move is considered *enabled* if it is not currently forbidden by the tabu map.
/// After a move is applied, it can be added to the tabu map with a given tenure.
pub trait EnabledTabu: Clone {
    /// The data structure used to store the tabu list (e.g., `HashMap<usize, u64>`).
    type TabuMap: Default;

    /// Returns `true` if this move is allowed under the current tabu map at the given iteration.
    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool;

    /// Adds this move to the tabu map with a randomly sampled tenure in the given range.
    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
    );
}
