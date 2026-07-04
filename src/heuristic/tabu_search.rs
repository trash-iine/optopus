use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::SearchState;
use crate::trait_defs::{EnabledTabu, MoveToNeighbor, ProblemTrait, Rankable, rank_cmp};

/// Tabu search heuristic.
///
/// At each iteration the best non-tabu move in the neighborhood is selected and applied.
/// A move is allowed even if it is tabu when it satisfies the aspiration criterion:
/// the resulting solution would be better than the current best.
///
/// After each applied move, the move is added to the tabu map with a tenure sampled
/// uniformly from the range `tabu_tenure = (min, max)`.
///
/// To use this heuristic, the neighbor type must implement [`EnabledTabu`] which defines how to manage the tabu map and tenure.
///
/// # References
///
/// - Glover, F. "Future Paths for Integer Programming and Links to Artificial Intelligence."
///   *Computers & Operations Research*, 13(5), 533-549, 1986.
///   [DOI](https://doi.org/10.1016/0305-0548(86)90048-1)
/// - Glover, F. "Tabu Search — Part I." *ORSA Journal on Computing*, 1(3), 190-206, 1989.
///   [DOI](https://doi.org/10.1287/ijoc.1.3.190)
pub struct TabuSearch<N>
where
    N: Clone + EnabledTabu,
{
    pub stop_condition: StopCondition,
    /// Tabu tenure range `(min, max)` in iterations.
    pub tabu_tenure: (u64, u64),
    tabu_map: N::TabuMap,
}

impl<N> TabuSearch<N>
where
    N: Clone + EnabledTabu,
{
    pub fn new(
        stop_condition: StopCondition,
        tabu_tenure: (u64, u64),
        tabu_map: Option<N::TabuMap>,
    ) -> Self {
        if tabu_tenure.0 > tabu_tenure.1 {
            panic!(
                "Invalid tabu tenure range: left side should be smaller than or equal to the right side ({} <= {})",
                tabu_tenure.0, tabu_tenure.1
            );
        }
        Self {
            stop_condition,
            tabu_tenure,
            tabu_map: tabu_map.unwrap_or(N::TabuMap::default()),
        }
    }

    /// Returns a shared reference to the tabu map.
    pub fn borrow_tabu_map(&self) -> &N::TabuMap {
        &self.tabu_map
    }

    /// Returns a mutable reference to the tabu map.
    pub fn borrow_mut_tabu_map(&mut self) -> &mut N::TabuMap {
        &mut self.tabu_map
    }

    /// Takes ownership of the tabu map, replacing it with its default value.
    pub fn take_tabu_map(&mut self) -> N::TabuMap {
        std::mem::take(&mut self.tabu_map)
    }

    /// Replaces the tabu map with the given value.
    pub fn set_tabu_map(&mut self, tabu_map: N::TabuMap) {
        self.tabu_map = tabu_map;
    }
}

impl<P, N> Heuristic<P> for TabuSearch<N>
where
    P: ProblemTrait,
    N: MoveToNeighbor<P> + Clone + EnabledTabu + Rankable,
{
    fn clear(&mut self) {
        self.tabu_map = N::TabuMap::default();
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    /// When every move is tabu and none satisfies the aspiration criterion, the
    /// iteration is counted as rejected (with a warning) rather than erroring —
    /// the tabu map will eventually expire entries and unblock the search.
    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        // `max_by(rank_cmp)` returns the last tied-best element — the same move
        // the previous `filter_best(..).pop()` selected — without collecting
        // the tie set into a Vec on every iteration.
        let best_move = N::iter(state.instance, &state.solution)
            .filter(|n| {
                // Accept a tabu move if it satisfies the aspiration criterion
                n.is_move_enabled(&self.tabu_map, state.iteration)
                    || state.is_neighbor_better_than_best(n)
            })
            .max_by(rank_cmp);

        if let Some(best_move) = best_move {
            best_move.add_to_tabu_map(&mut self.tabu_map, state.iteration, self.tabu_tenure);
            state.apply(&best_move)?;
        } else {
            tracing::warn!("No best move found");
            state.progress_iteration();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{MaxCut, MaxCutFlipNeighbor};
    use crate::search_state::SearchState;

    fn small_maxcut() -> MaxCut {
        MaxCut::from_edges([
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
        ])
    }

    #[test]
    fn tabu_search_improves_and_respects_budget() {
        let mc = small_maxcut();
        let mut state = SearchState::new_with_seed(&mc, 42);
        let initial_obj = state.best_solution.objective;

        let mut ts =
            TabuSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(200), (1, 3), None);
        ts.run(&mut state).unwrap();

        assert!(state.best_solution.objective >= initial_obj);
        assert_eq!(state.iteration, 200);
        assert_eq!(state.iteration, state.n_accepted + state.n_rejected);
    }

    #[test]
    fn tabu_search_progresses_when_all_moves_are_tabu() {
        // A huge tenure makes every vertex tabu after its first flip; without the
        // aspiration criterion or the reject path this would loop or error.
        let mc = small_maxcut();
        let mut state = SearchState::new_with_seed(&mc, 42);

        let mut ts = TabuSearch::<MaxCutFlipNeighbor>::new(
            StopCondition::iterations(50),
            (10_000, 10_001),
            None,
        );
        ts.run(&mut state).unwrap();
        assert_eq!(state.iteration, 50);
    }

    #[test]
    #[should_panic(expected = "Invalid tabu tenure range")]
    fn tabu_search_panics_on_inverted_tenure() {
        TabuSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(10), (5, 1), None);
    }

    #[test]
    fn tabu_search_clear_resets_tabu_map() {
        let mc = small_maxcut();
        let mut state = SearchState::new_with_seed(&mc, 42);
        let mut ts =
            TabuSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(10), (5, 10), None);
        ts.run(&mut state).unwrap();
        ts.clear();
        assert!(ts.borrow_tabu_map().is_empty());
    }
}
