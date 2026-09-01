use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::SearchState;
use crate::trait_defs::{EnabledTabu, MoveToNeighbor, ProblemTrait, Rankable, rank_cmp};
use std::marker::PhantomData;

/// Tabu search heuristic.
///
/// At each iteration the best non-tabu move in the neighborhood is selected and applied.
/// A move is allowed even if it is tabu when it satisfies the aspiration criterion:
/// the resulting solution would be better than the current best.
///
/// After each applied move, the move is added to the tabu map with a tenure sampled
/// uniformly from the range `tabu_tenure = (min, max)`.
///
/// The map itself lives on the [`SearchState`], not here: the state is what
/// applies a move, so the state is what records it. This heuristic owns only the
/// tenure, which it installs on the state at the top of every iteration. That is
/// also why it has no `clear` — a fresh state (or a sub-run clone, which every
/// meta-heuristic makes) already starts with no prohibitions, and
/// [`SearchState::reset_tabu`] drops them on demand.
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
    tabu_tenure: (u64, u64),
    _neighbor: PhantomData<N>,
}

impl<N> TabuSearch<N>
where
    N: Clone + EnabledTabu,
{
    /// # Panics
    ///
    /// Panics if `tabu_tenure.0 > tabu_tenure.1` (an empty range).
    pub fn new(stop_condition: StopCondition, tabu_tenure: (u64, u64)) -> Self {
        crate::common::tabu::assert_valid_tenure(tabu_tenure);
        Self {
            stop_condition,
            tabu_tenure,
            _neighbor: PhantomData,
        }
    }

    /// Tabu tenure range `(min, max)` in iterations.
    pub fn tabu_tenure(&self) -> (u64, u64) {
        self.tabu_tenure
    }
}

impl<P, N> Heuristic<P> for TabuSearch<N>
where
    P: ProblemTrait,
    N: MoveToNeighbor<P> + Clone + EnabledTabu + Rankable,
{
    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    /// When every move is tabu and none satisfies the aspiration criterion, the
    /// iteration is counted as rejected (with a warning) rather than erroring —
    /// the tabu map will eventually expire entries and unblock the search.
    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        state.set_tabu_tenure(self.tabu_tenure);

        // `max_by(rank_cmp)` returns the last tied-best element — the same move
        // the previous `filter_best(..).pop()` selected — without collecting
        // the tie set into a Vec on every iteration.
        let best_move = N::iter(state.instance, &state.solution)
            .filter(|n| {
                // Accept a tabu move if it satisfies the aspiration criterion
                state.tabu_allows(n) || state.is_neighbor_better_than_best(n)
            })
            .max_by(rank_cmp);

        if let Some(best_move) = best_move {
            // A move type that implements `EnabledTabu` but leaves
            // `MoveToNeighbor::tabu_policy` at its default records nothing, so
            // this search would degrade into a plain best-move descent without
            // ever saying so. One check per iteration, against the move that is
            // about to be applied.
            state.require_tabu_policy(&best_move)?;
            // The move is recorded by `apply`, at the iteration it was made on.
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

        let mut ts = TabuSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(200), (1, 3));
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

        let mut ts =
            TabuSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(50), (10_000, 10_001));
        ts.run(&mut state).unwrap();
        assert_eq!(state.iteration, 50);
    }

    #[test]
    #[should_panic(expected = "Invalid tabu tenure range")]
    fn tabu_search_panics_on_inverted_tenure() {
        TabuSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(10), (5, 1));
    }

    /// The prohibitions the run left behind live on the state, and
    /// [`SearchState::reset_tabu`] frees every move — the observable property a
    /// new episode depends on, which used to be `TabuSearch::clear`'s job.
    #[test]
    fn reset_tabu_frees_what_a_run_forbade() {
        let mc = small_maxcut();
        let mut state = SearchState::new_with_seed(&mc, 42);
        let mut ts = TabuSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(10), (5, 10));
        ts.run(&mut state).unwrap();

        let blocked = |state: &SearchState<'_, MaxCut>| {
            (0..mc.graph.len()).any(|i| !state.tabu_allows(&MaxCutFlipNeighbor { i, gain: 0.0 }))
        };
        assert!(
            blocked(&state),
            "the run must leave at least one vertex tabu"
        );

        state.reset_tabu();
        assert!(!blocked(&state), "reset_tabu must free every vertex");
    }

    /// A sub-run starts from no prohibitions whichever clone type made it —
    /// what every meta-heuristic relies on to isolate a phase, and what
    /// `Heuristic::clear` used to provide.
    #[test]
    fn a_sub_run_starts_with_an_empty_tabu_memory() {
        use crate::search_state::SearchStateCloneType;

        let mc = small_maxcut();
        let mut state = SearchState::new_with_seed(&mc, 42);
        let mut ts = TabuSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(10), (50, 50));
        ts.run(&mut state).unwrap();

        for clone_type in [
            SearchStateCloneType::Simple,
            SearchStateCloneType::ClearBest,
            SearchStateCloneType::StartBest,
        ] {
            let sub = state.clone_for_new_run(clone_type.clone());
            for i in 0..mc.graph.len() {
                assert!(
                    sub.tabu_allows(&MaxCutFlipNeighbor { i, gain: 0.0 }),
                    "vertex {i} is still tabu in a {clone_type:?} sub-run"
                );
            }
            assert_eq!(
                sub.tabu_tenure(),
                (0, 0),
                "the tenure is the child's to set"
            );
        }
    }
}
