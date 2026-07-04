use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::SearchState;
use crate::trait_defs::{Evaluate, MoveToNeighbor, ProblemTrait};

/// Late Acceptance Hill Climbing (LAHC) heuristic.
///
/// At each iteration a random neighbor is selected.
/// The move is accepted if the resulting objective is no worse than the objective
/// recorded `history_length` iterations ago. This creates an adaptive threshold
/// that automatically balances exploration and exploitation without temperature tuning.
///
/// Requires the neighbor type to implement [`Evaluate<f64>`].
///
/// # Reference
///
/// Burke, E. K., & Bykov, Y. (2017). The late acceptance hill-climbing heuristic.
/// *European Journal of Operational Research*, 258(1), 70–78.
pub struct LateAcceptanceHillClimbing<N> {
    pub stop_condition: StopCondition,
    pub history_length: usize,
    _neighbor: std::marker::PhantomData<N>,
    /// Circular buffer of "higher-is-better" scores.
    history: Vec<f64>,
    /// Current position in the circular buffer.
    history_index: usize,
    /// Running score of the current solution (higher is always better).
    current_score: f64,
    /// Whether the history has been initialized from the first solution.
    initialized: bool,
}

impl<N> LateAcceptanceHillClimbing<N> {
    /// Create a new [`LateAcceptanceHillClimbing`] heuristic.
    ///
    /// `history_length` controls the trade-off between exploitation and exploration:
    /// - Small values (e.g., 1) behave like hill climbing.
    /// - Large values (e.g., 50000) allow more diversification.
    /// - A robust default across many problem types is 5000.
    /// # Panics
    ///
    /// Panics if `history_length` is 0.
    pub fn new(stop_condition: StopCondition, history_length: usize) -> Self {
        assert!(history_length > 0, "history_length must be at least 1");
        Self {
            stop_condition,
            history_length,
            _neighbor: std::marker::PhantomData,
            history: Vec::new(),
            history_index: 0,
            current_score: 0.0,
            initialized: false,
        }
    }
}

impl<P, N> Heuristic<P> for LateAcceptanceHillClimbing<N>
where
    P: ProblemTrait,
    N: MoveToNeighbor<P> + Evaluate,
{
    fn clear(&mut self) {
        self.history.clear();
        self.history_index = 0;
        self.current_score = 0.0;
        self.initialized = false;
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        // Initialize history buffer on first call
        if !self.initialized {
            self.current_score = 0.0;
            self.history = vec![0.0; self.history_length];
            self.initialized = true;
        }

        let neighbor: N = state.random_neighbor("LateAcceptanceHillClimbing")?;

        // Compute the candidate score (higher is always better).
        // worsening_amount() is positive when the move is worse,
        // so subtracting it gives us a "higher is better" score.
        let delta = neighbor.evaluate();
        let candidate_score = self.current_score - delta.worsening_amount();

        let idx = self.history_index % self.history_length;
        let history_score = self.history[idx];

        // Accept if candidate is no worse than current OR no worse than history
        if candidate_score >= self.current_score || candidate_score >= history_score {
            state.apply(&neighbor)?;
            self.current_score = candidate_score;
        } else {
            state.progress_iteration();
        }

        // Always update history with the current score (after potential acceptance)
        self.history[idx] = self.current_score;
        self.history_index += 1;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::MaxCutFlipNeighbor;
    use crate::problem::max_cut::MaxCut;
    use crate::search_state::SearchState;

    fn small_maxcut() -> MaxCut {
        MaxCut::from_edges([
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (1, 2, 1.0),
            (1, 4, 1.0),
            (2, 5, 1.0),
            (3, 4, 1.0),
            (3, 5, 1.0),
            (4, 5, 1.0),
        ])
    }

    #[test]
    fn lahc_improves_maxcut() {
        let mc = small_maxcut();
        let mut state = SearchState::new(&mc);
        let initial_obj = state.best_solution.objective;

        let mut lahc = LateAcceptanceHillClimbing::<MaxCutFlipNeighbor>::new(
            StopCondition::iterations(10_000),
            100,
        );
        lahc.run(&mut state).unwrap();

        assert!(
            state.best_solution.objective >= initial_obj,
            "LAHC should not worsen the best solution"
        );
        assert!(state.iteration >= 10_000);
    }

    #[test]
    fn lahc_respects_stop_condition() {
        let mc = small_maxcut();
        let mut state = SearchState::new(&mc);

        let mut lahc = LateAcceptanceHillClimbing::<MaxCutFlipNeighbor>::new(
            StopCondition::iterations(500),
            50,
        );
        lahc.run(&mut state).unwrap();

        assert!(state.iteration >= 500);
        assert!(state.iteration <= 600); // some slack for iteration counting
    }

    #[test]
    fn lahc_clear_resets_state() {
        let mc = small_maxcut();
        let mut state = SearchState::new(&mc);

        let mut lahc = LateAcceptanceHillClimbing::<MaxCutFlipNeighbor>::new(
            StopCondition::iterations(100),
            50,
        );
        lahc.run(&mut state).unwrap();

        // After clear, internal state should be reset
        lahc.clear();
        assert!(lahc.history.is_empty());
        assert_eq!(lahc.history_index, 0);
        assert!(!lahc.initialized);
    }

    #[test]
    #[should_panic(expected = "history_length must be at least 1")]
    fn lahc_history_length_zero_panics() {
        LateAcceptanceHillClimbing::<MaxCutFlipNeighbor>::new(StopCondition::iterations(100), 0);
    }

    #[test]
    fn lahc_history_length_one_behaves_like_hc() {
        // With history_length=1, LAHC should behave similarly to hill climbing
        // (only accepts improvements or lateral moves)
        let mc = small_maxcut();
        let mut state = SearchState::new(&mc);

        let mut lahc = LateAcceptanceHillClimbing::<MaxCutFlipNeighbor>::new(
            StopCondition::iterations(1_000),
            1,
        );
        lahc.run(&mut state).unwrap();

        assert!(state.best_solution.objective >= 0.0);
    }
}
