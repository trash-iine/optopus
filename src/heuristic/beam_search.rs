use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::{MoveToNeighbor, ProblemTrait, Rankable, SearchState};
use crate::trait_defs::rank_cmp;

/// Beam search heuristic.
///
/// Maintains a beam of `beam_width` candidate solutions in parallel.
/// At each step, the neighborhood of every candidate is expanded and the top
/// `beam_width` solutions (by [`Rankable`] order) are kept as the next beam.
/// The best solution across all beam members is tracked in `SearchState::best_solution`.
///
/// # References
///
/// - Ow, P. S. and Morton, T. E. "Filtered Beam Search in Scheduling." *International Journal
///   of Production Research*, 26(1), 35-62, 1988.
///   [DOI](https://doi.org/10.1080/00207548808947840)
///
/// # Example
///
/// ```
/// use optopus::heuristic::{BeamSearch, StopCondition, Heuristic};
/// use optopus::search_state::SearchState;
/// use optopus::problem::{MaxCut, MaxCutFlipNeighbor};
///
/// let mc = MaxCut::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]);
///
/// let mut state = SearchState::new(&mc);
/// let mut bs = BeamSearch::<MaxCut, MaxCutFlipNeighbor>::new(
///     StopCondition::iterations(1000),
///     5,
/// );
/// bs.run(&mut state).unwrap();
/// ```
pub struct BeamSearch<P: ProblemTrait, N> {
    pub stop_condition: StopCondition,
    pub beam_width: usize,
    beam: Vec<P::Solution>,
    _phantom: std::marker::PhantomData<N>,
}

impl<P: ProblemTrait, N> BeamSearch<P, N> {
    /// Create a new [`BeamSearch`] with the given stopping condition and beam width.
    /// `beam_width` must be greater than 0.
    pub fn new(stop_condition: StopCondition, beam_width: usize) -> Self {
        if beam_width == 0 {
            panic!("beam_width must be greater than 0");
        }
        Self {
            stop_condition,
            beam_width,
            beam: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<P, N> Heuristic<P> for BeamSearch<P, N>
where
    P: ProblemTrait,
    N: MoveToNeighbor<P> + Rankable,
{
    /// Clear the beam to reset the heuristic state.
    fn clear(&mut self) {
        self.beam.clear();
    }

    fn is_done<'a>(&self, state: &SearchState<'a, P>) -> bool {
        self.stop_condition.is_done(state)
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        // Initialize the beam from solution
        if self.beam.is_empty() {
            self.beam.push(state.solution.clone());
        }

        // Expand the neighborhood of every beam candidate
        let mut candidates: Vec<_> = Vec::new();
        for beam_sol in self.beam.iter() {
            for neighbor in N::iter(state.instance, beam_sol) {
                let mut candidate = beam_sol.clone();
                neighbor.apply_to_solution(state.instance, &mut candidate)?;
                candidates.push(candidate);
            }
        }

        // If no neighbors exist, just advance the iteration counter
        if candidates.is_empty() {
            state.progress_iteration();
            return Ok(());
        }

        // Update the state with the best candidate among all neighbors
        state.solution = candidates
            .iter()
            .max_by(|a, b| rank_cmp(*a, *b))
            .expect("candidates must not be empty")
            .clone();
        state.update_best();
        state.progress_iteration();

        // Keep the top beam_width candidates for the next iteration.
        // select_nth_unstable_by is O(n) expected vs O(n log n) for a full sort;
        // ordering within the surviving beam members does not matter.
        if candidates.len() > self.beam_width {
            // Reversed comparator: better solutions sort first.
            candidates.select_nth_unstable_by(self.beam_width - 1, |a, b| rank_cmp(b, a));
            candidates.truncate(self.beam_width);
        }
        self.beam = candidates;

        Ok(())
    }
}
