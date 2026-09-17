use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::SearchState;
use crate::trait_defs::{MoveToNeighbor, ProblemTrait, Rankable, rank_cmp};

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
    _neighbor: std::marker::PhantomData<N>,
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
            _neighbor: std::marker::PhantomData,
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

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    /// An empty neighborhood only advances the iteration counter (the beam is
    /// kept); the stop condition eventually terminates the run.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::max_cut::MaxCut;
    use crate::problem::{MaxCutFlipNeighbor, MaxCutSolution};

    /// Five vertices, so one flip neighborhood is five candidates and a beam
    /// of two has to discard three of them.
    fn path5() -> MaxCut {
        MaxCut::from_edges([
            (0, 1, 1.0),
            (1, 2, 2.0),
            (2, 3, 1.0),
            (3, 4, 3.0),
            (0, 4, 1.0),
        ])
    }

    fn beam_search(width: usize, iterations: u64) -> BeamSearch<MaxCut, MaxCutFlipNeighbor> {
        BeamSearch::new(StopCondition::iterations(iterations), width)
    }

    /// The beam is trimmed by a reversed comparator, so the surviving members
    /// are the best candidates and not the worst. Sorting the other way keeps
    /// the beam the right size and the run still terminates, which is all a
    /// smoke test would see.
    #[test]
    fn the_beam_keeps_its_best_candidates_and_nothing_more() {
        let mc = path5();
        let start = MaxCutSolution::new_from_assignment(&mc, vec![false; 5]);
        let mut state = SearchState::with_solution(&mc, start.clone());

        let mut bs = beam_search(2, 1);
        bs.run_once(&mut state).unwrap();

        assert_eq!(bs.beam.len(), 2, "the beam is trimmed to its width");

        // Every candidate of that first expansion, ranked.
        let mut all: Vec<MaxCutSolution> = MaxCutFlipNeighbor::iter(&mc, &start)
            .map(|m| {
                let mut s = start.clone();
                m.apply_to_solution(&mc, &mut s).unwrap();
                s
            })
            .collect();
        all.sort_by(|a, b| rank_cmp(b, a));

        let kept: Vec<f32> = bs.beam.iter().map(|s| s.objective).collect();
        let best_two: Vec<f32> = all[..2].iter().map(|s| s.objective).collect();
        assert_eq!(kept, best_two, "the beam kept the wrong candidates");

        assert_eq!(state.solution.objective, all[0].objective);
        assert_eq!(state.best_solution.objective, all[0].objective);
    }

    /// A beam wider than the neighborhood must not trim, and the run has to
    /// end on the budget rather than on the beam running out.
    #[test]
    fn a_beam_wider_than_the_neighborhood_keeps_everything() {
        let mc = path5();
        let mut state = SearchState::new_with_seed(&mc, 3);

        let mut bs = beam_search(100, 4);
        bs.run(&mut state).unwrap();

        assert_eq!(state.iteration, 4);
        assert!(!bs.beam.is_empty());
        assert!(bs.beam.len() <= 100);
        assert_eq!(
            state.best_solution.objective,
            mc.calculate_cut_size(&state.best_solution.x)
        );
    }

    /// With no edges there are no vertices and so no moves. The run has to
    /// charge the iteration anyway or it never meets its budget.
    #[test]
    fn an_empty_neighborhood_still_charges_its_iteration() {
        let mc = MaxCut::new(crate::common::Graph::new());
        let mut state = SearchState::new_with_seed(&mc, 1);

        let mut bs = beam_search(3, 5);
        bs.run(&mut state).unwrap();

        assert_eq!(state.iteration, 5);
    }

    #[test]
    #[should_panic(expected = "beam_width must be greater than 0")]
    fn a_zero_width_beam_panics_at_construction() {
        let _ = beam_search(0, 1);
    }
}
