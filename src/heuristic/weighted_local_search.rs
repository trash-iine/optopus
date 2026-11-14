use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::SearchState;
use crate::trait_defs::{MoveToNeighbor, WeightedConstraints};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// A local search that weights the constraints it keeps violating.
///
/// Every constraint of the problem carries an integer weight that starts at
/// one, and the search descends on the weighted cost, the objective plus the
/// weights of the violated constraints. Each step it applies the move that
/// lowers that cost the most among the moves the tabu memory allows, drawing
/// uniformly among equals.
///
/// When no allowed move lowers it, the search stands at a weighted local
/// optimum and changes the weights instead. With probability `1 - smoothing`
/// every violated constraint gains one, up to the larger of `1000` and the
/// number of constraints. Otherwise every satisfied constraint above one
/// loses one. It then leaves by the best of the problem's
/// [`escape_moves`](WeightedConstraints::escape_moves), the moves that satisfy
/// one violated constraint drawn uniformly or, once nothing is violated, the
/// moves that work toward a better objective. An escape the tabu memory forbids is
/// taken only when it forbids them all.
///
/// A move that was applied may not be undone for a number of steps drawn from
/// `tabu_tenure`.
///
/// The best solution is kept by the problem's own order, which the weights do
/// not enter. The run ends on its [`StopCondition`], or at a weighted local
/// optimum with no escape, a solution that violates nothing and that no move
/// takes toward a better objective.
///
/// The weights are the search's own and start again at one on every
/// [`run`](Heuristic::run).
///
/// # Example
///
/// ```
/// use optopus::prelude::*;
///
/// let vc = VertexCover::new(Graph::from_edges([
///     (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0),
/// ]));
/// let mut state = SearchState::new_with_seed(&vc, 1);
/// let mut search = WeightedLocalSearch::<VertexCover>::new(StopCondition::iterations(200));
/// search.run(&mut state)?;
/// assert_eq!(state.best_solution.uncovered_edges, 0);
/// assert_eq!(state.best_solution.cover_size, 2);
/// # Ok::<(), optopus::error::OptError>(())
/// ```
///
/// # References
///
/// - Thornton, J., Pham, D. N., Bain, S., and Ferreira Jr., V. "Additive versus
///   Multiplicative Clause Weighting for SAT." Proc. AAAI-04, 191-196, 2004.
/// - Lin, P., Cai, S., Zou, M., and Lin, J. "New Characterizations and Efficient
///   Local Search for General Integer Linear Programming." 2023.
///   [arXiv:2305.00188](https://arxiv.org/abs/2305.00188)
pub struct WeightedLocalSearch<P: WeightedConstraints> {
    stop_condition: StopCondition,
    tabu_tenure: (u64, u64),
    smoothing: f64,
    weights: Option<P::Weights>,
    /// The stream ties in the scan are broken from, forked off the state's on
    /// the first step of a run so the scan can read the state while drawing.
    ties: Option<SmallRng>,
    escapes: Vec<P::Move>,
    /// Set at a weighted local optimum that has no escape.
    no_escape: bool,
}

impl<P: WeightedConstraints> WeightedLocalSearch<P> {
    /// The tenure range a search starts with.
    pub const TABU_TENURE: (u64, u64) = (3, 12);
    /// The smoothing probability a search starts with.
    pub const SMOOTHING: f64 = 0.0003;
    /// The least value the cap on a weight takes.
    pub const MIN_WEIGHT_CAP: u32 = 1000;

    /// A search with [`TABU_TENURE`](Self::TABU_TENURE) and
    /// [`SMOOTHING`](Self::SMOOTHING).
    pub fn new(stop_condition: StopCondition) -> Self {
        Self {
            stop_condition,
            tabu_tenure: Self::TABU_TENURE,
            smoothing: Self::SMOOTHING,
            weights: None,
            ties: None,
            escapes: Vec::new(),
            no_escape: false,
        }
    }

    /// Sets the range `(min, max)` a move's tenure is drawn from.
    ///
    /// # Panics
    ///
    /// Panics if `tabu_tenure.0 > tabu_tenure.1` (an empty range).
    pub fn with_tabu_tenure(mut self, tabu_tenure: (u64, u64)) -> Self {
        crate::building_blocks::search::assert_valid_tenure(tabu_tenure);
        self.tabu_tenure = tabu_tenure;
        self
    }

    /// Sets the probability that a weighted local optimum lowers the weights
    /// of the satisfied constraints rather than raising those of the violated.
    ///
    /// # Panics
    ///
    /// Panics if `smoothing` is not within `[0.0, 1.0]`.
    pub fn with_smoothing(mut self, smoothing: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&smoothing),
            "smoothing must be within [0.0, 1.0], got {smoothing}"
        );
        self.smoothing = smoothing;
        self
    }

    /// The weights of the run in progress, or of the last one.
    pub fn weights(&self) -> Option<&P::Weights> {
        self.weights.as_ref()
    }
}

/// Applies `mv` through the problem, so the weights are repriced with the
/// solution, and counts it as [`SearchState::apply`] would.
fn apply<P: WeightedConstraints>(
    state: &mut SearchState<'_, P>,
    weights: &mut P::Weights,
    mv: &P::Move,
) -> Result<(), OptError> {
    state.record_tabu(mv);
    state.iteration = mv.apply_to_iteration(state.iteration);
    state
        .instance
        .apply_weighted(&mut state.solution, weights, mv)?;
    state.n_accepted += 1;
    state.update_best();
    Ok(())
}

impl<P: WeightedConstraints> Heuristic<P> for WeightedLocalSearch<P> {
    fn clear(&mut self) {
        self.weights = None;
        self.ties = None;
        self.no_escape = false;
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn is_done<'a>(&self, state: &SearchState<'a, P>) -> bool {
        self.no_escape || self.stop_condition.is_done(state)
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        let instance = state.instance;
        let weights = self
            .weights
            .get_or_insert_with(|| instance.unit_weights(&state.solution));
        // The tenure is set without the mode, since the moves are applied
        // through the problem and recorded by hand.
        state.start_record_tabu(self.tabu_tenure);
        state.stop_record_tabu();

        let tie_rng = self
            .ties
            .get_or_insert_with(|| SmallRng::from_rng(&mut state.rng));

        // Driven by `for_each`, so a move type that scans faster through
        // `fold` is scanned that way.
        let mut best: Option<P::Move> = None;
        let mut best_gain = 0.0;
        let mut ties = 0u32;
        P::Move::iter(instance, &state.solution).for_each(|mv| {
            let gain = instance.weighted_gain(&state.solution, weights, &mv);
            if gain <= 0.0 || gain < best_gain || !state.tabu_allows(&mv) {
                return;
            }
            if gain > best_gain {
                best_gain = gain;
                ties = 0;
            }
            ties += 1;
            if ties == 1 || tie_rng.random_range(0..ties) == 0 {
                best = Some(mv);
            }
        });
        if let Some(mv) = best {
            return apply(state, weights, &mv);
        }

        if state.rng.random::<f64>() < self.smoothing {
            instance.lower_satisfied(&state.solution, weights);
        } else {
            let cap = Self::MIN_WEIGHT_CAP
                .max(u32::try_from(instance.constraint_count()).unwrap_or(u32::MAX));
            instance.raise_violated(&state.solution, weights, cap);
        }

        self.escapes.clear();
        instance.escape_moves(&state.solution, weights, &mut state.rng, &mut self.escapes);
        // (allowed, gain) orders an allowed escape above any forbidden one.
        let mut chosen: Option<(usize, (bool, f64))> = None;
        let mut ties = 0u32;
        for (at, mv) in self.escapes.iter().enumerate() {
            let key = (
                state.tabu_allows(mv),
                instance.weighted_gain(&state.solution, weights, mv),
            );
            match chosen {
                Some((_, held)) if key < held => continue,
                Some((_, held)) if key == held => {}
                _ => ties = 0,
            }
            ties += 1;
            if ties == 1 || state.rng.random_range(0..ties) == 0 {
                chosen = Some((at, key));
            }
        }
        match chosen {
            Some((at, _)) => apply(state, weights, &self.escapes[at]),
            None => {
                self.no_escape = true;
                state.progress_iteration();
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::building_blocks::instance::Graph;
    use crate::problem::{Sat, VertexCover};
    use crate::trait_defs::Rankable;

    /// A ring of `n` vertices with a chord from each vertex to the one two
    /// steps on. Its smallest cover leaves out every third vertex.
    fn chorded_ring(n: usize) -> VertexCover {
        let mut edges = Vec::new();
        for v in 0..n {
            edges.push((v, (v + 1) % n, 1.0));
            edges.push((v, (v + 2) % n, 1.0));
        }
        VertexCover::new(Graph::from_edges(edges))
    }

    /// A satisfiable 3-SAT formula, every clause agreeing with a hidden
    /// assignment in at least one literal.
    fn planted_3sat(n: usize, m: usize, seed: u64) -> Sat {
        use rand::SeedableRng;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let hidden: Vec<bool> = (0..n).map(|_| rng.random_bool(0.5)).collect();
        let mut sat = Sat::new(n);
        while sat.n_clauses() < m {
            let clause: Vec<i64> = (0..3)
                .map(|_| {
                    let v = rng.random_range(0..n);
                    if rng.random_bool(0.5) {
                        v as i64 + 1
                    } else {
                        -(v as i64 + 1)
                    }
                })
                .collect();
            if Sat::is_clause_satisfied(&clause, &hidden) {
                sat.add_clause(clause);
            }
        }
        sat
    }

    #[test]
    fn finds_the_smallest_cover_of_a_chorded_ring() {
        let vc = chorded_ring(30);
        let mut state = SearchState::new_with_seed(&vc, 4);
        let mut search = WeightedLocalSearch::<VertexCover>::new(StopCondition::iterations(20_000));
        search.run(&mut state).unwrap();
        assert_eq!(state.best_solution.uncovered_edges, 0);
        assert_eq!(state.best_solution.cover_size, 20);
        assert_eq!(state.iteration, state.n_accepted + state.n_rejected);
    }

    /// The chorded ring on a multiple of three vertices is made of triangles
    /// and takes three colors, each vertex colored by its index modulo three.
    #[test]
    fn colors_a_chorded_ring_with_three_colors() {
        use crate::problem::GraphColoring;
        let mut edges = Vec::new();
        for v in 0..30 {
            edges.push((v, (v + 1) % 30, 1.0));
            edges.push((v, (v + 2) % 30, 1.0));
        }
        let gc = GraphColoring::new(Graph::from_edges(edges));
        let mut state = SearchState::new_with_seed(&gc, 4);
        let mut search =
            WeightedLocalSearch::<GraphColoring>::new(StopCondition::iterations(20_000));
        search.run(&mut state).unwrap();
        assert_eq!(state.best_solution.conflicts, 0);
        assert_eq!(state.best_solution.colors_used, 3);
        assert_eq!(state.iteration, state.n_accepted + state.n_rejected);
    }

    #[test]
    fn satisfies_a_planted_formula_and_stops_there() {
        let sat = planted_3sat(60, 240, 8);
        let mut state = SearchState::new_with_seed(&sat, 3);
        let mut search = WeightedLocalSearch::<Sat>::new(StopCondition::iterations(200_000));
        search.run(&mut state).unwrap();
        assert_eq!(state.best_solution.n_satisfied, sat.n_clauses());
        assert!(
            state.iteration < 200_000,
            "a satisfying assignment has no escape, which ends the run"
        );
    }

    /// The weights leave the solution's own caches valid, so what the run
    /// hands on is what a fresh build of the same assignment gives.
    #[test]
    fn leaves_the_solution_consistent() {
        let sat = planted_3sat(40, 170, 1);
        let mut state = SearchState::new_with_seed(&sat, 6);
        let mut search = WeightedLocalSearch::<Sat>::new(StopCondition::iterations(300));
        search.run(&mut state).unwrap();
        let fresh =
            crate::problem::SatSolution::new_from_assignment(&sat, state.solution.x.clone());
        assert_eq!(state.solution.gain, fresh.gain);
        assert_eq!(state.solution.n_satisfied, fresh.n_satisfied);
        assert!(!state.solution.is_better_than(&state.best_solution));
    }

    #[test]
    fn the_same_seed_gives_the_same_run() {
        let vc = chorded_ring(40);
        let run = |seed| {
            let mut state = SearchState::new_with_seed(&vc, seed);
            let mut search =
                WeightedLocalSearch::<VertexCover>::new(StopCondition::iterations(2_000));
            search.run(&mut state).unwrap();
            (state.solution.x.clone(), state.best_iteration)
        };
        assert_eq!(run(12), run(12));
    }

    /// A move just made is not undone while its tenure lasts, unless every
    /// escape is forbidden.
    #[test]
    fn weights_rise_on_what_stays_violated() {
        // Two clauses that cannot both hold, so one is always violated.
        let mut sat = Sat::new(1);
        sat.add_clause([1]);
        sat.add_clause([-1]);
        let mut state = SearchState::new_with_seed(&sat, 0);
        let mut search =
            WeightedLocalSearch::<Sat>::new(StopCondition::iterations(50)).with_smoothing(0.0);
        search.run(&mut state).unwrap();
        let weights = search.weights().expect("the run built its weights");
        assert!(weights.weight(0) > 1 && weights.weight(1) > 1);
        assert_eq!(state.best_solution.n_satisfied, 1);
    }

    #[test]
    #[should_panic(expected = "smoothing must be within")]
    fn rejects_a_smoothing_outside_the_unit_interval() {
        let _ = WeightedLocalSearch::<Sat>::new(StopCondition::iterations(1)).with_smoothing(1.5);
    }
}
