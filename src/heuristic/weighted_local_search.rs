use super::{Heuristic, StopCondition};
use crate::building_blocks::search::{ConstraintWeights, assert_valid_tenure};
use crate::error::OptError;
use crate::search_state::SearchState;
use crate::trait_defs::{MoveToNeighbor, WeightedConstraints};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// The least value the cap on a constraint's weight takes.
const MIN_WEIGHT_CAP: u32 = 1000;
/// How many steps the best solution may go without improving before the
/// search restarts.
const RESTART_STEPS: u64 = 1_500_000;

/// A local search that weights the constraints it keeps violating.
///
/// Every constraint of the problem carries an integer weight that starts at
/// one, and so does the objective. Each step works in one of three modes, by
/// the solution in hand.
///
/// - Until a solution that violates nothing has been found, the search looks
///   for one. It draws a few violated constraints, lists their
///   [`tight_moves`](WeightedConstraints::tight_moves), the moves that satisfy
///   them, and applies the one that gains the most weight, if it gains any.
/// - While the solution violates nothing, the search betters the objective
///   without violating anything. It applies the move that keeps every
///   constraint satisfied and lowers the objective the most. When there is
///   none it takes the best priced step toward a better objective from
///   [`objective_escapes`](WeightedConstraints::objective_escapes), which
///   violates something.
/// - Once a solution that violates nothing has been found, a solution that
///   violates something is repaired toward one better than the best. A move
///   is then priced by the weight it gains, plus the objective's weight if it
///   leads below the best objective found and less it otherwise. The search
///   draws a few violated constraints as above, and if none of their tight
///   moves has a positive price, a few satisfied ones, whose tight moves keep
///   them satisfied and take up the slack they have.
///
/// When no drawn move has a positive price, the search changes the weights.
/// With probability `1 - smoothing` every violated constraint gains one, up to
/// the larger of `1000` and the number of constraints, and the objective gains
/// one, up to a tenth of that, if the solution is no better than the best.
/// Otherwise every satisfied constraint above one loses one, and the objective
/// loses one if the solution is better than the best. It then applies the best
/// priced tight move of one violated constraint drawn uniformly.
///
/// The moves of a few constraints are all priced when there are few of them,
/// and a sample of them is when there are more, so a step costs about the
/// same whatever the size of the problem. A tight move that was applied may
/// not be undone for a number of steps drawn from `tabu_tenure`, and a
/// forbidden move is taken only when the one constraint drawn at a weight
/// change, or the escapes toward a better objective, offer no other.
///
/// After 1,500,000 steps without a better best solution, the search restarts
/// from [`restart_from`](WeightedConstraints::restart_from) the best, with
/// every weight back at one.
///
/// The best solution is kept by the problem's own order, which the weights do
/// not enter. The run ends on its [`StopCondition`], or on a solution that
/// violates nothing and that no move takes toward a better objective.
///
/// The weights and the tabu memory start again on every
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
/// let mut search =
///     WeightedLocalSearch::<VertexCover>::new(StopCondition::iterations(200), (3, 12), 0.0003);
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
    violated_samples: (usize, usize),
    satisfied_samples: (usize, usize),
    random_constraint_samples: usize,
    weights: Option<ConstraintWeights<P>>,
    objective_weight: u32,
    /// The objective of the best solution that violates nothing, once there
    /// is one.
    best_objective: Option<f64>,
    /// The iteration the run started or last restarted on.
    restarted_at: u64,
    /// The stream the search draws from, forked off the state's on the first
    /// step of a run so it can draw while reading the state.
    rng: SmallRng,
    drawn: Vec<usize>,
    moves: Vec<P::Move>,
    /// Set on a solution that violates nothing and has no escape.
    no_escape: bool,
}

impl<P: WeightedConstraints> WeightedLocalSearch<P> {
    /// How many violated constraints a step draws, and how many of their
    /// moves it prices at most.
    pub const DEFAULT_VIOLATED_SAMPLES: (usize, usize) = (3, 2000);
    /// How many satisfied constraints a step draws, and how many of their
    /// moves it prices at most.
    pub const DEFAULT_SATISFIED_SAMPLES: (usize, usize) = (30, 350);
    /// How many moves of the one constraint drawn at a weight change a step
    /// prices at most.
    pub const DEFAULT_RANDOM_CONSTRAINT_SAMPLES: usize = 150;

    /// A search whose tight moves stay forbidden for a tenure drawn from
    /// `tabu_tenure`, and that lowers the weights at a weight change with
    /// probability `smoothing`. The tuned values are `(3, 12)` and `0.0003`.
    ///
    /// # Panics
    ///
    /// Panics if `tabu_tenure.0 > tabu_tenure.1` (an empty range), or if
    /// `smoothing` is not within `[0.0, 1.0]`.
    pub fn new(stop_condition: StopCondition, tabu_tenure: (u64, u64), smoothing: f64) -> Self {
        assert_valid_tenure(tabu_tenure);
        assert!(
            (0.0..=1.0).contains(&smoothing),
            "smoothing must be within [0.0, 1.0], got {smoothing}"
        );
        Self {
            stop_condition,
            tabu_tenure,
            smoothing,
            violated_samples: Self::DEFAULT_VIOLATED_SAMPLES,
            satisfied_samples: Self::DEFAULT_SATISFIED_SAMPLES,
            random_constraint_samples: Self::DEFAULT_RANDOM_CONSTRAINT_SAMPLES,
            weights: None,
            objective_weight: 1,
            best_objective: None,
            restarted_at: 0,
            rng: SmallRng::seed_from_u64(0),
            drawn: Vec::new(),
            moves: Vec::new(),
            no_escape: false,
        }
    }

    /// Sets how many violated constraints a step draws and how many of their
    /// moves it prices at most.
    ///
    /// # Panics
    ///
    /// Panics if either is zero.
    pub fn with_violated_samples(mut self, constraints: usize, moves: usize) -> Self {
        assert!(constraints > 0 && moves > 0, "samples must be positive");
        self.violated_samples = (constraints, moves);
        self
    }

    /// Sets how many satisfied constraints a step draws and how many of their
    /// moves it prices at most.
    ///
    /// # Panics
    ///
    /// Panics if either is zero.
    pub fn with_satisfied_samples(mut self, constraints: usize, moves: usize) -> Self {
        assert!(constraints > 0 && moves > 0, "samples must be positive");
        self.satisfied_samples = (constraints, moves);
        self
    }

    /// Sets how many moves of the one constraint drawn at a weight change a
    /// step prices at most.
    ///
    /// # Panics
    ///
    /// Panics if `moves` is zero.
    pub fn with_random_constraint_samples(mut self, moves: usize) -> Self {
        assert!(moves > 0, "samples must be positive");
        self.random_constraint_samples = moves;
        self
    }

    /// The weights of the run in progress, or of the last one.
    pub fn weights(&self) -> Option<&ConstraintWeights<P>> {
        self.weights.as_ref()
    }

    /// The weight of the objective in the run in progress, or in the last one.
    pub fn objective_weight(&self) -> u32 {
        self.objective_weight
    }

    /// Starts the weights again at one for the solution in hand.
    fn reset_weights(&mut self, state: &SearchState<'_, P>) {
        self.weights = Some(ConstraintWeights::new(
            state.instance,
            &state.solution,
            MIN_WEIGHT_CAP,
        ));
        self.objective_weight = 1;
        self.note_best_objective(state);
    }

    /// Keeps the objective of the solution in hand if it violates nothing and
    /// betters the best.
    fn note_best_objective(&mut self, state: &SearchState<'_, P>) {
        let weights = self.weights.as_ref().expect("the run built its weights");
        if weights.violated().is_empty() {
            let objective = state.instance.objective(&state.solution);
            if self.best_objective.is_none_or(|best| objective < best) {
                self.best_objective = Some(objective);
            }
        }
    }

    /// Draws into `self.drawn` all the violated constraints if there are at
    /// most `count`, and `count` of them uniformly otherwise.
    fn draw_violated(&mut self, count: usize) {
        let violated = self
            .weights
            .as_ref()
            .expect("the run built its weights")
            .violated()
            .as_slice();
        self.drawn.clear();
        if violated.len() <= count {
            self.drawn.extend_from_slice(violated);
        } else {
            self.drawn
                .extend((0..count).map(|_| violated[self.rng.random_range(0..violated.len())]));
        }
    }

    /// Draws into `self.drawn` `count` satisfied constraints uniformly, none
    /// if every constraint is violated.
    fn draw_satisfied(&mut self, count: usize) {
        let weights = self.weights.as_ref().expect("the run built its weights");
        self.drawn.clear();
        if weights.violated().len() == weights.len() {
            return;
        }
        for _ in 0..count {
            let c = loop {
                let c = self.rng.random_range(0..weights.len());
                if !weights.violated().contains(c) {
                    break c;
                }
            };
            self.drawn.push(c);
        }
    }

    /// Lists into `self.moves` the tight moves of the constraints in
    /// `self.drawn`.
    fn list_tight_moves(&mut self, state: &SearchState<'_, P>) {
        let prices = self
            .weights
            .as_ref()
            .expect("the run built its weights")
            .prices();
        self.moves.clear();
        for &c in &self.drawn {
            state
                .instance
                .tight_moves(&state.solution, prices, c, &mut self.moves);
        }
    }

    /// The tight move among `self.moves` with the highest price, of at most
    /// `limit` of them drawn uniformly when there are more. Ties are drawn
    /// uniformly. Only an allowed move with a positive price is taken when
    /// `positive` is set, and otherwise an allowed move is taken over any
    /// forbidden one.
    fn best_tight_move(
        &mut self,
        state: &SearchState<'_, P>,
        limit: usize,
        positive: bool,
    ) -> Option<usize> {
        let instance = state.instance;
        let weights = self.weights.as_ref().expect("the run built its weights");
        let objective = self
            .best_objective
            .map(|best| (best, instance.objective(&state.solution)));
        let objective_weight = f64::from(self.objective_weight);
        let n = self.moves.len();
        let sampled = n > limit;
        let mut chosen: Option<(usize, (bool, f64))> = None;
        let mut ties = 0u32;
        for k in 0..n.min(limit) {
            let at = if sampled {
                self.rng.random_range(0..n)
            } else {
                k
            };
            let mv = &self.moves[at];
            let mut price = instance.weight_gain(&state.solution, weights.prices(), mv) as f64;
            if let Some((best, now)) = objective {
                price += if now + instance.objective_change(&state.solution, mv) < best {
                    objective_weight
                } else {
                    -objective_weight
                };
            }
            let allowed = state.tabu_allows(mv);
            if positive && (!allowed || price <= 0.0) {
                continue;
            }
            let key = (allowed, price);
            match chosen {
                Some((_, held)) if key < held => continue,
                Some((_, held)) if key == held => {}
                _ => ties = 0,
            }
            ties += 1;
            if ties == 1 || self.rng.random_range(0..ties) == 0 {
                chosen = Some((at, key));
            }
        }
        chosen.map(|(at, _)| at)
    }

    /// The move that keeps every constraint satisfied and lowers the
    /// objective the most, ties drawn uniformly.
    fn best_lift(&mut self, state: &SearchState<'_, P>) -> Option<P::Move> {
        let instance = state.instance;
        let prices = self
            .weights
            .as_ref()
            .expect("the run built its weights")
            .prices();
        let rng = &mut self.rng;
        let mut best: Option<P::Move> = None;
        let mut best_drop = 0.0;
        let mut ties = 0u32;
        // Driven by `for_each`, so a move type that scans faster through
        // `fold` is scanned that way.
        P::Move::iter(instance, &state.solution).for_each(|mv| {
            let drop = -instance.objective_change(&state.solution, &mv);
            if drop <= 0.0
                || drop < best_drop
                || instance.weight_gain(&state.solution, prices, &mv) < 0
            {
                return;
            }
            if drop > best_drop {
                best_drop = drop;
                ties = 0;
            }
            ties += 1;
            if ties == 1 || rng.random_range(0..ties) == 0 {
                best = Some(mv);
            }
        });
        best
    }

    /// Raises the weights of what is violated, or with probability
    /// `smoothing` lowers those of what is satisfied, and the objective's
    /// with them once a solution that violates nothing has been found.
    fn change_weights(&mut self, state: &SearchState<'_, P>) {
        let instance = state.instance;
        let weights = self.weights.as_mut().expect("the run built its weights");
        let better = self
            .best_objective
            .map(|best| instance.objective(&state.solution) < best);
        if self.rng.random::<f64>() < self.smoothing {
            weights.lower_satisfied(instance, &state.solution);
            if better == Some(true) && self.objective_weight > 1 {
                self.objective_weight -= 1;
            }
        } else {
            weights.raise_violated(instance, &state.solution);
            if better == Some(false) && self.objective_weight < weights.cap() / 10 {
                self.objective_weight += 1;
            }
        }
    }

    /// Applies `mv` through the problem, so the weights are repriced with the
    /// solution, and counts it as [`SearchState::apply`] would.
    fn apply(&mut self, state: &mut SearchState<'_, P>, mv: &P::Move) -> Result<(), OptError> {
        let weights = self.weights.as_mut().expect("the run built its weights");
        state.record_tabu(mv);
        state.iteration = mv.apply_to_iteration(state.iteration);
        weights.apply(state.instance, &mut state.solution, mv)?;
        state.n_accepted += 1;
        state.update_best();
        self.note_best_objective(state);
        Ok(())
    }

    /// Applies `self.moves[at]`.
    fn apply_listed(&mut self, state: &mut SearchState<'_, P>, at: usize) -> Result<(), OptError> {
        let mv = self.moves.swap_remove(at);
        self.apply(state, &mv)
    }
}

impl<P: WeightedConstraints> Heuristic<P> for WeightedLocalSearch<P> {
    fn clear(&mut self) {
        self.weights = None;
        self.best_objective = None;
        self.no_escape = false;
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn is_done<'a>(&self, state: &SearchState<'a, P>) -> bool {
        self.no_escape || self.stop_condition.is_done(state)
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        if self.weights.is_none() {
            self.rng = SmallRng::from_rng(&mut state.rng);
            state.reset_tabu();
            self.restarted_at = state.iteration;
            self.reset_weights(state);
        }
        // The tenure is set without the mode, since the moves are applied
        // through the problem and recorded by hand.
        state.start_record_tabu(self.tabu_tenure);
        state.stop_record_tabu();

        let unimproved_since = state.best_iteration.max(self.restarted_at);
        if state.iteration - unimproved_since >= RESTART_STEPS {
            state.solution = state
                .instance
                .restart_from(&state.best_solution, &mut self.rng);
            state.progress_iteration();
            state.update_best();
            self.restarted_at = state.iteration;
            self.reset_weights(state);
            return Ok(());
        }

        let weights = self.weights.as_ref().expect("the run built its weights");
        if weights.violated().is_empty() {
            if let Some(mv) = self.best_lift(state) {
                return self.apply(state, &mv);
            }
            self.moves.clear();
            state
                .instance
                .objective_escapes(&state.solution, &mut self.moves);
            if self.moves.is_empty() {
                self.no_escape = true;
                state.progress_iteration();
                return Ok(());
            }
            let at = self
                .best_tight_move(state, usize::MAX, false)
                .expect("an escape was listed");
            return self.apply_listed(state, at);
        }

        let (constraints, limit) = self.violated_samples;
        self.draw_violated(constraints);
        self.list_tight_moves(state);
        if let Some(at) = self.best_tight_move(state, limit, true) {
            return self.apply_listed(state, at);
        }

        if self.best_objective.is_some() {
            let (constraints, limit) = self.satisfied_samples;
            self.draw_satisfied(constraints);
            self.list_tight_moves(state);
            if let Some(at) = self.best_tight_move(state, limit, true) {
                return self.apply_listed(state, at);
            }
        }

        self.change_weights(state);
        self.draw_violated(1);
        self.list_tight_moves(state);
        match self.best_tight_move(state, self.random_constraint_samples, false) {
            Some(at) => self.apply_listed(state, at),
            None => {
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

    fn search<P: WeightedConstraints>(iterations: u64) -> WeightedLocalSearch<P> {
        WeightedLocalSearch::new(StopCondition::iterations(iterations), (3, 12), 0.0003)
    }

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
        let mut rng = SmallRng::seed_from_u64(seed);
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
        search::<VertexCover>(20_000).run(&mut state).unwrap();
        assert_eq!(state.best_solution.uncovered_edges, 0);
        assert_eq!(state.best_solution.cover_size, 20);
        assert_eq!(state.iteration, state.n_accepted + state.n_rejected);
    }

    /// The chorded ring on a multiple of three vertices is made of triangles
    /// and takes three colors, each vertex colored by its index modulo three.
    #[test]
    fn colors_a_chorded_ring_with_three_colors() {
        use crate::problem::GraphColoring;
        let gc = GraphColoring::new(chorded_ring(30).graph);
        let mut state = SearchState::new_with_seed(&gc, 4);
        search::<GraphColoring>(20_000).run(&mut state).unwrap();
        assert_eq!(state.best_solution.conflicts, 0);
        assert_eq!(state.best_solution.colors_used, 3);
        assert_eq!(state.iteration, state.n_accepted + state.n_rejected);
    }

    #[test]
    fn satisfies_a_planted_formula_and_stops_there() {
        let sat = planted_3sat(60, 240, 8);
        let mut state = SearchState::new_with_seed(&sat, 3);
        search::<Sat>(200_000).run(&mut state).unwrap();
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
        search::<Sat>(300).run(&mut state).unwrap();
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
            search::<VertexCover>(2_000).run(&mut state).unwrap();
            (state.solution.x.clone(), state.best_iteration)
        };
        assert_eq!(run(12), run(12));
    }

    /// Of two clauses that cannot both hold, the one left violated gains
    /// weight at every weight change.
    #[test]
    fn weights_rise_on_what_stays_violated() {
        let mut sat = Sat::new(1);
        sat.add_clause([1]);
        sat.add_clause([-1]);
        let mut state = SearchState::new_with_seed(&sat, 0);
        let mut search =
            WeightedLocalSearch::<Sat>::new(StopCondition::iterations(50), (3, 12), 0.0);
        search.run(&mut state).unwrap();
        let weights = search.weights().expect("the run built its weights");
        assert!(weights.weight(0) > 1 && weights.weight(1) > 1);
        assert_eq!(state.best_solution.n_satisfied, 1);
    }

    /// What one run forbade does not carry into the next.
    #[test]
    fn a_second_run_starts_without_prohibitions() {
        use crate::problem::VertexCoverFlipNeighbor;
        let vc = chorded_ring(30);
        let forbidden = |state: &SearchState<'_, VertexCover>| {
            (0..30)
                .map(|v| VertexCoverFlipNeighbor::new(&vc, &state.solution, v))
                .filter(|mv| !state.tabu_allows(mv))
                .count()
        };
        let mut state = SearchState::new_with_seed(&vc, 2);
        let mut search = search::<VertexCover>(500);
        search.run(&mut state).unwrap();
        assert!(forbidden(&state) > 1, "the first run leaves prohibitions");
        search.stop_condition = StopCondition::iterations(state.iterations_this_run() + 1);
        search.run(&mut state).unwrap();
        assert!(forbidden(&state) <= 1, "only the one move just made");
    }

    /// Past the restart budget without a better best, the search starts
    /// again with every weight at one.
    #[test]
    fn restarts_after_steps_without_a_better_best() {
        // Two clauses that cannot both hold, so the best never improves.
        let mut sat = Sat::new(1);
        sat.add_clause([1]);
        sat.add_clause([-1]);
        let mut state = SearchState::new_with_seed(&sat, 0);
        let mut search =
            WeightedLocalSearch::<Sat>::new(StopCondition::iterations(100), (3, 12), 0.0);
        search.run(&mut state).unwrap();
        let raised = |s: &WeightedLocalSearch<Sat>| {
            let w = s.weights().expect("the run built its weights");
            w.weight(0) + w.weight(1)
        };
        assert!(raised(&search) > 20);
        state.iteration += RESTART_STEPS;
        search.run_once(&mut state).unwrap();
        assert_eq!(raised(&search), 2);
    }

    #[test]
    #[should_panic(expected = "smoothing must be within")]
    fn rejects_a_smoothing_outside_the_unit_interval() {
        let _ = WeightedLocalSearch::<Sat>::new(StopCondition::iterations(1), (3, 12), 1.5);
    }
}
