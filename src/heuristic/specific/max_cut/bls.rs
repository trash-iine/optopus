use super::ops;
use crate::error::OptError;
use crate::heuristic::{Heuristic, LocalSearch, StopCondition};
use crate::problem::MaxCut;
use crate::problem::max_cut::MaxCutFlipNeighbor;
use crate::search_state::SearchState;
use rand::Rng;
use rand::rngs::SmallRng;

/// One of the perturbation operators, in Benlic & Hao's vocabulary.
///
/// A *weak* perturbation is directed by the gains and the tabu memory,
/// a *strong* one is random.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerturbationType {
    /// random flip moves, ignoring gains.
    Strong,
    /// tabu flip moves: each taking a move which is the best non-tabu
    /// or satisfies the aspiration rule.
    WeakFlip,
    /// tabu swap moves: each taking the best non-tabu vertex per partition
    /// side, or one that satisfies the aspiration rule.
    WeakSwap,
}

/// Picks the perturbation type for one BLS-style iteration,
///
/// - `omega > t` zeroes the counter (Alg. 1, lines 24-27), so stagnation
///   arrives at the next test as `omega == 0`.
/// - `omega == 0` takes the **random (strong)** perturbation (Alg. 2, line 1).
/// - otherwise `p = max(exp(−omega / t), p0)` (Formula (2)) is the probability
///   of a *directed* (weak) perturbation — `p * q` for the flip variant `A1`,
///   `p * (1 − q)` for the swap variant `A2` — leaving `1 − p` for the random
///   one. `p` decays toward `p0` as `omega` grows, so the random perturbation
///   becomes steadily more likely the longer the best solution stands.
///
/// This selection rule is specific to [`BreakoutLocalSearch`]; the operators it
/// chooses between are the free functions in [`ops`](super::ops), and the other
/// heuristics in this directory drive those same operators from their own
/// schedules.
fn choose_perturbation(
    omega: &mut u64,
    t: u64,
    p0: f64,
    q: f64,
    rng: &mut SmallRng,
) -> PerturbationType {
    if *omega > t {
        *omega = 0;
    }
    if *omega == 0 {
        return PerturbationType::Strong;
    }

    let p = (-(*omega as f64 / t as f64)).exp().max(p0);

    let prob: f64 = rng.random_range(0.0..=1.0);
    if prob <= p {
        if prob <= p * q {
            PerturbationType::WeakFlip
        } else {
            PerturbationType::WeakSwap
        }
    } else {
        PerturbationType::Strong
    }
}

/// The adaptive perturbation schedule of [`BreakoutLocalSearch`]: the `omega`
/// stagnation counter, the perturbation length `l`, and the tunables that
/// drive [`choose_perturbation`].
struct BlsSchedule {
    t: u64,
    p0: f64,
    q: f64,
    l0: u64,
    omega: u64,
    l: u64,
    prev_best_objective: Option<f32>,
    /// The local optimum the previous round ended on — Benlic & Hao's `Cp`.
    ///
    /// This holds the assignment, not its objective value. The paper's rule is
    /// `if C = Cp then L ← L+1 else L ← L0`, and on the G-set the two readings
    /// are nowhere near equivalent: every edge weighs ±1, so cut values are
    /// small integers and distinct local optima collide on the same objective
    /// constantly. Measured on G11 with the paper's `l0 = 8`, the objective
    /// test fired on **82.7%** of rounds and pushed the median `l` to 12 and
    /// its maximum to 80 — a perturbation an order of magnitude stronger than
    /// the paper asks for, applied to the instances with the widest plateaus.
    prev_local_optimum: Option<Vec<bool>>,
}

impl BlsSchedule {
    fn new(t: u64, l0: u64, p0: f64, q: f64) -> Self {
        Self {
            t,
            p0,
            q,
            l0,
            omega: 0,
            l: l0,
            prev_best_objective: None,
            prev_local_optimum: None,
        }
    }

    fn reset(&mut self) {
        self.omega = 0;
        self.l = self.l0;
        self.prev_best_objective = None;
        // Dropped rather than kept: `clear()` also runs when the same
        // schedule is reused on a different instance — a meta-heuristic that
        // rebuilds its sub-problem every round does exactly that. A retained
        // assignment would then be compared against a solution of a different
        // length.
        self.prev_local_optimum = None;
    }

    /// Advances the schedule by one round and returns the perturbation to run
    /// together with its length.
    ///
    /// `omega` grows while the descent fails to beat the objective recorded at
    /// the previous round; `l` grows while the descent keeps landing on the
    /// very same local optimum, and resets to `l0` as soon as it escapes.
    fn next(&mut self, state: &mut SearchState<'_, MaxCut>) -> (PerturbationType, u64) {
        if let Some(prev) = self.prev_best_objective
            && prev >= state.solution.objective
        {
            self.omega += 1;
        } else {
            self.omega = 0;
        }
        self.prev_best_objective = Some(state.best_solution.objective);

        if self.prev_local_optimum.as_deref() == Some(state.solution.x.as_slice()) {
            self.l += 1;
        } else {
            self.l = self.l0;
            match &mut self.prev_local_optimum {
                Some(prev) => prev.clone_from(&state.solution.x),
                None => self.prev_local_optimum = Some(state.solution.x.clone()),
            }
        }

        let perturbation =
            choose_perturbation(&mut self.omega, self.t, self.p0, self.q, &mut state.rng);
        (perturbation, self.l)
    }

    /// Current counter values, for logging.
    fn state(&self) -> (u64, u64) {
        (self.omega, self.l)
    }
}

/// Breakout Local Search (BLS) for the MaxCut problem.
///
/// BLS alternates between a greedy local search phase (with tabu updates) and a
/// perturbation phase. The perturbation type is chosen from the `omega` counter
/// (number of consecutive non-improving local optima), with
/// `p = max(exp(−omega / t), p0)` the probability of a **weak** (directed)
/// perturbation:
///
/// - `omega == 0` — either the last descent improved the global best, or
///   `omega` just passed `t` and was reset: a **strong** (random) perturbation
///   runs.
/// - `0 < omega <= t` (stuck): **weak** perturbation with probability `p * q`
///   (flip) or `p * (1 − q)` (swap), and **strong** (random) otherwise; `p`
///   decays toward `p0` as `omega` grows, so strong perturbations become more
///   likely.
///
/// The perturbation length `l` increases by 1 each time the solution does not change,
/// resetting to `l0` whenever the solution changes.
///
/// # References
///
/// - Benlic, U. and Hao, J.-K. "Breakout Local Search for the Max-Cut problem." *Engineering
///   Applications of Artificial Intelligence*, 26(3), 1162-1173, 2013.
///   [DOI](https://doi.org/10.1016/j.engappai.2012.09.001)
///
/// # Parameters
///
/// - `tabu_tenure` — tabu tenure range `(min, max)` in iterations
/// - `t` — period of the `omega` counter before it resets
/// - `l0` — initial perturbation length
/// - `p0` — minimum perturbation probability
/// - `q` — fraction of weak perturbations that use the flip strategy (vs. swap)
pub struct BreakoutLocalSearch {
    /// The tenure the state's tabu memory records with while this heuristic
    /// drives it. The prohibitions themselves live on the state, where the
    /// descent and the perturbations share them — which is what stops a weak
    /// perturbation undoing the descent.
    tabu_tenure: (u64, u64),
    stop_condition: StopCondition,
    schedule: BlsSchedule,
}

impl BreakoutLocalSearch {
    /// # Panics
    ///
    /// Panics if `tabu_tenure.0 > tabu_tenure.1` (an empty range). The range is
    /// only sampled from once the search is running, so it is checked here.
    pub fn new(
        stop_condition: StopCondition,
        tabu_tenure: (u64, u64),
        t: u64,
        l0: u64,
        p0: f64,
        q: f64,
    ) -> Self {
        crate::common::tabu::assert_valid_tenure(tabu_tenure);
        Self {
            tabu_tenure: paper_effective_tenure(tabu_tenure),
            stop_condition,
            schedule: BlsSchedule::new(t, l0, p0, q),
        }
    }

    /// A BLS whose kicks the *caller* chooses: it is stepped through
    /// [`descend`](Self::descend) and [`kick`](Self::kick), and the schedule
    /// [`run_once`](Heuristic::run_once) would consult is never reached.
    ///
    /// `tabu_tenure` is therefore taken **literally**, without the doubling
    /// [`new`](Self::new) applies: `2γ` is a property of Benlic & Hao's
    /// perturbation rule, and a controller that replaces that rule brings its
    /// own tenure. The schedule the instance still carries is neutral, so
    /// driving it with [`run`](Heuristic::run) instead would merely kick once
    /// per round with `l = 1`.
    ///
    /// # Panics
    ///
    /// Panics if `tabu_tenure.0 > tabu_tenure.1`; see [`new`](Self::new).
    pub fn externally_driven(stop_condition: StopCondition, tabu_tenure: (u64, u64)) -> Self {
        crate::common::tabu::assert_valid_tenure(tabu_tenure);
        Self {
            tabu_tenure,
            stop_condition,
            schedule: BlsSchedule::new(1, 1, 0.0, 0.0),
        }
    }

    /// Points the state's tabu memory at this heuristic's tenure and grows the
    /// map to the instance.
    ///
    /// Called by both halves of a round rather than once per run, because
    /// [`descend`](Self::descend) and [`kick`](Self::kick) are public: a
    /// controller that replaces the paper's schedule
    /// (`examples/rl_bls.rs`) steps them directly and never goes through
    /// [`run`](Heuristic::run). Both operations are idempotent field-level
    /// work.
    fn prepare(&self, state: &mut SearchState<'_, MaxCut>) {
        let n = state.instance.graph.len();
        state.set_tabu_tenure(self.tabu_tenure);
        state.reserve_tabu_vars(n);
    }

    /// The first half of one round: greedy descent to a local optimum, writing
    /// the prohibitions the kick then has to respect.
    ///
    /// Split out because the point *between* the two phases is where a
    /// controller other than the paper's schedule has to act — a learned
    /// policy observes the local optimum it landed on before choosing the next
    /// kick.
    ///
    /// This is the generic [`LocalSearch`], not an operator of its own. The
    /// empty stop condition is the whole budget: `LocalSearch` halts at a local
    /// optimum on its own. Writing the tabu memory — Benlic & Hao's
    /// `H ← Iter + γ`, which sits inside their descent loop — comes from
    /// [`SearchState::apply`](crate::search_state::SearchState::apply), so the
    /// generic descent records exactly what the specialised one did.
    pub fn descend(&mut self, state: &mut SearchState<'_, MaxCut>) -> Result<(), OptError> {
        self.prepare(state);
        LocalSearch::<MaxCutFlipNeighbor>::new(StopCondition::new(None, None, None)).run(state)
    }

    /// The second half of one round: one perturbation of type `perturbation`
    /// and length `l`, against the same prohibitions the descent wrote, followed by
    /// the single `update_best` that closes the round.
    ///
    /// The match below is the only place a [`PerturbationType`] is turned into
    /// an operator: the operators define what each one does and leave the
    /// naming to whoever takes the vocabulary from outside, which is this
    /// method.
    pub fn kick(
        &mut self,
        state: &mut SearchState<'_, MaxCut>,
        perturbation: PerturbationType,
        l: u64,
    ) -> Result<(), OptError> {
        self.prepare(state);
        match perturbation {
            PerturbationType::Strong => ops::random_flips(l, state)?,
            PerturbationType::WeakFlip => ops::tabu_walk(l, state)?,
            PerturbationType::WeakSwap => ops::best_swap(l, state)?,
        }
        state.update_best();
        Ok(())
    }
}

/// Converts Benlic & Hao's tenure parameter `γ` into the prohibition length the
/// engine's tabu map actually stores.
///
/// The paper's tabu list `H` holds "the iteration when the vertex was last
/// moved **plus γ**", and the eligibility predicate of the directed
/// perturbations then asks for `(H_m + γ) < Iter` — so `γ` is counted twice and
/// a vertex stays forbidden for `2γ`. [`TabuMemory`](crate::common::TabuMemory)
/// stores the first iteration at which a move is allowed again, i.e. exactly
/// one tenure, so reproducing the paper means handing it twice the caller's
/// range. `tabu_tenure` therefore keeps the paper's meaning (`rand[3, |V|/10]`
/// on the G-set) instead of silently meaning something else.
///
/// Doubling only the upper bound does not reproduce it — the whole range has to
/// scale. The measurement record is in
/// `docs/heuristics/breakout_local_search.md`.
fn paper_effective_tenure((min, max): (u64, u64)) -> (u64, u64) {
    (min * 2, max * 2)
}

impl Heuristic<MaxCut> for BreakoutLocalSearch {
    /// Resets the schedule only. The prohibitions are the *state's*, and a
    /// sub-run clone — which is how every meta-heuristic starts a phase — comes
    /// with an empty tabu memory already; a caller re-running on one state calls
    /// [`SearchState::reset_tabu`].
    fn clear(&mut self) {
        self.schedule.reset();
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, MaxCut>) -> Result<(), OptError> {
        let (omega, l) = self.schedule.state();
        tracing::debug!(
            iteration = state.iteration,
            omega,
            l,
            "BLS: local search phase start"
        );

        self.descend(state)?;

        let (perturbation_type, l) = self.schedule.next(state);
        let (omega, _) = self.schedule.state();
        tracing::debug!(
            iteration = state.iteration,
            omega,
            l,
            perturbation = ?perturbation_type,
            "BLS: perturbation selected"
        );

        self.kick(state, perturbation_type, l)
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::Heuristic;
    use crate::problem::{MaxCut, MaxCutSolution};
    use crate::search_state::SearchState;

    /// Builds a small toroidal-like graph (degree 4, unit weights) that has both
    /// partition sides populated throughout the search.
    fn small_instance() -> MaxCut {
        let n = 30usize;
        let mut edges = Vec::new();
        for i in 0..n {
            edges.push((i, (i + 1) % n, 1.0));
            edges.push((i, (i + 2) % n, 1.0));
        }
        MaxCut::from_edges(edges)
    }

    /// The descent must stop exactly at a local optimum — no vertex left with a
    /// positive flip gain — and leave the vertices it moved behind in the
    /// state's tabu memory, which is what stops the following perturbation
    /// undoing it.
    ///
    /// That second half is Benlic & Hao's Algorithm 1 line 14, `H <- Iter + γ`,
    /// which sits inside their descent loop. It used to be written out by a
    /// specialised `ops::descent`; now [`BreakoutLocalSearch::descend`] drives
    /// the generic [`LocalSearch`] and the record comes from `apply` itself, so
    /// the property is pinned here rather than in the operator.
    #[test]
    fn descend_reaches_a_local_optimum_and_fills_the_tabu_memory() {
        let mc = small_instance();
        let mut state = SearchState::new_with_seed(&mc, 3);
        let mut bls =
            BreakoutLocalSearch::externally_driven(StopCondition::iterations(u64::MAX), (3, 15));

        let before = state.solution.objective;
        bls.descend(&mut state).unwrap();

        assert!(state.solution.objective >= before, "descent must not lose");
        for v in 0..state.solution.x.len() {
            assert!(
                mc.calculate_gain(&state.solution.x, v) <= 0.0,
                "vertex {v} still improves, so this is not a local optimum"
            );
        }
        assert!(
            (0..state.solution.x.len())
                .any(|v| { !state.tabu_allows(&MaxCutFlipNeighbor::new(&mc, &state.solution, v)) }),
            "the moves it applied must be recorded"
        );
        assert_eq!(
            state.best_solution.objective, state.solution.objective,
            "the local optimum has to be published before returning"
        );
    }

    /// Regression test: BLS must run to completion without erroring.
    ///
    /// The weak-swap perturbation previously returned `Err("No tabu v1")` when a
    /// partition side had no tabu vertex yet. Running the full loop many times
    /// exercises all three perturbation types and the swap fallback; it must
    /// never error and must find a non-trivial cut.
    #[test]
    fn bls_runs_without_error_and_improves() {
        let mc = small_instance();
        for _ in 0..10 {
            let mut state = SearchState::new(&mc);
            let mut bls = BreakoutLocalSearch::new(
                StopCondition::iterations(5_000),
                (3, 15),
                1_000,
                5,
                0.8,
                0.5,
            );
            bls.run(&mut state).expect("BLS must not error");
            assert!(
                state.best_solution.objective > 0.0,
                "BLS should find a positive cut, got {}",
                state.best_solution.objective
            );
        }
    }

    /// On a graph with no edged vertices — as produced by
    /// `SubProblemBasedCrossover` when the two parents disagree only on an
    /// independent set — a full BLS run must terminate cleanly via its stop
    /// condition. (The operator-level counterpart lives in `ops`.)
    #[test]
    fn bls_terminates_on_edgeless_graph() {
        let mc = MaxCut::new(crate::common::Graph::new());
        let mut state = SearchState::new_with_seed(&mc, 0);
        let mut bls = BreakoutLocalSearch::new(
            StopCondition::iterations(10_000).with_failed_updates(500),
            (3, 15),
            1_000,
            5,
            0.8,
            0.5,
        );
        bls.run(&mut state)
            .expect("BLS must terminate on an edgeless graph");
    }

    /// Every draw in the schedule comes from `state.rng`, so a seeded run must
    /// be identical across repetitions.
    #[test]
    fn bls_seeded_run_is_deterministic() {
        let mc = small_instance();
        let run = || {
            let mut state = SearchState::new_with_seed(&mc, 42);
            let mut bls = BreakoutLocalSearch::new(
                StopCondition::iterations(3_000),
                (3, 15),
                1_000,
                5,
                0.8,
                0.5,
            );
            bls.run(&mut state).unwrap();
            (
                state.best_solution.objective,
                state.best_iteration,
                state.best_solution.x.clone(),
            )
        };
        assert_eq!(run(), run());
    }

    /// The perturbation length must grow only when the descent lands on the
    /// *same* local optimum, not merely on one of equal cut value.
    ///
    /// Two independent edges give two distinct assignments of identical
    /// objective: `[T,F,T,F]` and `[T,F,F,T]` both cut 2, and neither is the
    /// complement of the other. Benlic & Hao raise `L` only on `C = Cp`, so
    /// moving between them has to reset `l` to `l0`. Comparing objectives
    /// instead — which is what this used to do — raises it, and on the G-set
    /// that misfires on the large majority of rounds.
    #[test]
    fn l_grows_on_a_repeated_solution_not_a_repeated_objective() {
        let mc = MaxCut::from_edges([(0, 1, 1.0), (2, 3, 1.0)]);
        let l0 = 5;
        let mut schedule = BlsSchedule::new(1_000, l0, 0.8, 0.5);
        let mut state = SearchState::new_with_seed(&mc, 42);

        let set = |state: &mut SearchState<'_, MaxCut>, x: Vec<bool>| {
            state.solution = MaxCutSolution::new_from_assignment(&mc, x);
        };

        set(&mut state, vec![true, false, true, false]);
        let (_, first) = schedule.next(&mut state);
        assert_eq!(first, l0, "the first round has nothing to compare against");

        // Same cut value, different assignment: the search escaped.
        set(&mut state, vec![true, false, false, true]);
        let (_, escaped) = schedule.next(&mut state);
        assert_eq!(
            state.solution.objective, 2.0,
            "both assignments must cut the same weight for this to test anything"
        );
        assert_eq!(escaped, l0, "an escape resets the perturbation length");

        // Identical assignment: the search came back to the same attractor.
        set(&mut state, vec![true, false, false, true]);
        let (_, repeated) = schedule.next(&mut state);
        assert_eq!(
            repeated,
            l0 + 1,
            "returning to Cp lengthens the perturbation"
        );
    }

    /// A tenure the sampler cannot draw from is rejected where it is
    /// configured, not on the first perturbation.
    #[test]
    #[should_panic(expected = "Invalid tabu tenure range")]
    fn an_inverted_tenure_panics_at_construction() {
        BreakoutLocalSearch::new(StopCondition::iterations(10), (9, 2), 100, 5, 0.8, 0.5);
    }
}
