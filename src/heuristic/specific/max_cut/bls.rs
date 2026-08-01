use super::ops::{MaxCutSearchOps, PerturbationType};
use crate::error::OptError;
use crate::heuristic::{Heuristic, StopCondition};
use crate::problem::MaxCut;
use crate::search_state::SearchState;
use rand::Rng;
use rand::rngs::SmallRng;

/// Picks the perturbation type for one BLS-style iteration, following the
/// adaptive scheme of Benlic & Hao: the probability of a directed (weak)
/// perturbation is `p = max(exp(−omega / t), p0)`, so right after an
/// improvement (`omega == 0`, hence `p = 1`) a directed perturbation always
/// runs to gently exploit the fresh region, and as `omega` grows the random
/// (strong) perturbation becomes more likely. Once `omega` exceeds `t` the
/// strongest random perturbation is forced and `omega` is reset.
///
/// `plateau_prob` is an extra draw *inside* the weak branch that switches to
/// an objective-preserving plateau traversal. The draw only happens when
/// `plateau_prob > 0`, so runs with `plateau_prob == 0.0` consume exactly the
/// RNG stream of the original scheme.
///
/// This selection rule is specific to [`BreakoutLocalSearch`]; the operators it
/// chooses between live in [`MaxCutSearchOps`], and the other heuristics in
/// this directory drive those operators from their own schedules.
fn choose_perturbation(
    omega: &mut u64,
    t: u64,
    p0: f64,
    q: f64,
    plateau_prob: f64,
    rng: &mut SmallRng,
) -> PerturbationType {
    if *omega > t {
        *omega = 0;
        return PerturbationType::Strong;
    }

    let p = (-(*omega as f64 / t as f64)).exp().max(p0);

    let prob: f64 = rng.random_range(0.0..=1.0);
    if prob <= p {
        // Weak (directed) branch: optionally traverse the plateau instead.
        if plateau_prob > 0.0 && rng.random_range(0.0..=1.0) <= plateau_prob {
            return PerturbationType::PlateauCluster;
        }
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
    plateau_prob: f64,
    l0: u64,
    omega: u64,
    l: u64,
    prev_best_objective: Option<f32>,
    prev_solution_objective: Option<f32>,
}

impl BlsSchedule {
    fn new(t: u64, l0: u64, p0: f64, q: f64, plateau_prob: f64) -> Self {
        Self {
            t,
            p0,
            q,
            plateau_prob,
            l0,
            omega: 0,
            l: l0,
            prev_best_objective: None,
            prev_solution_objective: None,
        }
    }

    fn reset(&mut self) {
        self.omega = 0;
        self.l = self.l0;
        self.prev_best_objective = None;
        self.prev_solution_objective = None;
    }

    /// Advances the schedule by one round and returns the perturbation to run
    /// together with its length.
    ///
    /// `omega` grows while the descent fails to beat the objective recorded at
    /// the previous round; `l` grows while the solution refuses to change at
    /// all, and resets to `l0` as soon as it does.
    fn next(&mut self, state: &mut SearchState<'_, MaxCut>) -> (PerturbationType, u64) {
        if let Some(prev) = self.prev_best_objective
            && prev >= state.solution.objective
        {
            self.omega += 1;
        } else {
            self.omega = 0;
        }
        self.prev_best_objective = Some(state.best_solution.objective);

        if let Some(prev) = self.prev_solution_objective
            && prev == state.solution.objective
        {
            self.l += 1;
        } else {
            self.l = self.l0;
        }
        self.prev_solution_objective = Some(state.solution.objective);

        let perturbation = choose_perturbation(
            &mut self.omega,
            self.t,
            self.p0,
            self.q,
            self.plateau_prob,
            &mut state.rng,
        );
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
/// perturbation phase. The perturbation type is chosen probabilistically based on
/// the `omega` counter (number of consecutive non-improving iterations). With
/// `p = max(exp(−omega / t), p0)` the probability of a **weak** (directed)
/// perturbation:
///
/// - `omega == 0` (after an improvement, so `p = 1`): always a **weak**
///   perturbation — `flip` with probability `q`, `swap` with probability `1 − q` —
///   to gently exploit the freshly found region.
/// - `0 < omega <= t` (stuck): **weak** perturbation with probability `p * q`
///   (flip) or `p * (1 − q)` (swap), and **strong** (random) otherwise; `p`
///   decays toward `p0` as `omega` grows, so strong perturbations become more
///   likely.
/// - `omega > t`: the strongest **random** perturbation is forced and `omega`
///   is reset to 0.
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
/// - `plateau_prob` — probability that a weak perturbation flips a connected
///   cluster of zero-gain vertices instead (objective-preserving plateau
///   traversal; useful on large sparse instances with wide plateaus). `0.0`
///   reproduces the original Benlic & Hao scheme exactly.
pub struct BreakoutLocalSearch {
    ops: MaxCutSearchOps,
    stop_condition: StopCondition,
    schedule: BlsSchedule,
}

impl BreakoutLocalSearch {
    /// # Panics
    ///
    /// Panics if `plateau_prob` is not within `[0.0, 1.0]`.
    pub fn new(
        stop_condition: StopCondition,
        tabu_tenure: (u64, u64),
        t: u64,
        l0: u64,
        p0: f64,
        q: f64,
        plateau_prob: f64,
    ) -> Self {
        assert!(
            (0.0..=1.0).contains(&plateau_prob),
            "plateau_prob must be within [0.0, 1.0], got {plateau_prob}"
        );
        Self {
            ops: MaxCutSearchOps::new(tabu_tenure),
            stop_condition,
            schedule: BlsSchedule::new(t, l0, p0, q, plateau_prob),
        }
    }
}

impl Heuristic<MaxCut> for BreakoutLocalSearch {
    fn clear(&mut self) {
        self.schedule.reset();
        self.ops.clear();
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, MaxCut>) -> Result<(), OptError> {
        self.ops.ensure_capacity(state.instance.graph.len());

        let (omega, l) = self.schedule.state();
        tracing::debug!(
            iteration = state.iteration,
            omega,
            l,
            "BLS: local search phase start"
        );

        self.ops.descent(state)?;

        let (perturbation_type, l) = self.schedule.next(state);
        let (omega, _) = self.schedule.state();
        tracing::debug!(
            iteration = state.iteration,
            omega,
            l,
            perturbation = ?perturbation_type,
            "BLS: perturbation selected"
        );

        self.ops.perturb(perturbation_type, l, state)?;

        // Update best once after the perturbation phase completes.
        state.update_best();

        Ok(())
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::Heuristic;
    use crate::problem::MaxCut;
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

    /// Regression test: BLS must run to completion without erroring.
    ///
    /// The weak-swap perturbation previously returned `Err("No tabu v1")` when a
    /// partition side had no tabu vertex yet — a path that is hit frequently now
    /// that directed (weak) perturbations run at `omega == 0`. Running the full
    /// loop many times exercises all three perturbation types and the swap
    /// fallback; it must never error and must find a non-trivial cut.
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
                0.0,
            );
            bls.run(&mut state).expect("BLS must not error");
            assert!(
                state.best_solution.objective > 0.0,
                "BLS should find a positive cut, got {}",
                state.best_solution.objective
            );
        }
    }

    /// BLS with plateau perturbations enabled must also run cleanly and find a
    /// non-trivial cut (exercises PlateauCluster through the full loop).
    #[test]
    fn bls_with_plateau_runs_without_error_and_improves() {
        let mc = small_instance();
        for seed in 0..10 {
            let mut state = SearchState::new_with_seed(&mc, seed);
            let mut bls = BreakoutLocalSearch::new(
                StopCondition::iterations(5_000),
                (3, 15),
                1_000,
                5,
                0.8,
                0.5,
                0.5,
            );
            bls.run(&mut state).expect("BLS+plateau must not error");
            assert!(state.best_solution.objective > 0.0);
        }
    }

    #[test]
    #[should_panic(expected = "plateau_prob must be within [0.0, 1.0]")]
    fn bls_rejects_invalid_plateau_prob() {
        let _ = BreakoutLocalSearch::new(
            StopCondition::iterations(1),
            (3, 15),
            1_000,
            5,
            0.8,
            0.5,
            1.5,
        );
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
            0.5,
        );
        bls.run(&mut state)
            .expect("BLS must terminate on an edgeless graph");
    }

    /// Seeded regression guard for the selection-rule refactor: with
    /// `plateau_prob = 0.0` no extra RNG draw happens, so a seeded run must be
    /// deterministic and identical across repetitions.
    #[test]
    fn bls_plateau_prob_zero_is_deterministic() {
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
                0.0,
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
}
