//! Breakout Local Search over any problem.

use crate::error::OptError;
use crate::heuristic::{Heuristic, StopCondition};
use crate::search_state::SearchState;
use crate::trait_defs::{Distance, Evaluate, ProblemTrait};
use rand::Rng;

/// The two decisions BLS takes between a descent and the kick that follows
/// it, together with the history they read.
///
/// The perturbation length is the implementor's own state rather than
/// something the loop threads back in. Handing it out and taking it again
/// would only let a caller return a length the schedule never produced.
pub trait PerturbationSchedule<P: ProblemTrait> {
    /// Drops everything learned about the current episode. Called from
    /// [`Heuristic::clear`].
    fn reset(&mut self);

    /// How far the kick goes, called with the local optimum the descent just
    /// reached.
    ///
    /// This is where a schedule writes. Whatever it records about a round is
    /// recorded here, and [`select`](Self::select) only reads it, which keeps
    /// the two decisions from disagreeing about the round they are deciding
    /// for.
    fn determine_jump_magnitude(&mut self, state: &mut SearchState<'_, P>) -> u64;

    /// Which perturbation to apply.
    ///
    /// The schedule owns the operators it chooses between, so there is no
    /// index to agree on and no order for a caller to get wrong.
    ///
    /// Both decisions are handed the state, as
    /// [`Heuristic::run_once`](crate::heuristic::Heuristic::run_once) is. A
    /// rule that picks its operator from where the search has got to is an
    /// ordinary thing to want, and the length decision above already reads the
    /// state for it. Draw from `state.rng` rather than an RNG of your own,
    /// which is what keeps a seeded run reproducible.
    fn select<'s>(&'s mut self, state: &mut SearchState<'_, P>) -> &'s mut dyn Heuristic<P>;

    /// Drops whatever the operators accumulated, alongside
    /// [`reset`](Self::reset)'s own state.
    fn clear_perturbations(&mut self);

    /// Called once the kick and its `update_best` have closed the round.
    ///
    /// [`determine_jump_magnitude`](Self::determine_jump_magnitude) sees the
    /// state after a descent, which is the wrong moment to read anything the
    /// kick is about to change. A schedule that wants to know what a descent
    /// alone did has to remember the value this hands it, since that is the
    /// same value the next descent starts from. A schedule that does not need
    /// it leaves the default, which does nothing.
    fn round_ended(&mut self, _state: &SearchState<'_, P>) {}
}

/// Benlic & Hao's Max-Cut schedule: the `omega` stagnation counter and the
/// perturbation length `l`.
///
/// Their `DetermineJumpMagnitude` is
/// [`determine_jump_magnitude`](PerturbationSchedule::determine_jump_magnitude)
/// and their `DeterminePerturbationType` is
/// [`select`](PerturbationSchedule::select), which is the division the trait
/// asks for, the first of the two writing the history and the second only
/// reading it.
///
/// `omega` counts consecutive rounds whose descent failed to beat the best
/// recorded a round ago. `p = max(exp(-omega / t), p0)` is then the probability
/// of a directed perturbation, and it decays toward `p0` as `omega` grows, so
/// the random one becomes steadily more likely the longer the best stands.
///
/// The operators are held here rather than indexed out of a bank the search
/// owns, so which one is the random one is something this was built with
/// rather than a position everyone has to agree on. `directed` carries a share
/// of the directed probability per operator, and Benlic & Hao's `q` is the
/// two-operator case, `[(first, q), (second, 1 - q)]`.
///
/// # Parameters
///
/// - `t`, period of the `omega` counter before it resets
/// - `l0`, initial perturbation length
/// - `p0`, minimum probability of a directed perturbation
pub struct AdaptivePerturbation<P: ProblemTrait> {
    t: u64,
    p0: f64,
    l0: u64,
    omega: u64,
    l: u64,
    prev_best_energy: Option<f64>,
    /// The local optimum the previous round ended on, Benlic & Hao's `Cp`.
    ///
    /// This holds the solution, not its objective value. The rule is to
    /// lengthen the perturbation when the descent lands on the same local
    /// optimum as the round before and to reset it otherwise, and the two
    /// readings of "the same" come apart wherever distinct local optima share
    /// an objective, which on a unit-weight graph they do constantly. Reading the objective there grows
    /// `l` on rounds that did escape, so the perturbation strengthens on the
    /// instances with the widest plateaus, which are the ones it should not.
    ///
    /// Sameness is `Distance::distance == 0` rather than `PartialEq`, since
    /// that is the comparison a problem already defines for its solutions and
    /// it reads only the assignment, not the caches beside it.
    prev_local_optimum: Option<P::Solution>,
    /// The perturbation that reads no history, run when the best has just
    /// moved or when the draw falls outside `p`.
    random: Box<dyn Heuristic<P>>,
    /// The perturbations that do read the history, each with its share of the
    /// directed probability. The shares are taken to sum to one.
    directed: Vec<(Box<dyn Heuristic<P>>, f64)>,
}

impl<P: ProblemTrait> AdaptivePerturbation<P>
where
    P::Solution: Distance + Evaluate,
{
    /// # Panics
    ///
    /// Panics if `directed` is empty, since the rule has to have something to
    /// choose when the draw falls inside `p`, or if any share is negative.
    pub fn new(
        t: u64,
        l0: u64,
        p0: f64,
        random: Box<dyn Heuristic<P>>,
        directed: Vec<(Box<dyn Heuristic<P>>, f64)>,
    ) -> Self {
        assert!(
            !directed.is_empty(),
            "AdaptivePerturbation needs at least one directed perturbation"
        );
        assert!(
            directed.iter().all(|(_, share)| *share >= 0.0),
            "a directed share cannot be negative"
        );
        Self {
            t,
            p0,
            l0,
            omega: 0,
            l: l0,
            prev_best_energy: None,
            prev_local_optimum: None,
            random,
            directed,
        }
    }
}

impl<P: ProblemTrait> PerturbationSchedule<P> for AdaptivePerturbation<P>
where
    P::Solution: Distance + Evaluate,
{
    fn reset(&mut self) {
        self.omega = 0;
        self.l = self.l0;
        self.prev_best_energy = None;
        // Dropped rather than kept: `clear()` also runs when the same schedule
        // is reused on a different instance, which a meta-heuristic that
        // rebuilds its sub-problem every round does. A retained solution would
        // then be compared against one of a different size.
        self.prev_local_optimum = None;
    }

    fn clear_perturbations(&mut self) {
        self.random.clear();
        for (op, _) in &mut self.directed {
            op.clear();
        }
    }

    fn determine_jump_magnitude(&mut self, state: &mut SearchState<'_, P>) -> u64 {
        let current = state.solution.evaluate().minimized();
        if let Some(prev) = self.prev_best_energy
            && prev <= current
        {
            self.omega += 1;
        } else {
            self.omega = 0;
        }
        self.prev_best_energy = Some(state.best_solution.evaluate().minimized());

        let unchanged = self
            .prev_local_optimum
            .as_ref()
            .is_some_and(|prev| prev.distance(&state.solution) == 0);
        if unchanged {
            self.l += 1;
        } else {
            self.l = self.l0;
            match &mut self.prev_local_optimum {
                Some(prev) => prev.clone_from(&state.solution),
                None => self.prev_local_optimum = Some(state.solution.clone()),
            }
        }

        // `omega > t` zeroes the counter, so stagnation arrives at the type
        // decision as `omega == 0`. It is a write to the schedule's own
        // history, and the paper has only this procedure write.
        if self.omega > self.t {
            self.omega = 0;
        }
        self.l
    }

    fn select<'s>(&'s mut self, state: &mut SearchState<'_, P>) -> &'s mut dyn Heuristic<P> {
        if self.omega == 0 {
            return self.random.as_mut();
        }
        let p = (-(self.omega as f64 / self.t as f64)).exp().max(self.p0);
        let prob: f64 = state.rng.random_range(0.0..=1.0);
        if prob > p {
            return self.random.as_mut();
        }
        // The shares partition `p`, so walking their running sum picks the
        // same operator a chain of thresholds would. The last one is returned
        // rather than compared against, because the sum of `p * share` is not
        // exactly `p` in floating point and the draw is already known to be
        // inside it.
        let last = self.directed.len() - 1;
        let mut bound = 0.0;
        for i in 0..last {
            bound += p * self.directed[i].1;
            if prob <= bound {
                return self.directed[i].0.as_mut();
            }
        }
        self.directed[last].0.as_mut()
    }
}

/// Breakout Local Search over any problem.
///
/// A descent, a bank of perturbations, and a schedule that picks one of them
/// and how far to go. None of the three needs the problem to be binary, which
/// is why this takes only `P: ProblemTrait`. The perturbations are ordinary
/// heuristics, stepped one move at a time, so a perturbation of length `L` is
/// `L` calls to `run_once`.
///
/// # What the framework supplies
///
/// BLS is not one algorithm. The paper states it as four procedures a concrete
/// BLS fills in, and the vertex separator problem it is introduced on is not a
/// binary problem at all.
///
/// `DescentBasedSearch` and `Perturb` are ordinary heuristics here.
/// `DetermineJumpMagnitude` and `DeterminePerturbationType`, the two decisions
/// taken between a descent and the kick that follows it, are the
/// [`PerturbationSchedule`] trait, which owns the operators it chooses
/// between.
///
/// What is left over is not BLS's to own. The initial solution comes from
/// [`SearchState`], a library-wide convention, and the loop that repeats the
/// round until the stopping condition is [`Heuristic::run`]'s. One
/// [`run_once`](Heuristic::run_once) is one round, a descent, the two
/// decisions, then the kick.
///
/// # The shared history
///
/// The history the three procedures read is the tabu memory on the
/// [`SearchState`]. The descent writes it, forbidding each vertex it moves for
/// a tenure's worth of iterations, which is where Benlic & Hao put the write
/// too, and the directed perturbations read it. That is what stops a
/// perturbation undoing the descent that just ran.
///
/// # References
///
/// - Benlic, U. and Hao, J.-K. "Breakout local search for the vertex separator
///   problem." IJCAI 2013, 461-467. The general framework.
/// - Benlic, U. and Hao, J.-K. "Breakout Local Search for the Max-Cut problem."
///   Engineering Applications of Artificial Intelligence, 26(3), 1162-1173,
///   2013. The instantiation [`AdaptivePerturbation`] implements.
pub struct BreakoutLocalSearch<P: ProblemTrait, S> {
    /// The tenure the state's tabu memory records with, set before the descent
    /// and again before each kick.
    ///
    /// It governs the descent and any bank entry that carries no tenure of its
    /// own. A bank entry that does carry one, as
    /// [`TabuSearch`](crate::heuristic::TabuSearch) and MaxCut's directed swap
    /// both do, installs its own on the state the moment it is stepped and this
    /// value never reaches it. Build the bank from the same tenure, which is
    /// what `bls_for_max_cut` does, or the descent will
    /// prohibit for one length and those kicks for another.
    tabu_tenure: (u64, u64),
    stop_condition: StopCondition,
    /// The descent. Driven with `run`, so it reaches a local optimum and stops
    /// there rather than taking one move.
    descent: Box<dyn Heuristic<P>>,
    /// The schedule, which also owns the perturbations it chooses between.
    schedule: S,
}

impl<P: ProblemTrait, S: PerturbationSchedule<P>> BreakoutLocalSearch<P, S> {
    /// # Panics
    ///
    /// Panics if `tabu_tenure.0 > tabu_tenure.1` (an empty range), which is
    /// only noticed once the search is running.
    pub fn new(
        stop_condition: StopCondition,
        tabu_tenure: (u64, u64),
        descent: Box<dyn Heuristic<P>>,
        schedule: S,
    ) -> Self {
        crate::common::tabu::assert_valid_tenure(tabu_tenure);
        Self {
            tabu_tenure,
            stop_condition,
            descent,
            schedule,
        }
    }

    /// The schedule this was built with.
    ///
    /// A schedule that learns is worth reading back after a run, which is what
    /// `examples/rl_bls.rs` does with its bandit weights.
    pub fn schedule(&self) -> &S {
        &self.schedule
    }
}

impl<P: ProblemTrait, S: PerturbationSchedule<P>> Heuristic<P> for BreakoutLocalSearch<P, S> {
    fn clear(&mut self) {
        self.schedule.reset();
        self.schedule.clear_perturbations();
        self.descent.clear();
    }

    /// One round, a descent, the two decisions, then the kick.
    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        state.start_record_tabu(self.tabu_tenure);
        self.descent.run(state)?;

        let l = self.schedule.determine_jump_magnitude(state);
        tracing::debug!(iteration = state.iteration, l, "BLS: perturbation selected");

        // The kick runs against the prohibitions the descent just wrote, and
        // the round closes on a single `update_best`.
        state.start_record_tabu(self.tabu_tenure);
        let op = self.schedule.select(state);
        for _ in 0..l {
            op.run_once(state)?;
        }
        state.update_best();
        self.schedule.round_ended(state);
        Ok(())
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::{LocalSearch, RandomWalk, TabuSearch};
    use crate::problem::{MaxCut, MaxCutFlipNeighbor, Tsp, TspRelocateNeighbor, TspTwoOptNeighbor};

    /// A schedule with one directed perturbation is an ordinary setting now,
    /// where before it was a bank the constructor accepted and `kick` panicked
    /// on at whatever iteration the draw first fell inside `p`.
    #[test]
    fn a_single_directed_perturbation_is_a_valid_schedule() {
        let mc = crate::problem::max_cut::test_fixtures::small_instance();
        let mut state = SearchState::new_with_seed(&mc, 4);
        let initial = state.solution.objective;

        let unreachable = || StopCondition::iterations(u64::MAX);
        let mut bls = BreakoutLocalSearch::new(
            StopCondition::iterations(400),
            (3, 9),
            Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(unreachable())),
            AdaptivePerturbation::<MaxCut>::new(
                50,
                4,
                0.8,
                Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(unreachable())),
                vec![(
                    Box::new(TabuSearch::<MaxCutFlipNeighbor>::new(unreachable(), (3, 9))),
                    1.0,
                )],
            ),
        );
        bls.run(&mut state).unwrap();

        assert!(state.best_solution.objective >= initial);
        // A round runs to its end, so the budget is checked between rounds and
        // the last kick carries the counter past it.
        assert!(state.iteration >= 400);
    }

    #[test]
    #[should_panic(expected = "at least one directed perturbation")]
    fn a_schedule_with_nothing_directed_is_rejected() {
        let unreachable = StopCondition::iterations(u64::MAX);
        let _ = AdaptivePerturbation::<MaxCut>::new(
            50,
            4,
            0.8,
            Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(unreachable)),
            vec![],
        );
    }

    /// The framework asks only for `ProblemTrait`, and the paper it comes from
    /// introduces BLS on a problem that is not binary at all. This builds one
    /// over TSP out of two generic heuristics, which is the statement that
    /// nothing in the search reads a variable.
    ///
    /// TSP is not registered as a BLS problem and no tuning here is measured.
    /// What is under test is that the type checks and the round runs.
    #[test]
    fn the_framework_runs_on_a_non_binary_problem() {
        let coordinates: Vec<(f64, f64)> = (0..20)
            .map(|i| {
                let a = f64::from(i) * 0.31;
                (a.cos() * 100.0, a.sin() * 100.0)
            })
            .collect();
        let tsp = Tsp::new("ring20".to_string(), coordinates);
        let mut state = SearchState::new_with_seed(&tsp, 7);
        let initial = state.solution.objective;

        let mut bls = BreakoutLocalSearch::new(
            StopCondition::iterations(500),
            (3, 9),
            Box::new(LocalSearch::<TspTwoOptNeighbor>::new(
                StopCondition::iterations(u64::MAX),
            )),
            AdaptivePerturbation::<Tsp>::new(
                200,
                3,
                0.8,
                Box::new(RandomWalk::<TspRelocateNeighbor>::new(
                    StopCondition::iterations(u64::MAX),
                )),
                vec![
                    (
                        Box::new(TabuSearch::<TspRelocateNeighbor>::new(
                            StopCondition::iterations(u64::MAX),
                            (3, 9),
                        )),
                        0.5,
                    ),
                    (
                        Box::new(TabuSearch::<TspTwoOptNeighbor>::new(
                            StopCondition::iterations(u64::MAX),
                            (3, 9),
                        )),
                        0.5,
                    ),
                ],
            ),
        );
        bls.run(&mut state).unwrap();

        assert!(
            state.best_solution.objective <= initial,
            "TSP minimizes, so the tour must not get longer"
        );
    }
}
