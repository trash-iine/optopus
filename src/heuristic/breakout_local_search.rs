//! Breakout Local Search, as the framework Benlic & Hao state it.
//!
//! BLS is not one algorithm. The paper defines it as four procedures a concrete
//! BLS supplies, and the vertex separator problem it is introduced on is not a
//! binary problem at all:
//!
//! ```text
//! Algorithm 1  BLS general framework
//! 1: p0 <- GenerateInitialSolution
//! 2: L  <- L0
//! 3: while stopping condition not reached do
//! 4:     p  <- DescentBasedSearch(p0)
//! 5:     L  <- DetermineJumpMagnitude(L, p, history)
//! 6:     T  <- DeterminePerturbationType(p, history)
//! 7:     p0 <- Perturb(L, T, p, history)
//! 8: end while
//! ```
//!
//! Line 1 is `SearchState`'s initial solution, which is a library-wide
//! convention rather than something a heuristic owns. Line 2 is the schedule's
//! `reset`, line 3 is `Heuristic::run`'s loop, and `run_once` is lines 4 to 7,
//! one line each. Lines 4 and 7 are ordinary heuristics, and lines 5 and 6 are
//! the [`PerturbationSchedule`] trait.
//!
//! The history the three procedures share is the tabu memory on the
//! `SearchState`. The descent writes it, Benlic & Hao's `H <- Iter + gamma`
//! sitting inside their descent loop, and the directed perturbations read it.

use crate::error::OptError;
use crate::heuristic::{Heuristic, StopCondition};
use crate::search_state::SearchState;
use crate::trait_defs::{Distance, Evaluate, ProblemTrait};
use rand::Rng;

/// Lines 5 and 6 of Algorithm 1, together with the history they read.
///
/// `L` is a loop variable in the paper and a field of the implementor here.
/// The information is the same, and threading it through the loop would only
/// let a caller hand back an `L` the schedule never produced.
pub trait PerturbationSchedule<P: ProblemTrait> {
    /// Drops everything learned about the current episode. Called from
    /// [`Heuristic::clear`].
    fn reset(&mut self);

    /// Line 5, `L <- DetermineJumpMagnitude(L, p, history)`. Called with the
    /// local optimum the descent just reached.
    ///
    /// Whatever the schedule records about a round is recorded here, which is
    /// the paper's division of labour. Its Algorithm 2 writes the hash table
    /// and the last-cycle marker, and Equation 2, the type decision, only reads
    /// them.
    fn determine_jump_magnitude(&mut self, state: &SearchState<'_, P>) -> u64;

    /// Line 6, `T <- DeterminePerturbationType(p, history)`, as an index into
    /// the perturbation bank the search was built with.
    ///
    /// Takes the state mutably for its RNG. Equation 3 draws a
    /// `random(0, 0.01, ..., 1)` to threshold on.
    fn determine_perturbation_type(&mut self, state: &mut SearchState<'_, P>) -> usize;
}

/// Benlic & Hao's Max-Cut schedule: the `omega` stagnation counter and the
/// perturbation length `l`.
///
/// `omega` counts consecutive rounds whose descent failed to beat the best
/// recorded a round ago. `p = max(exp(-omega / t), p0)` is then the probability
/// of a directed perturbation, and it decays toward `p0` as `omega` grows, so
/// the random one becomes steadily more likely the longer the best stands.
///
/// The bank it indexes is expected to be `[random, directed, directed]`, which
/// is what `BreakoutLocalSearch::for_max_cut` builds. `q` splits the directed
/// probability between the two.
///
/// # Parameters
///
/// - `t`, period of the `omega` counter before it resets
/// - `l0`, initial perturbation length
/// - `p0`, minimum probability of a directed perturbation
/// - `q`, fraction of directed perturbations that take the first of the two
pub struct AdaptivePerturbation<P: ProblemTrait>
where
    P::Solution: Distance + Evaluate,
{
    t: u64,
    p0: f64,
    q: f64,
    l0: u64,
    omega: u64,
    l: u64,
    prev_best_energy: Option<f64>,
    /// The local optimum the previous round ended on, Benlic & Hao's `Cp`.
    ///
    /// This holds the solution, not its objective value. The paper's rule is
    /// `if C = Cp then L <- L+1 else L <- L0`, and on the G-set the two
    /// readings are nowhere near equivalent: every edge weighs +/-1, so cut
    /// values are small integers and distinct local optima collide on the same
    /// objective constantly. Measured on G11 with the paper's `l0 = 8`, the
    /// objective test fired on 82.7% of rounds and pushed the median `l` to 12
    /// and its maximum to 80, a perturbation an order of magnitude stronger
    /// than the paper asks for, applied to the instances with the widest
    /// plateaus.
    ///
    /// Sameness is `Distance::distance == 0` rather than `PartialEq`, since
    /// that is the comparison a problem already defines for its solutions and
    /// it reads only the assignment, not the caches beside it.
    prev_local_optimum: Option<P::Solution>,
}

/// The bank index the Max-Cut schedule expects at each position.
const STRONG: usize = 0;
const DIRECTED_FIRST: usize = 1;
const DIRECTED_SECOND: usize = 2;

impl<P: ProblemTrait> AdaptivePerturbation<P>
where
    P::Solution: Distance + Evaluate,
{
    pub fn new(t: u64, l0: u64, p0: f64, q: f64) -> Self {
        Self {
            t,
            p0,
            q,
            l0,
            omega: 0,
            l: l0,
            prev_best_energy: None,
            prev_local_optimum: None,
        }
    }

    /// Current `(omega, l)`, for logging.
    pub fn counters(&self) -> (u64, u64) {
        (self.omega, self.l)
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

    fn determine_jump_magnitude(&mut self, state: &SearchState<'_, P>) -> u64 {
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
        // history, and Algorithm 1 has only this procedure write.
        if self.omega > self.t {
            self.omega = 0;
        }
        self.l
    }

    fn determine_perturbation_type(&mut self, state: &mut SearchState<'_, P>) -> usize {
        if self.omega == 0 {
            return STRONG;
        }
        let p = (-(self.omega as f64 / self.t as f64)).exp().max(self.p0);
        let prob: f64 = state.rng.random_range(0.0..=1.0);
        if prob <= p {
            if prob <= p * self.q {
                DIRECTED_FIRST
            } else {
                DIRECTED_SECOND
            }
        } else {
            STRONG
        }
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
/// All three phases share the tabu memory on the `SearchState`, which is what
/// stops a directed perturbation undoing the descent that just ran.
///
/// # References
///
/// - Benlic, U. and Hao, J.-K. "Breakout local search for the vertex separator
///   problem." IJCAI 2013, 461-467. The general framework.
/// - Benlic, U. and Hao, J.-K. "Breakout Local Search for the Max-Cut problem."
///   Engineering Applications of Artificial Intelligence, 26(3), 1162-1173,
///   2013. The instantiation [`AdaptivePerturbation`] implements.
pub struct BreakoutLocalSearch<P: ProblemTrait, S> {
    /// The tenure the state's tabu memory records with while this heuristic
    /// drives it.
    tabu_tenure: (u64, u64),
    stop_condition: StopCondition,
    /// Line 4. Driven with `run`, so it descends to a local optimum and stops
    /// there rather than taking one move.
    descent: Box<dyn Heuristic<P>>,
    /// Line 7, one entry per perturbation type. The schedule returns an index
    /// into this.
    perturbations: Vec<Box<dyn Heuristic<P>>>,
    schedule: S,
}

impl<P: ProblemTrait, S> BreakoutLocalSearch<P, S> {
    /// # Panics
    ///
    /// Panics if `tabu_tenure.0 > tabu_tenure.1` (an empty range), or if
    /// `perturbations` is empty. Both are only noticed once the search is
    /// running, so they are checked here.
    pub fn new(
        stop_condition: StopCondition,
        tabu_tenure: (u64, u64),
        descent: Box<dyn Heuristic<P>>,
        perturbations: Vec<Box<dyn Heuristic<P>>>,
        schedule: S,
    ) -> Self {
        crate::common::tabu::assert_valid_tenure(tabu_tenure);
        assert!(
            !perturbations.is_empty(),
            "BLS needs at least one perturbation to choose from"
        );
        Self {
            tabu_tenure,
            stop_condition,
            descent,
            perturbations,
            schedule,
        }
    }

    /// How many perturbation types the bank holds, which is the range a
    /// schedule's index has to stay inside.
    pub fn num_perturbations(&self) -> usize {
        self.perturbations.len()
    }

    /// Line 4: descend to a local optimum, writing the prohibitions the kick
    /// then has to respect.
    ///
    /// Public because the point between the two halves is where a controller
    /// other than a [`PerturbationSchedule`] has to act. A learned policy
    /// observes the local optimum it landed on before choosing the next kick.
    pub fn descend(&mut self, state: &mut SearchState<'_, P>) -> Result<(), OptError> {
        state.start_record_tabu(self.tabu_tenure);
        self.descent.run(state)
    }

    /// Line 7: `l` moves of perturbation type `perturbation`, against the same
    /// prohibitions the descent wrote, followed by the single `update_best`
    /// that closes the round.
    ///
    /// # Panics
    ///
    /// Panics if `perturbation` is not a valid index into the bank.
    pub fn kick(
        &mut self,
        state: &mut SearchState<'_, P>,
        perturbation: usize,
        l: u64,
    ) -> Result<(), OptError> {
        state.start_record_tabu(self.tabu_tenure);
        let op = self
            .perturbations
            .get_mut(perturbation)
            .expect("the schedule returned a perturbation index the bank does not hold");
        for _ in 0..l {
            op.run_once(state)?;
        }
        state.update_best();
        Ok(())
    }
}

impl<P: ProblemTrait, S: PerturbationSchedule<P>> Heuristic<P> for BreakoutLocalSearch<P, S> {
    fn clear(&mut self) {
        self.schedule.reset();
        self.descent.clear();
        for op in &mut self.perturbations {
            op.clear();
        }
    }

    /// One round of Algorithm 1, lines 4 to 7, one line each.
    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        self.descend(state)?;
        let l = self.schedule.determine_jump_magnitude(state);
        let perturbation = self.schedule.determine_perturbation_type(state);
        tracing::debug!(
            iteration = state.iteration,
            l,
            perturbation,
            "BLS: perturbation selected"
        );
        self.kick(state, perturbation, l)
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::{LocalSearch, RandomWalk, TabuSearch};
    use crate::problem::{TspRelocateNeighbor, TspTwoOptNeighbor, TspWithCoordinates};

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
        let tsp = TspWithCoordinates::new("ring20".to_string(), coordinates);
        let mut state = SearchState::new_with_seed(&tsp, 7);
        let initial = state.solution.objective;

        let mut bls = BreakoutLocalSearch::new(
            StopCondition::iterations(500),
            (3, 9),
            Box::new(LocalSearch::<TspTwoOptNeighbor>::new(
                StopCondition::iterations(u64::MAX),
            )),
            // `AdaptivePerturbation` indexes three, one random and two
            // directed, which is the shape Benlic & Hao's rule chooses between.
            vec![
                Box::new(RandomWalk::<TspRelocateNeighbor>::new(
                    StopCondition::iterations(u64::MAX),
                )),
                Box::new(TabuSearch::<TspRelocateNeighbor>::new(
                    StopCondition::iterations(u64::MAX),
                    (3, 9),
                )),
                Box::new(TabuSearch::<TspTwoOptNeighbor>::new(
                    StopCondition::iterations(u64::MAX),
                    (3, 9),
                )),
            ],
            AdaptivePerturbation::<TspWithCoordinates>::new(200, 3, 0.8, 0.5),
        );
        bls.run(&mut state).unwrap();

        assert!(
            state.best_solution.objective <= initial,
            "TSP minimizes, so the tour must not get longer"
        );
    }
}
