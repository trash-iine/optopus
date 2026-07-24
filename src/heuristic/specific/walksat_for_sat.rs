//! WalkSAT-style stochastic local search for MaxSAT.
//!
//! The generic [`LocalSearch`](crate::heuristic::LocalSearch) /
//! [`TabuSearch`](crate::heuristic::TabuSearch) scan **all** `n` variables every
//! step (`O(n)` per move), which is prohibitively slow on the large instances
//! this heuristic targets. WalkSAT instead keeps the search *focused*: each step
//! it samples a currently **unsatisfied clause** and flips one variable *inside
//! it*, so the per-step cost is `O(clause length × variable degree)` and is
//! independent of the total variable count. Because every literal of an unsat
//! clause is false, flipping any of its variables necessarily satisfies that
//! clause; the choice between them follows the classic Selman–Kautz–Cohen (SKC)
//! rule based on each variable's *break count* (how many other clauses its flip
//! would turn unsatisfied) plus a noise parameter for diversification.
//!
//! The move is not expressible through the uniform [`SatFlipNeighbor`] iterator,
//! so — like Breakout Local Search for MaxCut — it lives here as a
//! problem-specific heuristic. It keeps its own scratch state (satisfying-literal
//! counts, an unsatisfied-clause list, and a variable→clause index) and leaves
//! [`Sat`] / [`SatSolution`](crate::problem::SatSolution) untouched. Multi-restart
//! is composed externally via [`Restart`](crate::heuristic::Restart) /
//! [`Iterated`](crate::heuristic::Iterated).

use super::super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::problem::Sat;
use crate::search_state::SearchState;
use rand::Rng;

/// Sentinel for "clause not currently in the unsatisfied list".
const NOT_IN_UNSAT: usize = usize::MAX;

// Adaptive-noise schedule (Hoos 2002; see [`WalkSatForSat`] References).
/// Fraction of the clause count that may pass without improvement before the
/// noise is raised.
const ADAPT_THETA: f64 = 1.0 / 6.0;
/// Multiplicative step by which the noise is raised / lowered.
const ADAPT_PHI: f64 = 0.2;

/// Incrementally maintained scratch state for the focused move.
///
/// All three structures are rebuilt once from the current assignment at the
/// start of a run and then updated in `O(degree)` per flip.
struct WalkSatScratch {
    /// `true_count[c]` = number of literal occurrences in clause `c` that are
    /// currently satisfied. Clause `c` is satisfied iff `true_count[c] > 0`.
    true_count: Vec<u32>,
    /// Dense list of the currently unsatisfied clause indices.
    unsat_clauses: Vec<usize>,
    /// `pos_in_unsat[c]` = index of `c` within `unsat_clauses`, or
    /// [`NOT_IN_UNSAT`] when clause `c` is satisfied. Enables `O(1)` membership.
    pos_in_unsat: Vec<usize>,
    /// `var_clause_lits[v]` = `(clause index, literal is positive)` for every
    /// occurrence of variable `v`. Drives both break-count and incremental update.
    var_clause_lits: Vec<Vec<(u32, bool)>>,
}

impl WalkSatScratch {
    /// Builds the scratch state from scratch for assignment `x`.
    fn build(instance: &Sat, x: &[bool]) -> Self {
        let n_clauses = instance.n_clauses();
        let n_vars = instance.n_vars();

        let mut true_count = vec![0u32; n_clauses];
        let mut var_clause_lits: Vec<Vec<(u32, bool)>> = vec![Vec::new(); n_vars];

        for (c, clause) in instance.all_clauses().enumerate() {
            for &lit in clause {
                let v = lit.unsigned_abs() as usize - 1;
                let lit_positive = lit > 0;
                var_clause_lits[v].push((c as u32, lit_positive));
                if x[v] == lit_positive {
                    true_count[c] += 1;
                }
            }
        }

        let mut unsat_clauses = Vec::new();
        let mut pos_in_unsat = vec![NOT_IN_UNSAT; n_clauses];
        for (c, &tc) in true_count.iter().enumerate() {
            if tc == 0 {
                pos_in_unsat[c] = unsat_clauses.len();
                unsat_clauses.push(c);
            }
        }

        Self {
            true_count,
            unsat_clauses,
            pos_in_unsat,
            var_clause_lits,
        }
    }

    /// Number of clauses that variable `v` currently satisfies *alone* — flipping
    /// `v` would break exactly these. Computed against the pre-flip assignment `x`.
    fn break_count(&self, x: &[bool], v: usize) -> u32 {
        let mut breaks = 0;
        for &(c, lit_positive) in &self.var_clause_lits[v] {
            if x[v] == lit_positive && self.true_count[c as usize] == 1 {
                breaks += 1;
            }
        }
        breaks
    }

    /// Updates `true_count` and the unsatisfied-clause list after variable `v`
    /// has been flipped. `x` is the **post-flip** assignment. Returns
    /// `(newly satisfied, newly unsatisfied)` clause counts, whose difference is
    /// the change in the satisfied-clause objective (`O(degree)`).
    fn apply_flip(&mut self, x: &[bool], v: usize) -> (u32, u32) {
        let mut made = 0;
        let mut broke = 0;
        // Snapshot the occurrence list length to avoid holding a borrow of
        // `self.var_clause_lits` while mutating the other fields.
        for idx in 0..self.var_clause_lits[v].len() {
            let (c, lit_positive) = self.var_clause_lits[v][idx];
            let c = c as usize;
            if x[v] == lit_positive {
                // This occurrence became satisfying.
                self.true_count[c] += 1;
                if self.true_count[c] == 1 {
                    self.remove_from_unsat(c);
                    made += 1;
                }
            } else {
                // This occurrence stopped satisfying the clause.
                self.true_count[c] -= 1;
                if self.true_count[c] == 0 {
                    self.add_to_unsat(c);
                    broke += 1;
                }
            }
        }
        (made, broke)
    }

    fn add_to_unsat(&mut self, c: usize) {
        if self.pos_in_unsat[c] == NOT_IN_UNSAT {
            self.pos_in_unsat[c] = self.unsat_clauses.len();
            self.unsat_clauses.push(c);
        }
    }

    fn remove_from_unsat(&mut self, c: usize) {
        let pos = self.pos_in_unsat[c];
        if pos == NOT_IN_UNSAT {
            return;
        }
        let last = self.unsat_clauses.len() - 1;
        let moved = self.unsat_clauses[last];
        self.unsat_clauses.swap_remove(pos);
        self.pos_in_unsat[moved] = pos;
        self.pos_in_unsat[c] = NOT_IN_UNSAT;
    }
}

/// WalkSAT/SKC stochastic local search for [`Sat`] (MaxSAT).
///
/// See the [module docs](self) for the algorithm. Construct with [`new`](Self::new);
/// wire it into the benchmark via `kind = "WalkSat"` (SAT only).
///
/// # References
///
/// - Selman, B., Kautz, H. A., and Cohen, B. "Noise Strategies for Improving
///   Local Search." *Proc. AAAI-94*, 337-343, 1994.
/// - Hoos, H. H. "An Adaptive Noise Mechanism for WalkSAT." *Proc. AAAI-02*,
///   655-660, 2002.
pub struct WalkSatForSat {
    stop_condition: StopCondition,
    /// The configured noise probability (the working noise resets to this).
    base_noise: f64,
    /// Whether to adapt the noise to search progress (Hoos 2002).
    adaptive: bool,
    /// The working noise probability (equals `base_noise` when `adaptive` is off).
    noise: f64,
    scratch: Option<WalkSatScratch>,
    /// Fewest unsatisfied clauses seen this run (for adaptive noise).
    best_unsat: usize,
    /// Flips since `best_unsat` last improved (for adaptive noise).
    flips_since_improve: u64,
}

impl WalkSatForSat {
    /// Creates a WalkSAT heuristic.
    ///
    /// `noise` is the probability of a random (non-greedy) walk step within the
    /// chosen clause. `adaptive` enables Hoos' automatic noise adjustment, in
    /// which case `noise` is only the starting value.
    ///
    /// # Panics
    ///
    /// Panics if `noise` is not within `[0.0, 1.0]`.
    pub fn new(stop_condition: StopCondition, noise: f64, adaptive: bool) -> Self {
        assert!(
            (0.0..=1.0).contains(&noise),
            "noise must be within [0.0, 1.0], got {noise}"
        );
        Self {
            stop_condition,
            base_noise: noise,
            adaptive,
            noise,
            scratch: None,
            best_unsat: usize::MAX,
            flips_since_improve: 0,
        }
    }

    /// Rebuilds the scratch state from the current solution if it is missing or
    /// stale (e.g. the heuristic is reused on a different instance).
    fn ensure_scratch(&mut self, state: &SearchState<'_, Sat>) {
        let n_clauses = state.instance.n_clauses();
        let stale = self
            .scratch
            .as_ref()
            .is_none_or(|s| s.true_count.len() != n_clauses);
        if stale {
            let scratch = WalkSatScratch::build(state.instance, &state.solution.x);
            self.best_unsat = scratch.unsat_clauses.len();
            self.flips_since_improve = 0;
            self.noise = self.base_noise;
            self.scratch = Some(scratch);
        }
    }

    /// Adjusts the working noise based on whether the unsatisfied count improved.
    fn update_adaptive_noise(&mut self, n_clauses: usize) {
        let cur = self.scratch.as_ref().unwrap().unsat_clauses.len();
        if cur < self.best_unsat {
            self.best_unsat = cur;
            self.flips_since_improve = 0;
            // Gently lower the noise when making progress.
            self.noise -= self.noise * ADAPT_PHI / 2.0;
        } else {
            self.flips_since_improve += 1;
            let stagnation = (n_clauses as f64 * ADAPT_THETA) as u64;
            if self.flips_since_improve > stagnation {
                // Raise the noise to escape a stagnating region.
                self.noise += (1.0 - self.noise) * ADAPT_PHI;
                self.flips_since_improve = 0;
            }
        }
        self.noise = self.noise.clamp(0.0, 1.0);
    }

    /// Test-only: whether the scratch's unsatisfied set agrees with a fresh
    /// recomputation from the current assignment.
    #[cfg(test)]
    fn scratch_is_consistent(&self, instance: &Sat, x: &[bool]) -> bool {
        let scratch = match self.scratch.as_ref() {
            Some(s) => s,
            None => return false,
        };
        let mut expected: Vec<usize> = (0..instance.n_clauses())
            .filter(|&c| !Sat::is_clause_satisfied(instance.clause(c), x))
            .collect();
        let mut got = scratch.unsat_clauses.clone();
        expected.sort_unstable();
        got.sort_unstable();
        expected == got
    }
}

impl Heuristic<Sat> for WalkSatForSat {
    fn clear(&mut self) {
        self.scratch = None;
        self.noise = self.base_noise;
        self.best_unsat = usize::MAX;
        self.flips_since_improve = 0;
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    /// Stops on the configured [`StopCondition`], or early once every clause is
    /// satisfied (a global optimum for MaxSAT — no further improvement possible).
    fn is_done<'a>(&self, state: &SearchState<'a, Sat>) -> bool {
        if self.stop_condition.is_done(state) {
            return true;
        }
        // Only trust the empty check once the scratch has been built.
        self.scratch
            .as_ref()
            .is_some_and(|s| s.unsat_clauses.is_empty())
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, Sat>) -> Result<(), OptError> {
        self.ensure_scratch(state);
        let scratch = self.scratch.as_ref().unwrap();
        if scratch.unsat_clauses.is_empty() {
            // Fully satisfied; `is_done` will terminate the loop.
            return Ok(());
        }

        // 1. Sample a random unsatisfied clause. (`self.scratch` borrows `self`,
        //    `state.rng` borrows `state` — disjoint objects.)
        let clause_idx = state.rng.random_range(0..scratch.unsat_clauses.len());
        let clause = scratch.unsat_clauses[clause_idx];
        let clause_vars: Vec<usize> = state
            .instance
            .clause(clause)
            .iter()
            .map(|&lit| lit.unsigned_abs() as usize - 1)
            .collect();

        // 2. Find the variable with the smallest break count (pre-flip).
        let mut best_var = clause_vars[0];
        let mut best_break = u32::MAX;
        {
            let x = &state.solution.x;
            for &v in &clause_vars {
                let breaks = scratch.break_count(x, v);
                if breaks < best_break {
                    best_break = breaks;
                    best_var = v;
                }
            }
        }

        // 3. SKC rule: take a free (break-0) move if one exists; otherwise flip a
        //    random clause variable with probability `noise`, else the greedy one.
        let chosen = if best_break == 0 {
            best_var
        } else if state.rng.random::<f64>() < self.noise {
            let k = state.rng.random_range(0..clause_vars.len());
            clause_vars[k]
        } else {
            best_var
        };

        // 4. Commit the flip. We deliberately do *not* go through
        //    `SatFlipNeighbor::apply`, which recomputes the cached `gain[]` over
        //    every neighbor variable (`O(degree^2)` and the dominant cost on
        //    dense instances). WalkSAT never reads `gain[]` — it selects from
        //    `true_count` — so we update only `x`, `n_satisfied`, and our scratch
        //    (`O(degree)`), and restore a valid `gain[]` once at run end (see
        //    [`recompute_gains`]). This is the whole speed advantage of the move.
        state.solution.x[chosen] = !state.solution.x[chosen];
        let (made, broke) = self
            .scratch
            .as_mut()
            .unwrap()
            .apply_flip(&state.solution.x, chosen);
        let delta = made as i64 - broke as i64;
        state.solution.n_satisfied = (state.solution.n_satisfied as i64 + delta) as usize;
        state.iteration += 1;
        state.n_accepted += 1;
        state.update_best();

        if self.adaptive {
            self.update_adaptive_noise(state.instance.n_clauses());
        }

        Ok(())
    }

    fn run<'a>(&mut self, state: &mut SearchState<'a, Sat>) -> Result<(), OptError> {
        self.clear();
        while !self.is_done(state) {
            self.run_once(state)?;
        }
        // The walk left `gain[]` stale (see `run_once`); rebuild it on both the
        // current and best solutions so a following heuristic (ILS/restart
        // composition) sees a fully consistent `SatSolution`.
        recompute_gains(state.instance, &mut state.solution);
        recompute_gains(state.instance, &mut state.best_solution);
        Ok(())
    }
}

/// Recomputes the cached per-variable gain array of `sol` from scratch.
fn recompute_gains(instance: &Sat, sol: &mut crate::problem::SatSolution) {
    for i in 0..instance.n_vars() {
        sol.gain[i] = instance.calc_gain(&sol.x, i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::Sat;
    use crate::search_state::SearchState;
    use std::time::Duration;

    /// A small satisfiable formula over 3 variables.
    fn satisfiable_instance() -> Sat {
        let mut sat = Sat::new(3);
        sat.add_clause([1, 2, 3]);
        sat.add_clause([-1, 2]);
        sat.add_clause([1, -2, 3]);
        sat.add_clause([-3, 1]);
        sat
    }

    #[test]
    fn reaches_full_satisfaction_on_satisfiable_instance() {
        let sat = satisfiable_instance();
        let mut state = SearchState::new_with_seed(&sat, 7);
        let mut walksat = WalkSatForSat::new(StopCondition::iterations(10_000), 0.3, false);
        walksat.run(&mut state).unwrap();
        assert_eq!(
            state.best_solution.n_satisfied,
            sat.n_clauses(),
            "WalkSAT should satisfy every clause of a satisfiable instance"
        );
    }

    #[test]
    fn same_seed_is_bit_reproducible() {
        let sat = satisfiable_instance();
        let run = |seed: u64| {
            let mut state = SearchState::new_with_seed(&sat, seed);
            let mut w = WalkSatForSat::new(
                StopCondition::iterations(500).with_duration(Duration::from_secs(5)),
                0.4,
                true,
            );
            w.run(&mut state).unwrap();
            (
                state.best_solution.x.clone(),
                state.best_solution.n_satisfied,
            )
        };
        assert_eq!(run(42), run(42), "same seed must give identical results");
    }

    #[test]
    fn gain_array_is_valid_after_run() {
        // The walk leaves gain[] stale; run() must restore it for composition.
        let sat = satisfiable_instance();
        let mut state = SearchState::new_with_seed(&sat, 9);
        let mut walksat = WalkSatForSat::new(StopCondition::iterations(200), 0.3, false);
        walksat.run(&mut state).unwrap();
        for (sol, label) in [
            (&state.solution, "solution"),
            (&state.best_solution, "best_solution"),
        ] {
            for i in 0..sat.n_vars() {
                assert_eq!(
                    sol.gain[i],
                    sat.calc_gain(&sol.x, i),
                    "{label}.gain[{i}] must be consistent after run()"
                );
            }
        }
    }

    #[test]
    fn scratch_stays_consistent_across_flips() {
        let sat = satisfiable_instance();
        let mut state = SearchState::new_with_seed(&sat, 123);
        // Stop condition never fires within the manual loop below.
        let mut walksat = WalkSatForSat::new(StopCondition::iterations(u64::MAX), 0.5, false);
        for _ in 0..50 {
            walksat.run_once(&mut state).unwrap();
            assert!(
                walksat.scratch_is_consistent(&sat, &state.solution.x),
                "scratch unsatisfied set must match a fresh recomputation"
            );
            // The maintained satisfied count must match the ground truth.
            assert_eq!(
                state.solution.n_satisfied,
                sat.calc_satisfied(&state.solution.x)
            );
        }
    }
}
