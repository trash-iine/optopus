//! Example of defining your own optimization problem.
//!
//! Implements ProblemTrait / MoveToNeighbor / Evaluate and solves the
//! problem with the built-in LocalSearch, then adds EnabledTabu so that
//! TabuSearch runs on it too.
//!
//! The problem here is deliberately simple: maximize the number of `true`
//! bits in a binary vector (OneMax).
//!
//! Run with:
//! ```
//! cargo run --example custom_problem
//! ```

use optopus::building_blocks::search::{TabuKey, TabuMemory};
use optopus::prelude::*;
// optopus re-exports the rand it is built against. The trait signatures below
// name rand types, and this keeps them the same version without a dependency.
use optopus::rand;
use optopus::rand::rngs::SmallRng;

// ─── Problem definition ─────────────────────────────────────
/// Maximize the number of bits set to 1 among `n` binary variables (OneMax).
struct OneMaxProblem {
    n: usize,
}

// ─── Solution definition ────────────────────────────────────
#[derive(Clone)]
struct OneMaxSolution {
    bits: Vec<bool>,
}

impl OneMaxSolution {
    fn objective(&self) -> usize {
        self.bits.iter().filter(|&&b| b).count()
    }
}

impl Evaluate for OneMaxSolution {
    /// OneMax maximizes the number of set bits. Wrapping it in `Maximize` is
    /// the only place this problem states its direction, and it is also what
    /// gives the solution `Rankable`.
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Maximize(self.objective() as f64)
    }
}

impl ProblemTrait for OneMaxProblem {
    type Solution = OneMaxSolution;

    fn new_solution(&self, rng: &mut impl rand::Rng) -> OneMaxSolution {
        OneMaxSolution {
            bits: (0..self.n).map(|_| rng.random_bool(0.5)).collect(),
        }
    }
}

// ─── Neighborhood definition (single-bit flip) ──────────────
/// `Clone` is what `TabuSearch` asks of a move besides `EnabledTabu`.
#[derive(Clone)]
struct FlipMove {
    index: usize,
    /// Change in objective this flip would cause, cached at construction time.
    /// Every built-in move type carries one; it is what lets `Rankable` below
    /// compare two moves without touching a solution.
    gain: i32,
}

impl FlipMove {
    /// The only correct way to build a move: the cached `gain` must match the
    /// solution the move will be applied to, so it is computed here rather than
    /// filled in by the caller.
    fn new(_prob: &OneMaxProblem, sol: &OneMaxSolution, index: usize) -> Self {
        // Setting a 0 bit gains one satisfied variable; clearing a 1 bit loses one.
        let gain = if sol.bits[index] { -1 } else { 1 };
        FlipMove { index, gain }
    }
}

impl MoveToNeighbor<OneMaxProblem> for FlipMove {
    fn apply_to_solution(
        &self,
        _prob: &OneMaxProblem,
        sol: &mut OneMaxSolution,
    ) -> Result<(), optopus::error::OptError> {
        sol.bits[self.index] = !sol.bits[self.index];
        Ok(())
    }

    fn iter(prob: &OneMaxProblem, sol: &OneMaxSolution) -> impl Iterator<Item = Self> + Send {
        (0..prob.n).map(move |i| FlipMove::new(prob, sol, i))
    }

    /// Hands this move's tabu policy to the search state. Without this line
    /// the `EnabledTabu` impl below still compiles, and `TabuSearch` runs with
    /// no tabu list at all.
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

    fn move_to_be_better_than(
        &self,
        prob: &OneMaxProblem,
        src: &OneMaxSolution,
        other: &OneMaxSolution,
    ) -> bool {
        let mut cloned = src.clone();
        self.apply_to_solution(prob, &mut cloned)
            .expect("apply_to_solution should not fail");
        cloned.is_better_than(other)
    }
}

impl Evaluate for FlipMove {
    /// The change this move would make. `LocalSearch` selects with
    /// `max_by(rank_cmp)`, which reads this through the derived `Rankable`, so
    /// reporting the cached gain here is what makes the search
    /// *best*-improving; a constant would compile but leave every candidate
    /// tied, degrading the selection to an arbitrary improving move.
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Maximize(f64::from(self.gain))
    }
}

// ─── Tabu policy ────────────────────────────────────────────
impl EnabledTabu for FlipMove {
    /// The flip is allowed once bit `index` is no longer forbidden.
    fn is_move_enabled(&self, tabu: &TabuMemory, iteration: u64) -> bool {
        tabu.is_enabled(TabuKey::DenseVar(self.index), iteration)
    }

    /// Applying the flip forbids bit `index` for a tenure the memory draws
    /// from the range given to `TabuSearch::new`.
    fn add_to_tabu_map(&self, tabu: &mut TabuMemory, iteration: u64, rng: &mut SmallRng) {
        tabu.forbid(TabuKey::DenseVar(self.index), iteration, rng);
    }
}

// ─── Main ───────────────────────────────────────────────────
fn main() {
    let prob = OneMaxProblem { n: 20 };
    let mut state = SearchState::new(&prob);

    let mut ls = LocalSearch::<FlipMove>::new(StopCondition::iterations(10_000));
    ls.run(&mut state).unwrap();

    println!(
        "[LocalSearch] best = {:?}  (objective = {}/{})",
        state.best_solution.bits,
        state.best_solution.objective(),
        prob.n
    );

    let mut state = SearchState::new(&prob);
    let mut ts = TabuSearch::<FlipMove>::new(StopCondition::iterations(10_000), (2, 5));
    ts.run(&mut state).unwrap();

    println!(
        "[TabuSearch]  best = {:?}  (objective = {}/{})",
        state.best_solution.bits,
        state.best_solution.objective(),
        prob.n
    );
}
