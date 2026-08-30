//! Example of defining your own optimization problem.
//!
//! Implements ProblemTrait / MoveToNeighbor / Rankable and solves the
//! problem with the built-in LocalSearch.
//!
//! The problem here is deliberately simple: maximize the number of `true`
//! bits in a binary vector (OneMax).
//!
//! Run with:
//! ```
//! cargo run --example custom_problem
//! ```

use optopus::prelude::*;

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

impl Rankable for OneMaxSolution {
    fn is_better_than(&self, other: &Self) -> bool {
        self.objective() > other.objective()
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

impl Rankable for FlipMove {
    /// Ranks candidate moves against each other. `LocalSearch` selects with
    /// `max_by(rank_cmp)` over this, so comparing the cached gains is what makes
    /// it *best*-improving; a constant `false` would compile but leave every
    /// candidate tied, degrading the selection to an arbitrary improving move.
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain > other.gain
    }
}

// ─── Main ───────────────────────────────────────────────────
fn main() {
    let prob = OneMaxProblem { n: 20 };
    let mut state = SearchState::new(&prob);

    let mut ls = LocalSearch::<FlipMove>::new(StopCondition::iterations(10_000));
    ls.run(&mut state).unwrap();

    println!(
        "best = {:?}  (objective = {}/{})",
        state.best_solution.bits,
        state.best_solution.objective(),
        prob.n
    );
}
