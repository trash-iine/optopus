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

    fn iter(prob: &OneMaxProblem, _sol: &OneMaxSolution) -> impl Iterator<Item = Self> + Send {
        (0..prob.n).map(|i| FlipMove { index: i })
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
    fn is_better_than(&self, _other: &Self) -> bool {
        false // move-vs-move comparison is unused (LocalSearch ranks by resulting solutions)
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
