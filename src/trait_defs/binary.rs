//! Trait for problems over binary variables.

use rand::Rng;

use super::{MoveToNeighbor, ProblemTrait};
use crate::error::OptError;

/// A problem whose solutions assign a boolean value to each variable and cache
/// a per-variable flip gain.
///
/// MaxCut, QUBO, MaxSAT and Vertex Cover all fit this shape. Implementing this
/// trait gives a problem access to the generic binary-variable machinery in
/// [`representation`](crate::building_blocks::representation), such as
/// [`uniform_binary_crossover`](crate::building_blocks::representation::uniform_binary_crossover).
pub trait BinaryProblem: ProblemTrait + Sized {
    /// The single-variable flip move for this problem.
    type Flip: MoveToNeighbor<Self>;

    /// Returns an iterator over the indices of the problem's binary variables.
    ///
    /// `Send` so that the pair helpers in
    /// [`representation`](crate::building_blocks::representation) can build
    /// the `Send` neighborhoods [`MoveToNeighbor::iter`] requires.
    fn variable_indices(&self) -> impl Iterator<Item = usize> + Send + '_;

    /// One of [`variable_indices`](Self::variable_indices) drawn uniformly,
    /// or `None` when there are none.
    ///
    /// The default walks the indices, which is O(n). A problem that keeps
    /// them in a slice should answer in O(1), since the random swap moves
    /// draw through this.
    fn random_variable(&self, rng: &mut impl Rng) -> Option<usize> {
        let n = self.variable_indices().count();
        if n == 0 {
            return None;
        }
        self.variable_indices().nth(rng.random_range(0..n))
    }

    /// Returns the value of variable `i` in `sol`.
    fn variable(sol: &Self::Solution, i: usize) -> bool;

    /// Returns the flip move for variable `i`, carrying the gain cached in `sol`.
    fn flip_move(sol: &Self::Solution, i: usize) -> Self::Flip;

    /// The solution assigning `values[i]` to each variable `i`.
    ///
    /// Fails unless `values` covers every variable index and runs no further,
    /// the check [`check_assignment`] makes. A problem whose solutions span
    /// more than its variables, a graph with isolated vertices for one, may
    /// accept values up to that length instead.
    ///
    /// The default starts from a solution drawn with a fixed seed and flips
    /// each variable that differs, which keeps whatever the solution caches up
    /// to date. The built-in problems build the solution directly instead.
    fn solution_from_assignment(&self, values: &[bool]) -> Result<Self::Solution, OptError> {
        check_assignment(self, values, variable_count(self))?;
        use rand::SeedableRng;
        let mut sol = self.new_solution(&mut rand::rngs::SmallRng::seed_from_u64(0));
        for i in self.variable_indices() {
            if Self::variable(&sol, i) != values[i] {
                Self::flip_move(&sol, i)
                    .apply_to_solution(self, &mut sol)
                    .expect("flip on a valid variable index cannot fail");
            }
        }
        Ok(sol)
    }
}

/// One past the largest variable index, the length an assignment covering
/// every variable needs.
fn variable_count<P: BinaryProblem>(prob: &P) -> usize {
    prob.variable_indices().max().map_or(0, |i| i + 1)
}

/// Checks that `values` gives every variable of `prob` a value and holds no
/// more than `longest` values.
pub fn check_assignment<P: BinaryProblem>(
    prob: &P,
    values: &[bool],
    longest: usize,
) -> Result<(), OptError> {
    let needed = variable_count(prob);
    if values.len() < needed {
        return Err(OptError::InvalidState(format!(
            "{} values given, the variables need {needed}",
            values.len()
        )));
    }
    if values.len() > longest {
        return Err(OptError::InvalidState(format!(
            "{} values given, more than the {longest} a solution holds",
            values.len()
        )));
    }
    Ok(())
}

/// A binary problem that can fix some of its variables and fold them into a
/// smaller instance of itself, which is what lets
/// [`BranchAndBound`](crate::heuristic::BranchAndBound) search a node with any
/// heuristic of the problem.
///
/// Moving between the node and the whole problem, and the bound a node is
/// pruned by, are written once over this trait in
/// [`representation`](crate::building_blocks::representation). What a problem
/// supplies is how fixing folds into its instance, and a bound on a whole
/// instance.
pub trait FixVariables: BinaryProblem<Solution: super::Evaluate> {
    /// The instance over the variables `fixed` leaves free, with what the
    /// fixed ones contribute folded in. `fixed` has one entry per variable
    /// index, `Some(value)` where the variable is held at `value`.
    ///
    /// The folding may fix more variables than it was asked to, when a value
    /// is known to be no worse, and says so in
    /// [`placed`](FixedVariables::placed).
    fn fix(&self, fixed: &[Option<bool>]) -> FixedVariables<Self>;

    /// A value no solution of this instance beats, with the direction of the
    /// problem. Cheap and weak is fine. Each node asks it of its folded
    /// instance.
    fn trivial_bound(&self) -> super::Evaluable<f64>;
}

/// Where a variable of the whole problem went when variables were fixed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placed {
    /// Held at this value.
    Fixed(bool),
    /// Still free, as this variable of the folded instance.
    Free(usize),
}

/// The folded instance [`FixVariables::fix`] returns, and how to read a
/// solution of it as one of the whole problem.
///
/// The folding must not lose the optimum. The best objective of the
/// assignments it covers is at most as good as `offset` plus the folded
/// instance's best, and lifting that best attains it. The built-in problems
/// other than vertex cover keep `offset` plus the folded objective equal to
/// the whole objective for every solution, not only the best.
pub struct FixedVariables<P> {
    /// The folded instance.
    pub target: P,
    /// For each variable of the whole problem, its value or where it went.
    pub placed: Vec<Placed>,
    /// What the fixed variables contribute, added to the folded instance's
    /// raw objective.
    pub offset: f64,
    /// A variable of the folded instance every free variable is read relative
    /// to, for a problem whose objective does not change when every variable
    /// flips. A free variable's value is its value in the folded solution
    /// differing from this one's.
    pub reference: Option<usize>,
}
