//! Trait for problems over binary variables.

use super::{MoveToNeighbor, ProblemTrait};

/// A problem whose solutions assign a boolean value to each variable and cache
/// a per-variable flip gain.
///
/// MaxCut, QUBO, MaxSAT and Vertex Cover all fit this shape. Implementing this trait gives a problem
/// access to the generic binary-variable machinery in [`crate::common`], such as
/// [`uniform_binary_crossover`](crate::common::uniform_binary_crossover).
pub trait BinaryProblem: ProblemTrait + Sized {
    /// The single-variable flip move for this problem.
    type Flip: MoveToNeighbor<Self>;

    /// Returns an iterator over the indices of the problem's binary variables.
    ///
    /// `Send` so that the pair helpers in [`crate::common`] can build the
    /// `Send` neighborhoods [`MoveToNeighbor::iter`] requires.
    fn variable_indices(&self) -> impl Iterator<Item = usize> + Send + '_;

    /// Returns the value of variable `i` in `sol`.
    fn variable(sol: &Self::Solution, i: usize) -> bool;

    /// Returns the flip move for variable `i`, carrying the gain cached in `sol`.
    fn flip_move(sol: &Self::Solution, i: usize) -> Self::Flip;

    /// The solution assigning `values[i]` to each variable `i`, `values`
    /// covering every variable index.
    ///
    /// The default starts from a solution drawn with a fixed seed and flips
    /// each variable that differs, which keeps whatever the solution caches up
    /// to date. The built-in problems build the solution directly instead.
    fn solution_from_assignment(&self, values: &[bool]) -> Self::Solution {
        use rand::SeedableRng;
        let mut sol = self.new_solution(&mut rand::rngs::SmallRng::seed_from_u64(0));
        for i in self.variable_indices() {
            if Self::variable(&sol, i) != values[i] {
                Self::flip_move(&sol, i)
                    .apply_to_solution(self, &mut sol)
                    .expect("flip on a valid variable index cannot fail");
            }
        }
        sol
    }
}

/// A binary problem that can fix some of its variables and fold them into a
/// smaller instance of itself, which is what lets
/// [`BranchAndBound`](crate::heuristic::BranchAndBound) search a node with any
/// heuristic of the problem.
///
/// Moving between the node and the whole problem, and the bound a node is
/// pruned by, are written once over this trait in
/// [`crate::problem::branch`]. What a problem supplies is how fixing folds
/// into its instance, and a bound on a whole instance.
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
