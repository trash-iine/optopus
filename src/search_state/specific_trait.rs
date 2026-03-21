/// Is for comparing two solutions by quality.
///
/// This trait is used by almost heuristic algorithms to determine which solution is better.
///
/// [`Rankable::is_better_than`] returns `true` if `self` is strictly better than `other`.
pub trait Rankable {
    fn is_better_than(&self, other: &Self) -> bool;
}

/// Returns all elements that are tied for the best rank among the items yielded by `iter`.
///
/// If `iter` is empty, returns an empty `Vec`.
pub fn filter_best<R: Rankable, T: Iterator<Item = R>>(iter: T) -> Vec<R> {
    let mut best_list = vec![];
    for r in iter {
        if best_list.is_empty() {
            best_list = vec![r];
        } else {
            let sample = &best_list[0];
            if r.is_better_than(sample) {
                best_list = vec![r];
            } else if !sample.is_better_than(&r) {
                best_list.push(r);
            }
        }
    }

    return best_list;
}

/// Is a combinatorial optimization problem.
///
/// Implement this trait to define a custom problem. You must specify the associated
/// [`ProblemTrait::Solution`] type and provide a method to generate a random initial solution.
///
/// # Example
///
/// ```rust,ignore
/// struct MyProblem { /* ... */ }
/// struct MySolution { value: f64 }
///
/// impl Rankable for MySolution {
///     fn is_better_than(&self, other: &Self) -> bool { self.value > other.value }
/// }
/// impl ProblemTrait for MyProblem {
///     type Solution = MySolution;
///     fn new_solution(&self, _rng: &mut impl rand::Rng) -> MySolution {
///         MySolution { value: 0.0 }
///     }
/// }
/// ```
pub trait ProblemTrait {
    type Solution: Clone + Rankable;
    fn new_solution(&self, rng: &mut impl rand::Rng) -> Self::Solution;
}

/// Is a single neighborhood move (one step of change).
///
/// This trait is used by almost heuristic algorithms to define the structure of moves,
/// how they affect the solution and iteration count.
///
/// [`MoveToNeighbor::iter`] enumerates all moves reachable from the current solution, and
/// [`MoveToNeighbor::apply_to_solution`] applies a move to the solution in place.
pub trait MoveToNeighbor<Problem>
where
    Problem: ProblemTrait,
    Problem::Solution: Rankable,
{
    /// Returns the new iteration count after applying this move (default: `iter + 1`).
    fn apply_to_iteration(&self, iter: u64) -> u64 {
        iter + 1
    }

    /// Applies this move to `sol` in place.
    fn apply_to_solution(
        &self,
        prob: &Problem,
        sol: &mut Problem::Solution,
    ) -> Result<(), crate::error::OptError>;

    /// Returns an iterator over all moves reachable from the given solution.
    fn iter(prob: &Problem, sol: &Problem::Solution) -> impl Iterator<Item = Self> + Send;

    /// Returns `true` if applying this move to `src` yields a solution better than `other`.
    ///
    /// <div class="warning">
    /// The default implementation clones the solution and applies the move to it, which may be inefficient.
    /// Override this method with a more efficient implementation if possible.
    /// </div>
    fn move_to_be_better_than(
        &self,
        prob: &Problem,
        src: &Problem::Solution,
        other: &Problem::Solution,
    ) -> bool {
        let mut cloned = src.clone();
        self.apply_to_solution(prob, &mut cloned)
            .expect("apply_to_solution should not fail");
        cloned.is_better_than(other)
    }
}

/// Is for evaluating a move or solution as a scalar value.
///
/// Used by [`crate::heuristic::SimulatedAnnealing`] to compute acceptance probabilities.
pub trait Evaluable<T> {
    fn evaluate(&self) -> T;
}

/// Is for moves that support a tabu list mechanism.
///
/// A move is considered *enabled* if it is not currently forbidden by the tabu map.
/// After a move is applied, it can be added to the tabu map with a given tenure.
pub trait EnabledTabu {
    /// The data structure used to store the tabu list (e.g., `HashMap<usize, u64>`).
    type TabuMap;

    /// Returns `true` if this move is allowed under the current tabu map at the given iteration.
    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool;

    /// Adds this move to the tabu map with a randomly sampled tenure in the given range.
    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
    );
}
