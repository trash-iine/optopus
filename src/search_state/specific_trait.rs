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
    const RESERVE_CAPACITY: usize = 16;
    let mut best_list: Vec<R> = Vec::with_capacity(RESERVE_CAPACITY);
    for r in iter {
        if best_list.is_empty() {
            best_list.push(r);
        } else {
            let sample = &best_list[0];
            if r.is_better_than(sample) {
                best_list.clear();
                best_list.push(r);
            } else if !sample.is_better_than(&r) {
                best_list.push(r);
            }
        }
    }

    best_list
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

/// The change in objective value resulting from a move, with explicit optimization direction.
///
/// `T` is the numeric type of the change (commonly `f64`, but any type is accepted).
///
/// Choose the variant that matches your problem's optimization direction:
/// - [`Evaluable::Maximize`]: the objective is being maximized (positive = improvement).
/// - [`Evaluable::Minimize`]: the cost is being minimized (positive = worsening).
#[derive(Clone, Copy, Debug)]
pub enum Evaluable<T = f64> {
    /// Change in a maximized objective (positive = improvement, negative = worsening).
    Maximize(T),
    /// Change in a minimized cost (positive = worsening, negative = improvement).
    Minimize(T),
}

impl Evaluable<f64> {
    /// Returns the worsening amount: positive when the move degrades the objective.
    ///
    /// Used internally by `boltzmann_accept` to compute `exp(-worsening / T)`.
    pub fn worsening_amount(self) -> f64 {
        match self {
            Evaluable::Maximize(gain) => -gain,
            Evaluable::Minimize(cost_delta) => cost_delta,
        }
    }
}

/// Implemented by neighbor types that can evaluate their objective change for SA acceptance.
///
/// `T` is the numeric type returned (default `f64`). Use `T = f64` for compatibility
/// with [`crate::heuristic::SimulatedAnnealing`] and [`crate::heuristic::boltzmann_accept`].
pub trait Evaluate<T = f64> {
    fn evaluate(&self) -> Evaluable<T>;
}

/// Hamming-style distance between two solutions.
///
/// Used by parent-selection strategies that promote population diversity
/// (e.g. [`crate::heuristic::ParentSelection::HammingTopK`]).
///
/// For bit-vector solutions this is the standard Hamming distance — the
/// number of variables that differ. For other encodings any application-
/// meaningful integer dissimilarity measure works.
pub trait Distance {
    fn distance(&self, other: &Self) -> usize;
}

/// Combines parent solutions into a single offspring solution.
///
/// Implement this for each crossover operator you want to use.
///
/// # `&mut self`
///
/// `&mut self` is required so that operators like [`crate::heuristic::SubProblemBasedCrossover`]
/// can call an inner heuristic's `run` method during crossover.
/// Stateless operators (e.g. uniform crossover) simply do not mutate `self`.
pub trait Crossover<Problem: ProblemTrait> {
    /// Combines two parent solutions into a single offspring.
    ///
    /// Returns `Err` only when the operator genuinely cannot produce an
    /// offspring (e.g. an inner sub-heuristic failed). Stateless operators
    /// such as the per-problem uniform crossovers never fail and simply
    /// return `Ok(...)`.
    fn crossover(
        &mut self,
        prob: &Problem,
        sol1: &Problem::Solution,
        sol2: &Problem::Solution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Result<Problem::Solution, crate::error::OptError>;
}

/// Forwards [`Crossover`] through a boxed trait object so `GeneticAlgorithm`
/// can use a runtime-selected operator (needed by the benchmark TOML config).
impl<P: ProblemTrait> Crossover<P> for Box<dyn Crossover<P>> {
    fn crossover(
        &mut self,
        prob: &P,
        sol1: &P::Solution,
        sol2: &P::Solution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Result<P::Solution, crate::error::OptError> {
        (**self).crossover(prob, sol1, sol2, rng)
    }
}

/// Enables a problem to create a sub-problem from multiple parent solutions and lift the result back.
///
/// Used by [`crate::heuristic::SubProblemBasedCrossover`] to implement crossover by:
/// 1. Finding variables where the parents disagree (free variables).
/// 2. Building a reduced sub-problem containing only those variables.
///    Contributions from fixed (agreeing) variables are folded into the sub-problem objective.
/// 3. Solving the sub-problem with any [`crate::heuristic::Heuristic`].
/// 4. Lifting the sub-solution back to the full solution space.
///
/// # Example (MaxCut)
///
/// - Vertices with the same side in both parents → fixed; their edges become bias terms.
/// - Vertices with different sides → free; form the sub-MaxCut instance.
/// - `lift_solution` merges fixed assignments (from `sol1`) with the sub-problem result.
///
/// # Sized bound
///
/// Required because `extract_sub_problem` returns `Self`.
pub trait SubProblemExtractable: ProblemTrait + Sized {
    /// Creates a sub-problem containing only the variables that differ between the two parents.
    ///
    /// Fixed variables' contributions (edges to free variables) must be incorporated
    /// into the sub-problem objective so the sub-problem remains self-contained.
    fn extract_sub_problem(&self, sol1: &Self::Solution, sol2: &Self::Solution) -> Self;

    /// Reconstructs a full solution from a sub-problem solution.
    ///
    /// - Variables that agreed in both parents: value taken from `sol1`.
    /// - Variables that differed (free variables): value taken from `sub_solution`.
    fn lift_solution(
        &self,
        sol1: &Self::Solution,
        sol2: &Self::Solution,
        sub_solution: &Self::Solution,
    ) -> Self::Solution;
}

/// Is for moves that support a tabu list mechanism.
///
/// A move is considered *enabled* if it is not currently forbidden by the tabu map.
/// After a move is applied, it can be added to the tabu map with a given tenure.
pub trait EnabledTabu: Clone {
    /// The data structure used to store the tabu list (e.g., `HashMap<usize, u64>`).
    type TabuMap: Default;

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
