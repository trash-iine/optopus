use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::problem::integer::{Branchable, IntVar, IntVars, Relaxation};
use crate::search_state::{Evaluable, Evaluate, SearchState};

/// Branch-and-bound over integer variables, with any heuristic finding the
/// solutions and a [`Relaxation`] bounding what is left.
///
/// A node is the problem with some ranges narrowed. Each iteration takes the
/// open node whose bound is best, and drops it when that bound cannot beat
/// the best solution found so far. Otherwise the inner heuristic searches the
/// node, starting from the best solution moved into its ranges, and the node
/// splits one variable's range in two at that solution's value. A node whose
/// variables all hold a single value is evaluated directly.
///
/// The inner heuristic only decides how soon good solutions turn up. Whatever
/// it is, a run that empties the open nodes has proven its best solution
/// optimal, as far as the relaxation is a bound, and
/// [`is_proven_optimal`](Self::is_proven_optimal) says so. A run stopped
/// earlier reports the best bound still open through
/// [`dual_bound`](Self::dual_bound).
///
/// Each node counts as one iteration, on top of the inner heuristic's own,
/// and the inner stop condition is a budget per node. Improvements are
/// recorded when a node's search returns, not while it runs.
///
/// # Example
///
/// ```
/// use optopus::prelude::*;
///
/// // 0-1 knapsack, capacity 10, the overweight penalized
/// let (value, weight) = ([6.0, 5.0, 4.0, 3.0], [5.0, 4.0, 3.0, 2.0]);
/// let vars: IntVars = (0..4).map(|_| IntVar::binary()).collect();
/// let prob = IntegerProblem::maximize(vars, |x: &[i64]| {
///     let v: f64 = x.iter().zip(&value).map(|(&xi, v)| xi as f64 * v).sum();
///     let w: f64 = x.iter().zip(&weight).map(|(&xi, w)| xi as f64 * w).sum();
///     v - 10.0 * (w - 10.0).max(0.0)
/// });
///
/// // Every item taken where its range allows it and the capacity ignored,
/// // which no assignment in the ranges beats.
/// let relaxation = |_: &_, vars: &IntVars| {
///     Evaluable::Maximize(vars.iter().zip(&value).map(|(r, v)| r.upper() as f64 * v).sum())
/// };
///
/// let mut bnb = BranchAndBound::new(
///     StopCondition::new(None, None, None),
///     Box::new(LocalSearch::<IntChangeNeighbor>::new(StopCondition::iterations(100))),
///     relaxation,
/// );
/// let mut state = SearchState::new_with_seed(&prob, 1);
/// bnb.run(&mut state).unwrap();
/// assert_eq!(state.best_solution.evaluate().minimized(), -13.0);
/// assert!(bnb.is_proven_optimal(&state.best_solution));
/// ```
///
/// # References
///
/// - Land, A. H. and Doig, A. G. "An automatic method of solving discrete
///   programming problems". *Econometrica* 28(3), 497-520, 1960.
pub struct BranchAndBound<P, R> {
    pub stop_condition: StopCondition,
    /// The heuristic searching each node.
    pub heuristic: Box<dyn Heuristic<P>>,
    /// The bound a node is pruned by.
    pub relaxation: R,
    open: BinaryHeap<Node>,
    /// Whether the whole problem is still to be opened, which the first
    /// iteration of a run does since only the state knows the problem.
    root_pending: bool,
}

/// An open node, the ranges narrowed from the whole problem's and the bound
/// they were given, lower being better.
struct Node {
    bound: f64,
    /// The narrowed ranges in the order they were narrowed, a later entry
    /// for a variable replacing an earlier one.
    narrowed: Vec<(usize, IntVar)>,
}

impl Node {
    fn ranges(&self, whole: &IntVars) -> IntVars {
        let mut vars: Vec<IntVar> = whole.to_vec();
        for &(i, range) in &self.narrowed {
            vars[i] = range;
        }
        IntVars::new(vars)
    }
}

/// The heap pops the best bound first, and among equal bounds the deepest
/// node, which reaches a single assignment sooner.
impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .bound
            .total_cmp(&self.bound)
            .then(self.narrowed.len().cmp(&other.narrowed.len()))
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Node {}

impl<P: Branchable, R: Relaxation<P>> BranchAndBound<P, R> {
    /// A search that stops when no open node can beat the best solution, or
    /// earlier at `stop_condition`, whose fields may all be `None`.
    pub fn new(
        stop_condition: StopCondition,
        heuristic: Box<dyn Heuristic<P>>,
        relaxation: R,
    ) -> Self {
        Self {
            stop_condition,
            heuristic,
            relaxation,
            open: BinaryHeap::new(),
            root_pending: true,
        }
    }

    /// The best objective an assignment not yet ruled out could reach, in the
    /// direction of `incumbent`. It equals the incumbent's objective once the
    /// search has proven it optimal, and before the first iteration it is
    /// unbounded.
    pub fn dual_bound(&self, incumbent: &P::Solution) -> Evaluable<f64> {
        let incumbent = incumbent.evaluate();
        let open = if self.root_pending {
            f64::NEG_INFINITY
        } else {
            self.open.peek().map_or(f64::INFINITY, |n| n.bound)
        };
        directed_like(incumbent, open.min(incumbent.minimized()))
    }

    /// Whether no assignment can beat `incumbent`, as far as the relaxation is
    /// a bound.
    pub fn is_proven_optimal(&self, incumbent: &P::Solution) -> bool {
        let best = incumbent.evaluate().minimized();
        !self.root_pending && self.open.peek().is_none_or(|n| n.bound >= best)
    }

    /// The bound of `vars`, lower being better, after checking the relaxation
    /// states it in the problem's direction.
    fn bound(
        &mut self,
        prob: &P,
        vars: &IntVars,
        incumbent: Evaluable<f64>,
    ) -> Result<f64, OptError> {
        let bound = self.relaxation.bound(prob, vars);
        if std::mem::discriminant(&bound) != std::mem::discriminant(&incumbent) {
            return Err(OptError::Config(format!(
                "BranchAndBound: the relaxation returned {bound:?} for a problem whose \
                 solutions report {incumbent:?}"
            )));
        }
        Ok(bound.minimized())
    }
}

/// `minimized` read back in the direction of `like`.
fn directed_like(like: Evaluable<f64>, minimized: f64) -> Evaluable<f64> {
    match like {
        Evaluable::Maximize(_) => Evaluable::Maximize(-minimized),
        Evaluable::Minimize(_) => Evaluable::Minimize(minimized),
    }
}

impl<P: Branchable, R: Relaxation<P>> Heuristic<P> for BranchAndBound<P, R> {
    fn clear(&mut self) {
        self.open.clear();
        self.root_pending = true;
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    /// Done when the stop condition is met, or when no node is left open.
    fn is_done<'a>(&self, state: &SearchState<'a, P>) -> bool {
        self.stop_condition.is_done(state) || (!self.root_pending && self.open.is_empty())
    }

    /// Takes the open node with the best bound, and prunes, evaluates or
    /// searches and splits it.
    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        let prob = state.instance;
        let whole = prob.domains();
        if self.root_pending {
            if whole.is_permutation() {
                return Err(OptError::Config(
                    "BranchAndBound narrows one variable at a time, which a permutation \
                     does not allow"
                        .into(),
                ));
            }
            let bound = self.bound(prob, whole, state.best_solution.evaluate())?;
            self.open.push(Node {
                bound,
                narrowed: vec![],
            });
            self.root_pending = false;
        }
        let Some(node) = self.open.pop() else {
            return Ok(());
        };
        state.progress_iteration();
        if node.bound >= state.best_solution.evaluate().minimized() {
            return Ok(());
        }

        let vars = node.ranges(whole);
        let Some(first_free) = vars.iter().position(|v| v.num_changes() > 0) else {
            let values = vars.iter().map(IntVar::lower).collect();
            state.solution = prob.solution_from_values(values);
            state.update_best();
            return Ok(());
        };

        let reduction = crate::problem::integer::DomainRestriction {
            target: prob.restricted(vars.clone()),
        };
        state.solution = state.best_solution.clone();
        let mut sub = state.open_reduction(&reduction);
        self.heuristic.run(&mut sub)?;
        state.close_reduction(&reduction, &sub);

        let var = self
            .relaxation
            .branch_hint(prob, &vars)
            .filter(|&i| vars.get(i).is_some_and(|v| v.num_changes() > 0))
            .unwrap_or(first_free);
        let range = vars[var];
        let split = P::get(&state.best_solution, var).clamp(range.lower(), range.upper() - 1);
        let incumbent = state.best_solution.evaluate();
        for half in [
            IntVar::new(range.lower(), split),
            IntVar::new(split + 1, range.upper()),
        ] {
            let mut narrowed = node.narrowed.clone();
            narrowed.push((var, half));
            let child = Node {
                bound: 0.0,
                narrowed,
            };
            let bound = self
                .bound(prob, &child.ranges(whole), incumbent)?
                .max(node.bound);
            if bound < incumbent.minimized() {
                self.open.push(Node { bound, ..child });
            }
        }
        Ok(())
    }
}
