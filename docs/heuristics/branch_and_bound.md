# BranchAndBound

**API:** [`BranchAndBound`](../api/optopus/heuristic/struct.BranchAndBound.html)

Branch-and-bound over [integer variables](../problems/integer.md). Any
heuristic finds the solutions, and a relaxation bounds what the unexplored
part of the problem could still hold. A run that finishes has proven its best
solution optimal. A run stopped earlier reports how far from optimal it can
be.

## Example

```rust
use optopus::prelude::*;

// maximize x0 + 2 x1 + 3 x2 with at most two of them set
let vars: IntVars = (0..3).map(|_| IntVar::binary()).collect();
let objective = Expr::Var(0) + 2.0 * Expr::Var(1) + 3.0 * Expr::Var(2);
let prob = FormulaProblem::maximize(vars, objective).with_constraint(Constraint::Comparison {
    lhs: Expr::Var(0) + Expr::Var(1) + Expr::Var(2),
    rel: ConstraintRel::Le,
    rhs: Expr::Const(2.0),
    penalty_weight: 10.0,
});

let mut bnb = BranchAndBound::new(
    StopCondition::new(None, None, None),
    Box::new(LocalSearch::<IntChangeNeighbor>::new(StopCondition::iterations(100))),
    IntervalRelaxation,
);
let mut state = SearchState::new_with_seed(&prob, 1);
bnb.run(&mut state)?;
assert!(bnb.is_proven_optimal(&state.best_solution));
```

`examples/integer_branch_and_bound.rs` solves a 40 item knapsack this way, with
the fractional fill of Dantzig as the bound.

## Algorithm sketch

A node is the problem with the ranges of some variables narrowed. The open
nodes are kept by their bound, and each `run_once` takes the best of them.

1. If the node's bound cannot beat the best solution found so far, the node
   is dropped.
2. If every variable of the node holds a single value, that assignment is
   evaluated and the node is done.
3. Otherwise the inner heuristic searches the node, starting from the best
   solution with each value moved into the node's ranges. It runs on the
   problem rebuilt over those ranges, so its moves never leave the node.
4. One variable's range is split in two at its value in the best solution.
   The relaxation bounds each half, and a half is kept only if its bound can
   beat the best solution.

The run ends when no node is left, or earlier at the stop condition.

## What the problem needs

The problem implements `Branchable`, which `IntegerProblem` and
`FormulaProblem` already do. A problem of your own implements one method,
`restricted`, which returns the same problem over narrower ranges. The objective
must not depend on the ranges. A permutation cannot be narrowed one variable at
a time, and the run returns an error on one.

`IntegerProblem` is `Branchable` when its closures can be cloned. Each node
clones them, so a closure over a large table is better given a reference to the
table than the table itself.

## Relaxations

A relaxation implements `Relaxation<P>` and returns, for some ranges, a value
no assignment in them can beat, with the direction of the problem attached.
**A value that is not such a bound prunes the optimum without any error.**
Checking a new relaxation against enumeration on small instances is worth the
time. A bound in the wrong direction is reported as an error.

- `IntervalRelaxation` works on any `FormulaProblem`. It multiplies the ranges
  of the variables in each monomial, and charges each constraint the smallest
  penalty its expression could reach. It is valid on every formula and weak on
  most, since a variable in several monomials is taken at its worst in each.
- A closure `|prob, vars| -> Evaluable<f64>` is a relaxation, and is the way
  in for a bound that knows the problem.
- `branch_hint` may name the variable to split next. By default the search
  splits the lowest numbered variable that still has a choice.

## Constructor

```rust
BranchAndBound::new(
    stop_condition: StopCondition,
    heuristic: Box<dyn Heuristic<P>>,
    relaxation: R,
) -> Self
```

`P: Branchable` and `R: Relaxation<P>`.

## Behavior

- Each node counts as one iteration, on top of what the inner heuristic runs.
  The inner stop condition is a budget per node.
- `is_proven_optimal(&best)` is true once no open node can beat `best`.
  `dual_bound(&best)` is the best objective an assignment not yet ruled out
  could reach, which equals the best solution's objective once proven.
- An improvement found inside a node is recorded when the node's search
  returns, so the trajectory steps once per node.
- The result is only as exact as the relaxation. A constraint written as a
  penalty is optimized as a penalty, and a weight too small to rule out a
  violation leaves an optimum that violates it.

## Benchmark config

`BranchAndBound` is library only for now. The benchmark has no integer problem
kind to run it on.

## References

- Land, A. H. and Doig, A. G. "An automatic method of solving discrete
  programming problems". *Econometrica* 28(3), 497-520, 1960.
