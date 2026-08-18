# Defining a Custom Heuristic

Implement `Heuristic<P>` to plug your own algorithm into the rest of the
library — `SearchState`, the meta-heuristics (`Sequential`, `Iterated`,
`VariableNeighborhoodSearch`, `Restart`), and the benchmark runner all work
with it unchanged.

The full runnable example lives at
[`examples/custom_heuristic.rs`](https://github.com/trash-iine/optopus/blob/main/examples/custom_heuristic.rs)
(`cargo run --example custom_heuristic`).

## The `Heuristic<P>` trait

```rust
pub trait Heuristic<Problem: ProblemTrait> {
    fn clear(&mut self) {}
    fn stop_condition(&self) -> &StopCondition;
    fn run_once<'a>(&mut self, state: &mut SearchState<'a, Problem>) -> Result<(), OptError>;

    // Default `is_done` delegates to `stop_condition()`.
    fn is_done<'a>(&self, state: &SearchState<'a, Problem>) -> bool { … }

    // Default `run` calls `clear()` then loops `run_once` while `!is_done`.
    fn run<'a>(&mut self, state: &mut SearchState<'a, Problem>) -> Result<(), OptError> { … }
}
```

You implement `stop_condition` and `run_once`; `is_done` and `run` are
provided. Override `is_done` only when your heuristic has a termination rule
the stop condition cannot express — "stop at a local optimum", as `LocalSearch`
and `LinKernighanHelsgaunForTsp` do, on top of the default. Override `clear` if
your heuristic carries per-run state (counters, learned weights, etc.).

## Minimal first-improving search

```rust
use optopus::error::OptError;
use optopus::prelude::*;

struct FirstImprovingSearch<N> {
    stop_condition: StopCondition,
    _neighbor: std::marker::PhantomData<N>,
}

impl<N> FirstImprovingSearch<N> {
    fn new(stop_condition: StopCondition) -> Self {
        Self {
            stop_condition,
            _neighbor: std::marker::PhantomData,
        }
    }
}

impl<P, N> Heuristic<P> for FirstImprovingSearch<N>
where
    P: ProblemTrait,
    N: MoveToNeighbor<P>,
{
    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        let instance = state.instance;
        let solution = &state.solution;
        let next_move = N::iter(instance, solution)
            .find(|neighbor| neighbor.move_to_be_better_than(instance, solution, solution));

        if let Some(neighbor) = next_move {
            state.apply(&neighbor)?;
        } else {
            state.progress_iteration();
        }

        Ok(())
    }
}
```

Key API touchpoints:

- `state.apply(&neighbor)`:  applies the move, increments iteration, updates
  best if improved.
- `state.apply_move_only(&neighbor)`: same, but defers the best update.
  `state.update_best()` must be applied at the end of a multi-move step.
- `state.progress_iteration()`: increments iteration without applying anything
  (use this when you can't make progress this step).
- `state.random_neighbor::<N>(context)`: draws one uniformly random move, or
  `OptError::InvalidState` when the neighborhood is empty. This is what
  SA / LAHC / `RandomWalk` call each step.
- `N::iter(prob, sol)`: lazy iterator over moves; combine with `max_by`,
  `find`, `filter_best`, `.choose(&mut rng)` etc. as your strategy demands.

## Optional: parallel evaluation

There is no parallel variant of `Heuristic` to implement. Parallelism belongs
to the neighbor type: `MoveToNeighbor::iter` returns `impl Iterator + Send`, so
an `iter` implementation whose per-candidate cost is heavy can evaluate
candidates with rayon and yield the results in order. `JobShopSwapNeighbor` does
exactly that above a size threshold (`src/problem/job_shop_scheduling/neighbor.rs`),
which keeps results independent of the thread count while every heuristic —
including yours — stays sequential.

## Composing your heuristic

Once it implements `Heuristic<P>`, your algorithm can be:

- Wrapped in [`Restart`](../heuristics/meta.md#restart) to reset to a random
  solution on stagnation.
- Used as a phase of [`Iterated`](../heuristics/meta.md#iterated).
- Listed inside [`Sequential`](../heuristics/meta.md#sequential).
- Used as the local search or a shake of
  [`VariableNeighborhoodSearch`](../heuristics/meta.md#variableneighborhoodsearch).
- Passed as the `mutation` argument of
  [`GeneticAlgorithm`](../heuristics/genetic_algorithm.md).

## Next reading

- [SearchState API](../search_state.md)
- [Meta-heuristics](../heuristics/meta.md)
