# Defining a Custom Heuristic

Implement `Heuristic<P>` to plug your own algorithm into the rest of the
library — `SearchState`, the meta-heuristics (`Sequential`, `Iterated`,
`Restart`), and the benchmark runner all work with it unchanged.

The full runnable example lives at
[`examples/custom_heuristic.rs`](../../examples/custom_heuristic.rs)
(`cargo run --example custom_heuristic`).

## The `Heuristic<P>` trait

```rust
pub trait Heuristic<Problem: ProblemTrait> {
    fn clear(&mut self) {}
    fn is_done<'a>(&self, state: &SearchState<'a, Problem>) -> bool;
    fn run_once<'a>(&mut self, state: &mut SearchState<'a, Problem>) -> Result<(), OptError>;

    // Default `run` calls `clear()` then loops `run_once` while `!is_done`.
    fn run<'a>(&mut self, state: &mut SearchState<'a, Problem>) -> Result<(), OptError> { … }
}
```

You implement `is_done` and `run_once`; `run` is provided. Override `clear` if
your heuristic carries per-run state (counters, learned weights, etc.).

## Minimal first-improving search

```rust
use optopus::error::OptError;
use optopus::prelude::*;

struct FirstImprovingSearch<N> {
    stop_condition: StopCondition,
    _phantom: std::marker::PhantomData<N>,
}

impl<N> FirstImprovingSearch<N> {
    fn new(stop_condition: StopCondition) -> Self {
        Self { stop_condition, _phantom: std::marker::PhantomData }
    }
}

impl<P, N> Heuristic<P> for FirstImprovingSearch<N>
where
    P: ProblemTrait,
    N: MoveToNeighbor<P>,
{
    fn is_done<'a>(&self, state: &SearchState<'a, P>) -> bool {
        self.stop_condition.is_done(state)
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        let next_move = N::iter(state.instance, &state.solution)
            .find(|n| n.move_to_be_better_than(state.instance, &state.solution, &state.solution));

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

- `state.apply(&neighbor)` — applies the move, increments iteration, updates
  best if improved.
- `state.progress_iteration()` — increments iteration without applying anything
  (use this when you can't make progress this step).
- `N::iter(prob, sol)` — lazy iterator over moves; combine with `max_by`,
  `find`, `filter_best`, `.choose(&mut rng)` etc. as your strategy demands.

## Optional: parallel execution

Implement `ParallelHeuristic<P>` if your `run_once_par` can use rayon. The
default delegates to `run_once`, so this is purely a perf opt.

```rust
impl<P, N> ParallelHeuristic<P> for FirstImprovingSearch<N>
where P: ProblemTrait, N: MoveToNeighbor<P> + Send + Sync
{ /* override run_once_par */ }
```

`SearchState::get_best_move_par_chunks(iter, chunk_size)` evaluates a move
iterator in parallel and returns the best move — useful when neighborhoods are
large.

## Composing your heuristic

Once it implements `Heuristic<P>`, your algorithm can be:

- Wrapped in [`Restart`](../heuristics/meta.md#restart) to reset to a random
  solution on stagnation.
- Used as a phase of [`Iterated`](../heuristics/meta.md#iterated).
- Listed inside [`Sequential`](../heuristics/meta.md#sequential).
- Passed as the `mutation` argument of
  [`GeneticAlgorithm`](../heuristics/genetic_algorithm.md).

## Next reading

- [Concepts → SearchState API](../concepts.md#searchstate)
- [Composing heuristics](composing.md)
