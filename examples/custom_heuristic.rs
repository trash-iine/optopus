//! Minimal example of implementing a custom heuristic.
//!
//! Implements the `Heuristic` trait twice. `FirstImprovingSearch` scans the
//! neighborhood and applies the first improving move it finds.
//! `RandomDescent` draws one random move per iteration and decides whether to
//! take it from the move's own delta, the shape simulated annealing, late
//! acceptance and threshold-style rules share.
//!
//! How to run:
//! ```
//! cargo run --example custom_heuristic
//! ```

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
        let next_move = N::iter(state.instance, &state.solution)
            .find(|neighbor| state.is_neighbor_better_than_current(neighbor));

        if let Some(neighbor) = next_move {
            state.apply(&neighbor)?;
        } else {
            state.progress_iteration();
        }

        Ok(())
    }
}

struct RandomDescent<N> {
    stop_condition: StopCondition,
    _neighbor: std::marker::PhantomData<N>,
}

impl<N> RandomDescent<N> {
    fn new(stop_condition: StopCondition) -> Self {
        Self {
            stop_condition,
            _neighbor: std::marker::PhantomData,
        }
    }
}

impl<P, N> Heuristic<P> for RandomDescent<N>
where
    P: ProblemTrait,
    // `Evaluate` on the move is what lets the rule read the delta without
    // applying the move.
    N: MoveToNeighbor<P> + Evaluate,
{
    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        let neighbor: N = state.random_neighbor("RandomDescent")?;
        // How much applying the move would worsen the objective, whichever
        // way the problem optimizes. Positive is worse, negative is better.
        let worsening = neighbor.evaluate().minimized();
        if worsening <= 0.0 {
            // Takes the move, counts the iteration and updates the best.
            state.apply(&neighbor)?;
        } else {
            // Rejects it, counting the iteration only.
            state.progress_iteration();
        }
        Ok(())
    }
}

fn main() {
    let mc = MaxCut::new(Graph::from_edges([
        (0, 1, 1.0),
        (0, 2, 1.0),
        (0, 3, 1.0),
        (1, 2, 1.0),
        (2, 3, 1.0),
    ]));

    let mut state = SearchState::new(&mc);
    let mut heuristic =
        FirstImprovingSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(100));
    heuristic.run(&mut state).unwrap();

    println!(
        "[FirstImprovingSearch] best objective = {:.1} (iter {})",
        state.best_solution.objective, state.best_iteration
    );

    let mut state = SearchState::new(&mc);
    let mut heuristic = RandomDescent::<MaxCutFlipNeighbor>::new(StopCondition::iterations(100));
    heuristic.run(&mut state).unwrap();

    println!(
        "[RandomDescent]        best objective = {:.1} (iter {})",
        state.best_solution.objective, state.best_iteration
    );
}
