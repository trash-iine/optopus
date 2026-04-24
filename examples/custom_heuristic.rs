//! 独自ヒューリスティック実装の最小サンプル。
//!
//! `Heuristic` trait を実装して、各反復で「最初に見つかった改善手」を採用する
//! シンプルな first-improving hill climber を定義します。
//!
//! 実行方法:
//! ```
//! cargo run --example custom_heuristic
//! ```

use optopus::error::OptError;
use optopus::prelude::*;

struct FirstImprovingSearch<N> {
    stop_condition: StopCondition,
    _phantom: std::marker::PhantomData<N>,
}

impl<N> FirstImprovingSearch<N> {
    fn new(stop_condition: StopCondition) -> Self {
        Self {
            stop_condition,
            _phantom: std::marker::PhantomData,
        }
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

fn main() {
    let mc = MaxCut::new(Graph::from_edges([
        (0, 1, 1.0),
        (0, 2, 1.0),
        (0, 3, 1.0),
        (1, 2, 1.0),
        (2, 3, 1.0),
    ]));

    let mut state = SearchState::new(&mc);
    let mut heuristic = FirstImprovingSearch::<MaxCutFlipNeighbor>::new(
        StopCondition::iterations(100),
    );
    heuristic.run(&mut state).unwrap();

    println!(
        "[FirstImprovingSearch] best objective = {:.1} (iter {})",
        state.best_solution.objective, state.best_iteration
    );
}
