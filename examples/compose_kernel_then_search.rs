//! Exact data reduction followed by a heuristic, composed from `MaxCutKernel`
//! and Breakout Local Search.
//!
//! `MaxCutKernel::new` shrinks a sparse graph by rules that keep the optimum,
//! and `SearchState::open_reduction` and `close_reduction` carry a search
//! across it. The heuristic only ever sees the smaller kernel, and every
//! solution it finds is lifted back to the original graph with its exact
//! objective. The loop below is the whole recipe, and
//! `docs/problems/max_cut_kernel.md` explains each line.
//!
//! The graph is drawn from a fixed seed with average degree near 2.5, where
//! the rules cascade. On a dense graph they remove nothing and the crossing is
//! pure overhead.
//!
//! Run with:
//! ```
//! cargo run --release --example compose_kernel_then_search
//! ```

use optopus::prelude::*;
use optopus::problem::MaxCutKernel;

fn main() {
    let n = 3_000;
    let mc = MaxCut::new(Graph::erdos_renyi(n, 2.5 / n as f64, &mut seeded_rng(5)));
    let budget = StopCondition::iterations(300_000);
    let bls = |stop| bls_for_max_cut(stop, (6, 160), 1000, 8, 0.8, 0.5);

    let mut state = SearchState::new_with_seed(&mc, 42);
    bls(budget.clone()).run(&mut state).unwrap();
    report("BLS on the graph", &state);

    let kernel = MaxCutKernel::new(&mc);
    println!(
        "kernel keeps {} of {} vertices",
        kernel.kernel().graph.num_vertices(),
        n
    );
    let mut inner = bls(StopCondition::iterations(30_000));
    let mut state = SearchState::new_with_seed(&mc, 42);
    while !budget.is_done(&state) {
        let before = state.iteration;
        let mut sub = state.open_reduction(&kernel);
        inner.run(&mut sub).unwrap();
        state.close_reduction(&kernel, &sub);
        if state.iteration == before {
            state.progress_iteration();
        }
    }
    report("BLS on the kernel", &state);
}

fn report(name: &str, state: &SearchState<MaxCut>) {
    println!(
        "{name:>18}: cut = {:.0} (after {} iterations)",
        state.best_solution.objective, state.iteration
    );
}
