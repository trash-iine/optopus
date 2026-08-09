//! Runs any MaxCut heuristic on an exactly reduced instance.

use rand::RngCore;

use crate::error::OptError;
use crate::heuristic::{Heuristic, StopCondition};
use crate::problem::{MaxCut, MaxCutFlipNeighbor, MaxCutKernel, MaxCutSolution};
use crate::search_state::{SearchState, SearchStateCloneType};

/// Kernelization wrapper: reduces the instance once with [`MaxCutKernel`],
/// runs `inner` on the kernel, and lifts the result back.
///
/// The reduction is exact, so this changes *what the inner heuristic has to
/// search*, never which solutions are reachable. On instances the rules cannot
/// touch — regular graphs, dense graphs — the wrapper detects that up front
/// and hands the original state straight to `inner`, so it is safe to leave on.
///
/// ```
/// use optopus::prelude::*;
/// use optopus::heuristic::KernelizedSearchForMaxCut;
///
/// // A caterpillar: the pendant vertices reduce away before the search runs.
/// let mut edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)];
/// edges.extend((4..12).map(|v| (v % 4, v, 1.0)));
/// let mc = MaxCut::from_edges(edges);
///
/// let mut state = SearchState::new_with_seed(&mc, 1);
/// let mut search = KernelizedSearchForMaxCut::new(
///     StopCondition::iterations(1_000),
///     Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
///         StopCondition::iterations(1_000),
///     )),
/// );
/// search.run(&mut state).unwrap();
/// assert_eq!(
///     state.best_solution.objective,
///     mc.calculate_cut_size(&state.best_solution.x)
/// );
/// ```
pub struct KernelizedSearch {
    stop_condition: StopCondition,
    inner: Box<dyn Heuristic<MaxCut>>,
    /// Built on the first cycle after `clear`, then reused.
    kernel: Option<MaxCutKernel>,
}

impl KernelizedSearch {
    /// Wraps `inner`, which will be run on the kernel.
    pub fn new(stop_condition: StopCondition, inner: Box<dyn Heuristic<MaxCut>>) -> Self {
        Self {
            stop_condition,
            inner,
            kernel: None,
        }
    }

    /// Builds the kernel on first use and reports what it achieved.
    fn kernel_for(&mut self, instance: &MaxCut) -> &MaxCutKernel {
        if self.kernel.is_none() {
            let kernel = MaxCutKernel::reduce(instance);
            tracing::info!(
                vertices = instance.graph.num_vertices(),
                edges = instance.graph.num_edges(),
                kernel_vertices = kernel.kernel().graph.num_vertices(),
                kernel_edges = kernel.kernel().graph.num_edges(),
                removed = kernel.removed_vertices(),
                offset = kernel.offset(),
                "kernelize: reduction finished"
            );
            self.kernel = Some(kernel);
        }
        self.kernel.as_ref().expect("just built")
    }
}

impl Heuristic<MaxCut> for KernelizedSearch {
    fn clear(&mut self) {
        self.inner.clear();
        // A new episode may be on a different instance.
        self.kernel = None;
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, MaxCut>) -> Result<(), OptError> {
        let instance = state.instance;
        let before = state.iteration;
        if self.kernel_for(instance).is_trivial() {
            // Nothing was reduced, so there is no point in solving a copy of
            // the same instance through an extra index mapping.
            state.run_sub(self.inner.as_mut(), SearchStateCloneType::Simple)?;
            ensure_progress(state, before);
            return Ok(());
        }
        let kernel = self.kernel.as_ref().expect("built above");

        // Warm-start the kernel from the incumbent so repeated cycles keep
        // improving rather than restarting from scratch.
        let seed = state.rng.next_u64();
        let start =
            MaxCutSolution::new_from_assignment(kernel.kernel(), kernel.project(&state.solution.x));
        let mut sub_state = SearchState::with_solution_and_seed(kernel.kernel(), start, seed);
        self.inner.run(&mut sub_state)?;
        // The kernel is a separate instance, so `update_state` cannot be used;
        // the counters are merged explicitly instead (same as the contracted
        // solve in CorrelationContractionSearch). Only the solution crosses by
        // hand, through the lifting below.
        state.merge_foreign_sub_run(&sub_state);

        // Apply the lifted assignment one flip at a time so the incremental
        // gain cache and objective stay exact.
        let lifted = kernel.lift(&sub_state.best_solution.x);
        for (v, &target) in lifted.iter().enumerate() {
            if state.solution.x[v] != target {
                let flip = MaxCutFlipNeighbor {
                    i: v,
                    gain: state.solution.gain[v],
                };
                state.apply_move_only(&flip)?;
            }
        }
        state.update_best();
        ensure_progress(state, before);
        Ok(())
    }
}

/// Guarantees the outer loop makes progress.
///
/// An inner heuristic that halts at a local optimum (LocalSearch, LKH) returns
/// immediately without consuming an iteration, so a wrapper that only forwards
/// to it would spin forever against an iteration budget. Counting one
/// iteration per cycle keeps the outer stop condition finite.
fn ensure_progress(state: &mut SearchState<'_, MaxCut>, before: u64) {
    if state.iteration == before {
        state.progress_iteration();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::{LocalSearch, TabuSearch};
    use crate::problem::MaxCutFlipNeighbor;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn sparse_instance(seed: u64, n: usize) -> MaxCut {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut edges = Vec::new();
        // Average degree ~2.5: low enough that the reduction cascades.
        for i in 0..n {
            for j in (i + 1)..n {
                if rng.random_bool(2.5 / n as f64) {
                    edges.push((i, j, 1.0));
                }
            }
        }
        edges.push((n - 1, n - 2, 1.0));
        MaxCut::from_edges(edges)
    }

    fn run(mc: &MaxCut, kernelize: bool) -> MaxCutSolution {
        let mut state = SearchState::new_with_seed(mc, 3);
        let inner = || {
            Box::new(TabuSearch::<MaxCutFlipNeighbor>::new(
                StopCondition::iterations(2_000),
                (2, 8),
                None,
            )) as Box<dyn Heuristic<MaxCut>>
        };
        if kernelize {
            let mut search = KernelizedSearch::new(StopCondition::iterations(20_000), inner());
            search.run(&mut state).unwrap();
        } else {
            let mut search = inner();
            search.run(&mut state).unwrap();
        }
        state.best_solution.clone()
    }

    /// Solving through the kernel must leave an exactly evaluated solution of
    /// the original instance.
    #[test]
    fn kernelized_search_keeps_the_original_objective_exact() {
        for seed in 0..5u64 {
            let mc = sparse_instance(seed, 200);
            let best = run(&mc, true);
            assert_eq!(
                best.objective,
                mc.calculate_cut_size(&best.x),
                "objective diverged from the assignment (seed {seed})"
            );
        }
    }

    /// The reduction must be worth something on a sparse instance: the kernel
    /// is strictly smaller and the search still reaches at least as good a cut
    /// as the same heuristic run on the full graph.
    #[test]
    fn kernelizing_shrinks_a_sparse_instance_without_losing_quality() {
        let mc = sparse_instance(11, 300);
        let kernel = MaxCutKernel::reduce(&mc);
        assert!(
            kernel.kernel().graph.num_vertices() < mc.graph.num_vertices(),
            "the test instance must actually reduce"
        );
        assert!(run(&mc, true).objective >= run(&mc, false).objective);
    }

    /// On an instance the rules cannot touch the wrapper must be inert, not
    /// merely harmless — it hands the state to the inner heuristic unchanged.
    #[test]
    fn a_non_reducible_instance_falls_through_to_the_inner_heuristic() {
        // 4-regular circulant: no vertex has degree below 3.
        let n = 40usize;
        let edges: Vec<_> = (0..n)
            .flat_map(|i| [(i, (i + 1) % n, 1.0), (i, (i + 2) % n, 1.0)])
            .collect();
        let mc = MaxCut::from_edges(edges);
        assert!(MaxCutKernel::reduce(&mc).is_trivial());

        let wrapped = {
            let mut state = SearchState::new_with_seed(&mc, 5);
            let mut search = KernelizedSearch::new(
                StopCondition::iterations(3_000),
                Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
                    StopCondition::iterations(3_000),
                )),
            );
            search.run(&mut state).unwrap();
            state.best_solution.x.clone()
        };
        let bare = {
            let mut state = SearchState::new_with_seed(&mc, 5);
            let mut search =
                LocalSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(3_000));
            search.run(&mut state).unwrap();
            state.best_solution.x.clone()
        };
        assert_eq!(wrapped, bare, "the wrapper must not perturb the search");
    }
}
