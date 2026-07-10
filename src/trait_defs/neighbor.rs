use super::problem::ProblemTrait;
use super::rankable::Rankable;

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

    /// Picks a uniformly random move from the neighborhood, or `None` when the
    /// neighborhood is empty. Used every step by SA / LAHC / RandomWalk.
    ///
    /// <div class="warning">
    /// The default implementation reservoir-samples the full
    /// [`iter`](Self::iter) neighborhood, which may be inefficient (O(n) or
    /// worse per step). Override it with a direct sampler when a random move
    /// can be constructed cheaply. The override must draw uniformly from the
    /// same move set that <code>iter</code> yields and return <code>None</code>
    /// exactly when <code>iter</code> is empty.
    ///
    /// When this default is invoked at runtime, a one-shot
    /// <code>tracing::warn!</code> is emitted per concrete Move type (via
    /// <code>OnceLock</code>). Providing an override bypasses the default
    /// body entirely, so the warning never fires — no opt-in flag required.
    /// </div>
    fn random_neighbor(
        prob: &Problem,
        sol: &Problem::Solution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Option<Self>
    where
        Self: Sized,
    {
        // Monomorphization gives every concrete (Problem, Self) pair its own
        // copy of this static, so the warning fires once per Move type per
        // process — not once total, and not once per call site.
        static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        WARNED.get_or_init(|| {
            tracing::warn!(
                move_type = std::any::type_name::<Self>(),
                "Using the default reservoir-sampling implementation of \
                 random_neighbor, which walks the full neighborhood every \
                 step. Override it with a direct sampler for hot-path \
                 heuristics."
            );
        });
        use rand::seq::IteratorRandom;
        Self::iter(prob, sol).choose(rng)
    }

    /// Returns `true` if applying this move to `src` yields a solution better than `other`.
    ///
    /// <div class="warning">
    /// The default implementation clones the solution and applies the move to it, which may be inefficient.
    /// Override this method with a more efficient implementation if possible.
    ///
    /// When this default is invoked at runtime, a one-shot
    /// <code>tracing::warn!</code> is emitted per concrete Move type (via
    /// <code>OnceLock</code>). Providing an override bypasses the default
    /// body entirely, so the warning never fires — no opt-in flag required.
    /// </div>
    fn move_to_be_better_than(
        &self,
        prob: &Problem,
        src: &Problem::Solution,
        other: &Problem::Solution,
    ) -> bool {
        // Monomorphization gives every concrete (Problem, Self) pair its own
        // copy of this static, so the warning fires once per Move type per
        // process — not once total, and not once per call site.
        static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        WARNED.get_or_init(|| {
            tracing::warn!(
                move_type = std::any::type_name::<Self>(),
                "Using the default clone+apply implementation of \
                 move_to_be_better_than. Override it with a gain-based \
                 O(1) implementation for hot-path heuristics."
            );
        });
        let mut cloned = src.clone();
        self.apply_to_solution(prob, &mut cloned)
            .expect("apply_to_solution should not fail");
        cloned.is_better_than(other)
    }
}

#[cfg(test)]
mod default_move_warning_tests {
    use super::*;

    use crate::trait_defs::ProblemTrait;
    use std::sync::{Arc, Mutex};
    use tracing_subscriber::fmt::MakeWriter;

    // A toy problem whose moves rely on the default move_to_be_better_than.
    #[derive(Debug)]
    struct ToyProblem;

    #[derive(Clone, Debug)]
    struct ToySolution {
        value: i32,
    }

    impl Rankable for ToySolution {
        fn is_better_than(&self, other: &Self) -> bool {
            self.value > other.value
        }
    }

    impl ProblemTrait for ToyProblem {
        type Solution = ToySolution;
        fn new_solution(&self, _rng: &mut impl rand::Rng) -> Self::Solution {
            ToySolution { value: 0 }
        }
    }

    // Naive move: uses the default move_to_be_better_than (no override).
    #[derive(Clone, Debug)]
    struct NaiveAddOne;

    impl MoveToNeighbor<ToyProblem> for NaiveAddOne {
        fn apply_to_solution(
            &self,
            _prob: &ToyProblem,
            sol: &mut ToySolution,
        ) -> Result<(), crate::error::OptError> {
            sol.value += 1;
            Ok(())
        }
        fn iter(_p: &ToyProblem, _s: &ToySolution) -> impl Iterator<Item = Self> + Send {
            std::iter::empty()
        }
    }

    // Efficient move: overrides move_to_be_better_than so the warning path is bypassed.
    #[derive(Clone, Debug)]
    struct EfficientAddOne;

    impl MoveToNeighbor<ToyProblem> for EfficientAddOne {
        fn apply_to_solution(
            &self,
            _prob: &ToyProblem,
            sol: &mut ToySolution,
        ) -> Result<(), crate::error::OptError> {
            sol.value += 1;
            Ok(())
        }
        fn iter(_p: &ToyProblem, _s: &ToySolution) -> impl Iterator<Item = Self> + Send {
            std::iter::empty()
        }
        fn move_to_be_better_than(
            &self,
            _prob: &ToyProblem,
            src: &ToySolution,
            other: &ToySolution,
        ) -> bool {
            // Closed-form: applying +1 to src yields a better-than-other result iff src.value + 1 > other.value.
            src.value + 1 > other.value
        }
    }

    #[derive(Default, Clone)]
    struct BufWriter(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for BufWriter {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for BufWriter {
        type Writer = BufWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    fn captured<F: FnOnce()>(f: F) -> String {
        let buf = BufWriter::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(buf.clone())
            .with_max_level(tracing::Level::WARN)
            .with_ansi(false)
            .finish();
        tracing::subscriber::with_default(subscriber, f);
        String::from_utf8(buf.0.lock().unwrap().clone()).unwrap()
    }

    #[test]
    fn default_move_to_be_better_than_emits_warning() {
        let prob = ToyProblem;
        let src = ToySolution { value: 0 };
        let other = ToySolution { value: 0 };

        let logs = captured(|| {
            let m = NaiveAddOne;
            let result = m.move_to_be_better_than(&prob, &src, &other);
            assert!(result, "applying +1 to src (=0) should beat other (=0)");
            // Second call: the OnceLock has fired; no additional warning emitted.
            let _ = m.move_to_be_better_than(&prob, &src, &other);
        });

        assert!(
            logs.contains("default clone+apply"),
            "expected default warning in captured logs, got: {logs}"
        );
        // Warning should fire only once for this Move type.
        let count = logs.matches("default clone+apply").count();
        assert_eq!(
            count, 1,
            "expected exactly one warning, got {count}: {logs}"
        );
    }

    #[test]
    fn overridden_move_to_be_better_than_emits_no_warning() {
        let prob = ToyProblem;
        let src = ToySolution { value: 0 };
        let other = ToySolution { value: 0 };

        let logs = captured(|| {
            let m = EfficientAddOne;
            let result = m.move_to_be_better_than(&prob, &src, &other);
            assert!(result);
        });

        assert!(
            !logs.contains("default clone+apply"),
            "override path must skip the warning entirely, got: {logs}"
        );
    }

    #[test]
    fn default_random_neighbor_emits_warning_once_and_handles_empty() {
        use rand::SeedableRng;
        let prob = ToyProblem;
        let sol = ToySolution { value: 0 };
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);

        let logs = captured(|| {
            // NaiveAddOne::iter is empty, so the default must return None.
            let m = NaiveAddOne::random_neighbor(&prob, &sol, &mut rng);
            assert!(m.is_none());
            // Second call: the OnceLock has fired; no additional warning.
            let _ = NaiveAddOne::random_neighbor(&prob, &sol, &mut rng);
        });

        let count = logs.matches("reservoir-sampling").count();
        assert_eq!(
            count, 1,
            "expected exactly one random_neighbor warning, got {count}: {logs}"
        );
    }
}
