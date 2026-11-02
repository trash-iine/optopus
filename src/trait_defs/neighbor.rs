use super::problem::ProblemTrait;
use super::rankable::Rankable;
use super::tabu::EnabledTabu;

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
    /// body entirely, so the warning never fires, no opt-in flag required.
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

    /// This move's tabu policy, or `None` when it has none.
    ///
    /// [`apply`](crate::search_state::SearchState::apply) records the move it
    /// applies while the state is recording, and it is handed the move as a
    /// bare [`MoveToNeighbor`], so it cannot name [`EnabledTabu`]. This is how it
    /// asks. A move implementing that trait returns itself.
    ///
    /// ```ignore
    /// fn tabu_policy(&self) -> Option<&dyn EnabledTabu> { Some(self) }
    /// ```
    ///
    /// A move without one returns `None`, and applying it records nothing and
    /// succeeds normally, which is what lets it run
    /// [`LocalSearch`](crate::heuristic::LocalSearch) and
    /// [`SimulatedAnnealing`](crate::heuristic::SimulatedAnnealing).
    ///
    /// There is no default. A default of `None` would let a move that
    /// implements [`EnabledTabu`] forget this line and run with no tabu list
    /// and no complaint, so every move states which of the two it is.
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu>;

    /// Returns `true` if applying this move to `src` yields a solution better than `other`.
    ///
    /// <div class="warning">
    /// The default implementation clones the solution and applies the move to it, which may be inefficient.
    /// Override this method with a more efficient implementation if possible.
    /// A gain-based override built on
    /// <code>Evaluable::improves_over</code> fits a solution that ranks by its
    /// objective, and one ranked by hand needs an override that compares in
    /// the same order as its <code>Rankable</code>.
    ///
    /// When this default is invoked at runtime, a one-shot
    /// <code>tracing::warn!</code> is emitted per concrete Move type (via
    /// <code>OnceLock</code>). Providing an override bypasses the default
    /// body entirely, so the warning never fires, no opt-in flag required.
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

    use crate::trait_defs::{Evaluable, Evaluate, ProblemTrait, Rankable};
    use std::sync::{Arc, Mutex};
    use tracing_subscriber::fmt::MakeWriter;

    // A toy problem whose moves rely on the default move_to_be_better_than.
    #[derive(Debug)]
    struct ToyProblem;

    #[derive(Clone, Debug, Rankable, Evaluate)]
    struct ToySolution {
        #[objective(maximize)]
        value: i32,
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
        fn tabu_policy(&self) -> Option<&dyn crate::trait_defs::EnabledTabu> {
            None
        }

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
    // A binary problem that keeps the default random_variable.
    struct ToyBits;

    #[derive(Clone, Debug, Rankable)]
    struct ToyBitsSolution(Vec<bool>);

    impl Evaluate for ToyBitsSolution {
        fn evaluate(&self) -> Evaluable<f64> {
            Evaluable::Maximize(self.0.iter().filter(|&&b| b).count() as f64)
        }
    }

    impl ProblemTrait for ToyBits {
        type Solution = ToyBitsSolution;
        fn new_solution(&self, _rng: &mut impl rand::Rng) -> Self::Solution {
            ToyBitsSolution(vec![false, true, false])
        }
    }

    #[derive(Clone, Debug)]
    struct ToyFlip(usize);

    impl MoveToNeighbor<ToyBits> for ToyFlip {
        fn tabu_policy(&self) -> Option<&dyn crate::trait_defs::EnabledTabu> {
            None
        }

        fn apply_to_solution(
            &self,
            _prob: &ToyBits,
            sol: &mut ToyBitsSolution,
        ) -> Result<(), crate::error::OptError> {
            sol.0[self.0] = !sol.0[self.0];
            Ok(())
        }
        fn iter(_p: &ToyBits, s: &ToyBitsSolution) -> impl Iterator<Item = Self> + Send {
            (0..s.0.len()).map(ToyFlip)
        }
    }

    impl crate::trait_defs::BinaryProblem for ToyBits {
        type Flip = ToyFlip;
        fn variable_indices(&self) -> impl Iterator<Item = usize> + Send + '_ {
            0..3
        }
        fn variable(sol: &ToyBitsSolution, i: usize) -> bool {
            sol.0[i]
        }
        fn flip_move(_sol: &ToyBitsSolution, i: usize) -> ToyFlip {
            ToyFlip(i)
        }
    }

    /// The default draw still lands on every variable, and says once that it
    /// walks the indices.
    #[test]
    fn default_random_variable_warns_once_and_draws_every_variable() {
        use crate::trait_defs::BinaryProblem;
        use rand::SeedableRng;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(3);
        let mut seen = [false; 3];
        let logs = captured(|| {
            for _ in 0..100 {
                seen[ToyBits.random_variable(&mut rng).unwrap()] = true;
            }
        });
        assert_eq!(seen, [true; 3]);
        let count = logs.matches("BinaryProblem::random_variable").count();
        assert_eq!(
            count, 1,
            "expected exactly one warning, got {count}: {logs}"
        );
    }
}
