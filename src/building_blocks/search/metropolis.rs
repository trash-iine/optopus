//! The Metropolis rule, and the sweep every replica-based annealer runs.
//!
//! Simulated annealing accepts one proposal at a time by the rule. Population
//! annealing and parallel tempering hold many replicas, each at its own or a
//! shared temperature, and move every one of them by the same sweep of
//! proposals. The sweep knows nothing of how the temperatures are chosen, so it
//! lives here once and both call it.

use rand::Rng;
use rand::rngs::SmallRng;

use crate::trait_defs::{Evaluable, Evaluate, MoveToNeighbor, ProblemTrait, Rankable, rank_cmp};

/// Returns `true` with Boltzmann probability `exp(-worsening / temperature)`.
///
/// Accepts an [`Evaluable<f64>`] value that encodes both the optimization direction and
/// the objective change. Improving moves are always accepted, and worsening moves are
/// accepted with probability `exp(-worsening / T)`.
///
/// This decides by the number alone, which is right for a candidate that has
/// nothing but a number, as a ruin-and-recreate partial has. A move of the
/// problem goes through [`metropolis_accept`], which asks the problem's order
/// whether it improves.
pub fn boltzmann_accept(delta: Evaluable<f64>, temperature: f64, rng: &mut SmallRng) -> bool {
    let worsening = delta.minimized();
    worsening < 0.0 || rng.random::<f64>() < (-worsening / temperature).exp()
}

/// The Metropolis rule for the move `mv` from `sol`.
///
/// Whether the move improves is the problem's order to say, through
/// [`MoveToNeighbor::move_to_be_better_than`], and an improving move is taken
/// without touching the RNG. Only a move that does not improve is priced by its
/// [`Evaluate`], and taken with probability `exp(-worsening / temperature)`.
/// That number may be a surrogate the order does not follow, a penalized cost
/// beside a violations-first order for example, and what the search reports as
/// its best still follows the order.
#[inline]
pub fn metropolis_accept<P, N>(
    mv: &N,
    prob: &P,
    sol: &P::Solution,
    temperature: f64,
    rng: &mut SmallRng,
) -> bool
where
    P: ProblemTrait,
    N: MoveToNeighbor<P> + Evaluate,
{
    mv.move_to_be_better_than(prob, sol, sol)
        || rng.random::<f64>() < (-mv.evaluate().minimized() / temperature).exp()
}

/// Sweeps one replica `sweeps` times at `temperature`, each sweep proposing
/// `proposals` random moves of type `N` and applying the ones the Metropolis
/// rule accepts.
///
/// A proposal is drawn by [`MoveToNeighbor::random_neighbor`] and judged by
/// [`metropolis_accept`], so the cost per proposal is whatever those cost on
/// the problem. An empty neighborhood skips the proposal.
pub fn metropolis_sweeps<P, N>(
    replica: &mut P::Solution,
    rng: &mut SmallRng,
    prob: &P,
    temperature: f64,
    sweeps: usize,
    proposals: usize,
) where
    P: ProblemTrait,
    N: MoveToNeighbor<P> + Evaluate,
{
    for _ in 0..sweeps * proposals {
        let Some(mv) = N::random_neighbor(prob, replica, rng) else {
            continue;
        };
        if metropolis_accept(&mv, prob, replica, temperature, rng) {
            // `apply_to_solution` refreshes gain/objective incrementally.
            let _ = mv.apply_to_solution(prob, replica);
        }
    }
}

/// The proposals in one sweep, `fixed` if given and otherwise the size of
/// `N`'s neighborhood of `sol`.
///
/// A sweep is one pass over the system, so by default the length is the
/// neighborhood counted from [`MoveToNeighbor::iter`], once per episode since
/// it does not change size while the search runs. That is O(n) for a
/// single-variable move and is what the physics literature calls a sweep, but
/// it is O(n²) for a pairwise move such as 2-opt, and the count builds every
/// move only to discard it. A fixed length is how to avoid that.
pub fn sweep_length<P, N>(prob: &P, sol: &P::Solution, fixed: Option<usize>) -> usize
where
    P: ProblemTrait,
    N: MoveToNeighbor<P>,
{
    fixed.unwrap_or_else(|| N::iter(prob, sol).count())
}

/// The index of the best of `replicas`, the last of them on a tie.
pub fn best_replica<S: Rankable>(replicas: &[S]) -> usize {
    replicas
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| rank_cmp(*a, *b))
        .map_or(0, |(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// The two deterministic ends of the Metropolis rule need one draw each,
    /// not a hundred: an improving move takes the short-circuit before the RNG
    /// is touched, and at 1e-12 the exponential underflows to exactly 0.0, so
    /// no draw can clear it. Only the warm end is statistical.
    #[test]
    fn boltzmann_accept_follows_the_metropolis_rule() {
        let mut rng = SmallRng::seed_from_u64(42);

        assert!(boltzmann_accept(Evaluable::Maximize(1.0), 1e-12, &mut rng));
        assert!(!boltzmann_accept(
            Evaluable::Maximize(-1.0),
            1e-12,
            &mut rng
        ));

        let accepted = (0..1000)
            .filter(|_| boltzmann_accept(Evaluable::Maximize(-1.0), 1e9, &mut rng))
            .count();
        assert!(accepted > 900, "accepted only {accepted} of 1000");
    }

    /// The order decides whether a move improves, and the number only prices
    /// one that does not. Each case is at a temperature where the number alone
    /// would give the opposite answer.
    #[test]
    fn metropolis_accept_asks_the_order_before_the_number() {
        struct Line;
        #[derive(Clone)]
        struct At(i32);
        impl Rankable for At {
            fn is_better_than(&self, other: &Self) -> bool {
                self.0 < other.0
            }
        }
        impl ProblemTrait for Line {
            type Solution = At;
            fn new_solution(&self, _: &mut impl Rng) -> At {
                At(0)
            }
        }
        /// A step along the line, with a surrogate number that need not agree
        /// with the step.
        struct Step {
            by: i32,
            surrogate: f64,
        }
        impl Evaluate for Step {
            fn evaluate(&self) -> Evaluable<f64> {
                Evaluable::Minimize(self.surrogate)
            }
        }
        impl MoveToNeighbor<Line> for Step {
            fn apply_to_solution(
                &self,
                _: &Line,
                sol: &mut At,
            ) -> Result<(), crate::error::OptError> {
                sol.0 += self.by;
                Ok(())
            }
            fn iter(_: &Line, _: &At) -> impl Iterator<Item = Self> + Send {
                std::iter::empty()
            }
            fn move_to_be_better_than(&self, _: &Line, src: &At, other: &At) -> bool {
                At(src.0 + self.by).is_better_than(other)
            }
        }

        let mut rng = SmallRng::seed_from_u64(7);
        let cold = 1e-12;
        let at = At(0);
        let improves_but_priced_worse = Step {
            by: -1,
            surrogate: 5.0,
        };
        let worsens_and_priced_worse = Step {
            by: 1,
            surrogate: 5.0,
        };
        let level_but_priced_better = Step {
            by: 0,
            surrogate: -1.0,
        };

        assert!(metropolis_accept(
            &improves_but_priced_worse,
            &Line,
            &at,
            cold,
            &mut rng
        ));
        assert!(!metropolis_accept(
            &worsens_and_priced_worse,
            &Line,
            &at,
            cold,
            &mut rng
        ));
        assert!(metropolis_accept(
            &level_but_priced_better,
            &Line,
            &at,
            cold,
            &mut rng
        ));
    }
}
