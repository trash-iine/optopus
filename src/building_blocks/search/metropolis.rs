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
/// the objective change. Improving moves are always accepted; worsening moves are
/// accepted with probability `exp(-worsening / T)`.
pub fn boltzmann_accept(delta: Evaluable<f64>, temperature: f64, rng: &mut SmallRng) -> bool {
    let worsening = delta.minimized();
    worsening < 0.0 || rng.random::<f64>() < (-worsening / temperature).exp()
}

/// Sweeps one replica `sweeps` times at `temperature`, each sweep proposing
/// `proposals` random moves of type `N` and applying the ones the Metropolis
/// rule accepts.
///
/// A proposal is drawn by [`MoveToNeighbor::random_neighbor`] and priced by its
/// [`Evaluate`], so the cost per proposal is whatever those cost on the
/// problem. An empty neighborhood skips the proposal.
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
        if boltzmann_accept(mv.evaluate(), temperature, rng) {
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
}
