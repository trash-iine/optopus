# ParallelTempering

**API:** [`ParallelTempering`](../api/optopus/heuristic/struct.ParallelTempering.html)

Parallel tempering, also called replica exchange Monte Carlo, runs
`num_replicas` copies of the search at fixed temperatures and lets copies at
neighbouring temperatures trade places. A configuration found in a good basin
by a hot replica walks down the ladder to where it is refined, and one stuck at
the cold end walks up to where it can leave.

Runs on any problem whose solution implements [`Evaluate`](../traits.md), with
the Metropolis proposal supplied as the move type `N`, the same requirement as
[PopulationAnnealing](population_annealing.md).

## Example

```rust
use optopus::heuristic::ParallelTempering;
use optopus::prelude::*;

let mc = MaxCut::new(Graph::erdos_renyi(800, 0.02, &mut seeded_rng(42)));
let mut state = SearchState::new_with_seed(&mc, 42);

let mut pt = ParallelTempering::<MaxCut, MaxCutFlipNeighbor>::new(
    StopCondition::duration(std::time::Duration::from_secs(10)),
    /* num_replicas        = */ 20,
    /* beta_min            = */ 0.1,
    /* beta_max            = */ 5.0,
    /* sweeps_per_exchange = */ 1,
);
pt.run(&mut state)?;
println!("cut weight = {}", state.best_solution.objective);
println!("exchanges accepted = {:?}", pt.exchange_acceptance());
```

`ParallelTempering` is not in the prelude, import it from `optopus::heuristic`.

## Algorithm sketch

The inverse temperatures are geometric from `beta_min` to `beta_max`, and each
replica starts from a random solution. Each `run_once` is one round.

1. Metropolis sweeps, every replica is swept `sweeps_per_exchange` times at
   its own temperature. A proposed move is accepted with probability
   `min(1, exp(−β · ΔE))`. This is the same sweep population annealing runs.
2. Exchange, the replicas at temperatures `k` and `k + 1` swap with
   probability `min(1, exp((β_k − β_{k+1})(E_k − E_{k+1})))`. One round tries
   the even pairs and the next the odd ones, so every pair is tried every
   other round and no replica is in two exchanges at once.

Nothing cools. The ladder is fixed, which is the difference from population
annealing, whose replicas share one temperature that rises and are resampled
toward the good ones.

`state.iteration` advances by `sweeps_per_exchange` per round, and the
incumbent is the best replica of the round. All randomness flows through
`state.rng` in a fixed order, so seeded runs are bit-reproducible.

## Choosing the ladder

The temperatures are in the units of the objective. `beta_max` should be cold
enough that the coldest replica is a descent, and `beta_min` hot enough that
the hottest one moves freely. Between them the ladder has to be dense enough
that neighbours overlap in energy, or exchanges are rarely accepted and a
configuration never crosses it. `exchange_acceptance()` reports the share of
exchanges accepted so far, and a value near zero means more replicas or a
narrower range.

## Constructor

```rust
ParallelTempering::<P, N>::new(
    stop_condition: StopCondition,
    num_replicas: usize,         // >= 2
    beta_min: f64,               // hottest, > 0
    beta_max: f64,               // coldest, > beta_min
    sweeps_per_exchange: usize,  // >= 1
) -> Self
```

`with_sweep_length(n)` fixes the proposals per sweep instead of counting the
neighborhood once per run, which matters for a pairwise move whose
neighborhood is quadratic, as it does for population annealing. `betas()`
returns the ladder.

## Benchmark config

```toml
[[heuristics]]
kind = "ParallelTempering"
neighbor = "Flip"
num_replicas = 20
beta_min = 0.1              # optional (default shown)
beta_max = 5.0              # optional (default shown)
sweeps_per_exchange = 1     # optional (default shown)
# sweep_length = 800        # optional, the neighborhood size by default
[heuristics.stop_condition]
max_duration_secs = 30.0
```

The ranges are checked when the config is read, so a bad ladder fails before
any run starts.

## References

- Hukushima, K. and Nemoto, K. "Exchange Monte Carlo method and application to
  spin glass simulations." *Journal of the Physical Society of Japan*, 65(6),
  1604-1608, 1996.
- Wang, W., Machta, J. and Katzgraber, H. G. "Comparing Monte Carlo methods for
  finding ground states of Ising spin glasses. Population annealing, simulated
  annealing, and parallel tempering." *Physical Review E*, 92, 013303, 2015
  (arXiv:1412.2104).
