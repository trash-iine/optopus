# PopulationAnnealing

**API:** [`PopulationAnnealing`](../api/optopus/heuristic/struct.PopulationAnnealing.html)

Population Annealing Monte Carlo (PAMC) keeps a population of `population_size`
replicas and cools a shared inverse temperature `β` upward, resampling the
population at every temperature step.

Runs on any problem whose solution implements [`Evaluate`](../traits.md), with
the Metropolis proposal supplied as the move type `N`.

## Example

```rust
use optopus::heuristic::PopulationAnnealing;
use optopus::prelude::*;

let mut rng = seeded_rng(42);
let mc = MaxCut::new(Graph::erdos_renyi(800, 0.02, &mut rng));
let mut state = SearchState::new_with_seed(&mc, 42);

let mut pa = PopulationAnnealing::<MaxCut, MaxCutFlipNeighbor>::new(
    StopCondition::iterations(100_000),
    /* population_size = */ 50,
    /* initial_beta    = */ 0.1,
    /* delta_beta      = */ 0.02,
    /* sweeps_per_step = */ 50,
    /* reset_period    = */ Some(400),
);
pa.run(&mut state)?;
println!("cut weight = {}", state.best_solution.objective);
```

`PopulationAnnealing` is not in the prelude, import it from
`optopus::heuristic`.

## Algorithm sketch

The population is seeded with `population_size` random solutions. Each
`run_once` is one annealing step, in the order the method is defined in:

1. Resampling, replica `j` gets `τ_j = exp(−Δβ (E_j − E_min)) / Z · R`
   expected copies, with `E_j` read through `Evaluate` and shifted by `E_min` for
   numerical stability. Low-energy replicas are preferentially replicated and
   the population is restored to exactly `population_size`.
2. `β` advances by `delta_beta`. Every `reset_period` steps it returns to
   `initial_beta` instead, recovering diversity once the population has
   converged; the global best survives the reset.
3. Metropolis sweeps, every replica is swept `sweeps_per_step` times at the new
   `β`. A proposed move is accepted with probability `min(1, exp(-β · ΔE))`
   through the same `boltzmann_accept` helper
   [SA](simulated_annealing.md) uses.

There is no sweep before the first resampling. The method starts from a random
population, which at high temperature is already the distribution the sweeps
would produce.

`state.iteration` advances by `sweeps_per_step` per step, so time-to-best and
the anytime trajectory stay meaningful against the other heuristics. All
randomness flows through `state.rng` in a fixed order, so seeded runs are
bit-reproducible.

### Sweep length

A sweep is one pass over the system, so by default it proposes as many moves as
`N`'s neighborhood holds, counted once per episode. For a single-variable move
that is the variable count and costs O(n) once. For a pairwise move such as
2-opt the neighborhood is O(n²), and the count builds every move only to discard
it, so pin the length instead:

```rust
let pa = PopulationAnnealing::<TspWithCoordinates, TspTwoOptNeighbor>::new(
    StopCondition::duration(std::time::Duration::from_secs(30)),
    50, 0.1, 0.02, 50, Some(400),
).with_sweep_length(1_000);
```

## Constructor

```rust
PopulationAnnealing::<P, N>::new(
    stop_condition: StopCondition,
    population_size: usize,        // R, replicas
    initial_beta: f64,             // starting inverse temperature
    delta_beta: f64,               // increment per step
    sweeps_per_step: usize,        // Metropolis sweeps per replica per step
    reset_period: Option<usize>,   // None = never reset
) -> Self
```

Panics if `population_size < 2`, `initial_beta <= 0`, `delta_beta <= 0`, or
`sweeps_per_step == 0`. `with_sweep_length` panics on `0`.

`clear()` drops the population and resets `β` and the step counter, so a fresh
episode restarts the anneal from `initial_beta`.

## Benchmark config

```toml
[[heuristics]]
kind = "PopulationAnnealing"
neighbor = "Flip"
population_size = 50
initial_beta = 0.1        # optional (default shown)
delta_beta = 0.02         # optional (default shown)
sweeps_per_step = 50      # optional (default shown)
reset_period = 400        # optional (default shown; 0 disables resets)
sweep_length = 2000       # optional (default: the neighborhood size)
[heuristics.stop_condition]
max_duration_secs = 30.0
```

## References

- Wang, Machta, Katzgraber. "Population annealing: Theory and application in
  spin glasses." Phys. Rev. E 92, 063307, 2015
  ([arXiv:1508.05647](https://arxiv.org/abs/1508.05647)), the algorithm
  implemented here.
- Machta, J. "Population annealing with weighted averages: A Monte Carlo method
  for rough free-energy landscapes." Phys. Rev. E 82, 026704, 2010.
