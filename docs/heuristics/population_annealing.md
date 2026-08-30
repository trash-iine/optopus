# PopulationAnnealingForMaxCut

**API:** [`PopulationAnnealingForMaxCut`](../api/optopus/heuristic/struct.PopulationAnnealingForMaxCut.html)

Problem-specific heuristic for [MaxCut](../problems/max_cut.md). Population
Annealing Monte Carlo (PAMC) keeps a population of `population_size` replicas
and cools a shared inverse temperature `β` upward, resampling the population at
every temperature step.

## Example

```rust
use optopus::heuristic::PopulationAnnealingForMaxCut;
use optopus::prelude::*;

let mut rng = seeded_rng(42);
let mc = MaxCut::new(Graph::erdos_renyi(800, 0.02, &mut rng));
let mut state = SearchState::new_with_seed(&mc, 42);

let mut pa = PopulationAnnealingForMaxCut::new(
    StopCondition::iterations(100_000),
    /* population_size = */ 50,
    /* initial_beta    = */ 0.1,
    /* delta_beta      = */ 0.02,
    /* sweeps_per_step = */ 50,
    /* reset_period    = */ Some(400),
    /* cluster_moves   = */ true,
);
pa.run(&mut state)?;
println!("cut weight = {}", state.best_solution.objective);
# Ok::<(), optopus::error::OptError>(())
```

`PopulationAnnealingForMaxCut` is not in the prelude — import it from
`optopus::heuristic`.

## Algorithm sketch

The population is seeded with `population_size` random solutions. Each
`run_once` is one temperature step:

1. **Metropolis sweeps** — every replica is swept `sweeps_per_step` times at the
   current `β`. One sweep proposes one flip per edged vertex; a flip with cut
   change `gain` is accepted with probability `min(1, exp(β · gain))`, through
   the same `boltzmann_accept` helper [SA](simulated_annealing.md) uses.
2. **Non-local cluster move** (when `cluster_moves` is on) — see below.
3. **Resampling** — replica `j` gets `τ_j = exp(−Δβ (E_j − E_min)) / Z · R`
   expected copies, with `E_j = −cut_j` shifted by `E_min` for numerical
   stability. High-cut replicas are preferentially replicated and the population
   is restored to exactly `population_size`.
4. **Periodic reset** — every `reset_period` steps `β` returns to
   `initial_beta`, recovering diversity once the population has converged. The
   global best survives the reset.

`state.iteration` advances by `sweeps_per_step` per step, so time-to-best and
the anytime trajectory stay meaningful against the other heuristics. All
randomness flows through `state.rng` in a fixed order, so seeded runs are
bit-reproducible.

### The non-local cluster move

A maximal **independent set** of zero-gain vertices is flipped in
each replica. Independence is what makes each flip exactly objective-preserving
— no two flipped vertices are adjacent, so no flip changes another's gain — and
that lets the population traverse energy plateaus single-spin Metropolis cannot
cross.

The set is built from `MaxCutSolution`'s optional
[`zero_gain` index](../problems/max_cut.md#notes), which this heuristic enables
on each replica, walking it from a random offset and marking each selection's
neighborhood ineligible via an `EpochMarks` scratch set.

## Constructor

```rust
PopulationAnnealingForMaxCut::new(
    stop_condition: StopCondition,
    population_size: usize,        // R, replicas
    initial_beta: f64,             // starting inverse temperature
    delta_beta: f64,               // increment per step
    sweeps_per_step: usize,        // Metropolis sweeps per replica per step
    reset_period: Option<usize>,   // None = never reset β
    cluster_moves: bool,
) -> Self
```

**Panics** if `population_size < 2`, `initial_beta <= 0`, `delta_beta <= 0`, or
`sweeps_per_step == 0`.

`clear()` drops the population and resets `β` and the step counter, so a fresh
episode restarts the anneal from `initial_beta`.

## Benchmark config

```toml
[[heuristics]]
kind = "PopulationAnnealingForMaxCut"
population_size = 50
initial_beta = 0.1        # optional (default shown)
delta_beta = 0.02         # optional (default shown)
sweeps_per_step = 50      # optional (default shown)
reset_period = 400        # optional (default shown; 0 disables resets)
cluster_moves = true      # optional (default shown)
[heuristics.stop_condition]
max_duration_secs = 30.0
```

## References

- Machta, J. "Population annealing with weighted averages: A Monte Carlo method
  for rough free-energy landscapes." *Phys. Rev. E* 82, 026704, 2010.
- Augmented PAMC with adaptive control and non-local cluster moves,
  [arXiv:2606.25203](https://arxiv.org/abs/2606.25203); a new G63 best-known via
  PAMC, [arXiv:2510.21105](https://arxiv.org/abs/2510.21105).
