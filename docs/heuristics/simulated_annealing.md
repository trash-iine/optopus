# SimulatedAnnealing

Pick a uniformly random neighbor; accept with Boltzmann probability
`exp(−worsening / T)`. Improving moves are always accepted. The temperature
is multiplied by `cooling_rate` after every step.

## Constructor

```rust
SimulatedAnnealing::<N>::new(
    stop_condition: StopCondition,
    initial_temperature: f64,
    cooling_rate: f64,
) -> Self
```

`N` must satisfy `MoveToNeighbor<P> + Evaluate` (i.e. `Evaluate<f64>`). The
worsening amount is read from `Evaluable::worsening_amount()`, so the
direction of the underlying objective is handled automatically.

`clear()` resets the current temperature to `initial_temperature`.

## Acceptance rule

The shared helper `boltzmann_accept(delta: Evaluable<f64>, T: f64)` returns
`true` if `delta` improves the score, otherwise it draws a uniform random
number and compares against `exp(−worsening / T)`.

## Example

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]));
let mut state = SearchState::new(&mc);
let mut sa = SimulatedAnnealing::<MaxCutFlipNeighbor>::new(
    StopCondition::iterations(100_000),
    /* initial_temperature = */ 1.0,
    /* cooling_rate        = */ 0.9999,
);
sa.run(&mut state)?;
# Ok::<(), optopus::error::OptError>(())
```

## BangBangSimulatedAnnealing

Variant with an oscillating temperature schedule:

```rust
BangBangSimulatedAnnealing::<N>::new(
    stop_condition: StopCondition,
    initial_temperature: f64,
    cooling_rate: f64,
    min_wave_threashold: f64,
    max_wave_threashold: f64,
)
```

The temperature decays multiplicatively until it drops below
`min_wave_threashold`, then *grows* by dividing by `cooling_rate` until it
exceeds `max_wave_threashold`, and so on. The sawtooth profile occasionally
re-injects exploration when the search becomes too greedy.
