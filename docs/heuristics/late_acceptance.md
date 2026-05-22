# LateAcceptanceHillClimbing

LAHC: at each step, sample a random neighbor and accept it if the resulting
score is no worse than either the current score *or* the score recorded
`history_length` iterations ago. The history acts as an adaptive threshold
that requires no temperature tuning.

## Constructor

```rust
LateAcceptanceHillClimbing::<N>::new(
    stop_condition: StopCondition,
    history_length: usize,
) -> Self
```

`N` must satisfy `MoveToNeighbor<P> + Evaluate`.

**Panics** if `history_length == 0`.

`history_length` controls the exploitation/exploration trade-off:

| `history_length` | Behavior |
|---|---|
| `1` | Roughly hill climbing (only accepts non-worse moves vs. one step ago). |
| `5_000` | Reasonable default for many problems. |
| Larger | More diversification; slower convergence. |

## Example

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]));
let mut state = SearchState::new(&mc);
let mut lahc = LateAcceptanceHillClimbing::<MaxCutFlipNeighbor>::new(
    StopCondition::iterations(100_000),
    5_000,
);
lahc.run(&mut state)?;
# Ok::<(), optopus::error::OptError>(())
```

`clear()` empties the history buffer; the buffer is re-initialized on the
first `run_once` call after a `run`.
