# LateAcceptanceHillClimbing

**API:** [`LateAcceptanceHillClimbing`](../api/optopus/heuristic/struct.LateAcceptanceHillClimbing.html)

Accepts a move when the resulting score is no worse than the score recorded
`history_length` iterations ago. That lagged score is an adaptive threshold,
which is what lets the search leave a local optimum without a temperature to
tune.

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
println!("cut weight = {}", state.best_solution.objective);
# Ok::<(), optopus::error::OptError>(())
```

## Algorithm sketch

Each `run_once`:

1. **Sample** a uniformly random neighbor.
2. **Score** it: the running score is always higher-is-better (to minimize,
   using `- score`), so the candidate is `current − worsening_amount()` and the
   direction of the underlying objective is handled by `Evaluate`.
3. **Accept** if the candidate is no worse than the current score or no worse
   than `history[i mod history_length]` — the score from `history_length` steps
   ago. A rejected move only advances the iteration counter.
4. **Record** the (possibly unchanged) current score into that same slot, and
   advance `i`.

Both the running score and the history buffer start at `0.0`, so what the
buffer holds is each step's score *relative to the initial solution*, not the
objective itself. Every entry carries the same offset, so the comparison in
step 3 is unaffected.

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

`clear()` empties the history buffer; the buffer is re-initialized on the
first `run_once` call after a `run`.

## Benchmark config

```toml
[[heuristics]]
kind = "LateAcceptanceHillClimbing"
neighbor = "Flip"        # required; the valid values are per-problem
history_length = 5_000   # required; must be >= 1
[heuristics.stop_condition]
max_iteration = 100_000
```

## References

- Burke, E. K. and Bykov, Y. "The Late Acceptance Hill-Climbing Heuristic."
  *European Journal of Operational Research*, 258(1), 70-78, 2017.
