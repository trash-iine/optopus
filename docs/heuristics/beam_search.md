# BeamSearch

Maintains a beam of `beam_width` candidate solutions in parallel. Each step:

1. Expand every neighbor of every beam member.
2. Set `state.solution` to the best candidate (refresh `best_solution`).
3. Keep the top `beam_width` candidates — the rest are dropped.

Step 3 uses `select_nth_unstable_by` (O(n) expected), since order *within*
the surviving beam doesn't matter.

## Constructor

```rust
BeamSearch::<P, N>::new(
    stop_condition: StopCondition,
    beam_width: usize,
) -> Self
```

`N` must satisfy `MoveToNeighbor<P> + Rankable`.

**Panics** if `beam_width == 0`.

`clear()` empties the beam; the beam is re-seeded from `state.solution` on
the first `run_once` call after a `run`.

## Cost

Unlike `LocalSearch`/`TabuSearch`, BeamSearch *materializes* every neighbor:
each step is O(beam_width × |neighborhood|) memory and applies that many
`apply_to_solution` calls. Use modest `beam_width` for problems with large
neighborhoods.

## Example

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]));
let mut state = SearchState::new(&mc);
let mut bs = BeamSearch::<MaxCut, MaxCutFlipNeighbor>::new(
    StopCondition::iterations(1_000),
    /* beam_width = */ 5,
);
bs.run(&mut state)?;
# Ok::<(), optopus::error::OptError>(())
```

## References

- Ow, P. S. and Morton, T. E. "Filtered Beam Search in Scheduling."
  *International Journal of Production Research*, 26(1), 35-62, 1988.
