# Stop Conditions

Every heuristic takes a [`StopCondition`] that decides when to stop. The
condition is checked at the top of each iteration and **fires as soon as any
configured limit is reached** (logical OR, not AND).

## Builder API

```rust
use optopus::prelude::*;
use std::time::Duration;

// Single-criterion constructors:
StopCondition::iterations(1_000_000);
StopCondition::duration(Duration::from_secs(30));
StopCondition::failed_updates(10_000);

// Chain extra criteria with `with_*`:
StopCondition::iterations(1_000_000)
    .with_duration(Duration::from_secs(30))
    .with_failed_updates(10_000);
```

| Criterion | Meaning |
|---|---|
| `max_iteration` | Stop after this many iterations have elapsed in the current run. |
| `max_duration` | Stop after wall-clock duration since the run started. |
| `max_failed_update` | Stop when this many iterations have passed without improving `best_solution`. |

`new(max_iteration, max_duration, max_failed_update)` is also available for
constructing a `StopCondition` from `Option` fields directly (useful when
deserializing from config).

## Sub-runs

Inside the sub-run clone/merge pattern (see
[concepts](../concepts.md#sub-run-clonemerge-pattern)), iteration counts are
measured **relative to the start of the sub-run**, not the global iteration.
The outer condition still uses the global counter, so an inner
`failed_updates(100)` triggers based on the inner phase's progress while the
outer `iterations(10_000)` budget governs the overall run.

## Tips

- `iterations` is the most reproducible (no clock dependency) — use it in
  tests and benchmarks where you need deterministic stops.
- `duration` is best for time-budgeted comparisons across machines.
- `failed_updates` is the natural condition for "run until it stops finding
  improvements" — pair with `LocalSearch` (which forces it to `1`) or with the
  inner phase of `Iterated`.
- Combine criteria when you want a soft target plus a hard cap: e.g.,
  `failed_updates(1_000).with_duration(Duration::from_secs(60))`.
