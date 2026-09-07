# The clone/merge triad has no wrapper, and should not get one

- Status: adopted
- Area: search_state
- Date: 2026-08
- Code: src/search_state/mod.rs

## Decision

Every meta-heuristic isolates a phase by hand.

```rust
let mut sub = state.clone_for_new_run(kind);
inner.run(&mut sub)?;
state.update_state(sub);
```

There is deliberately no `run_sub(kind, heuristic)` wrapping the three calls.
Both halves are public, every user-facing document teaches this form, and a
wrapper would have to name `Heuristic` in `search_state`, which must not depend
on it. Nothing about cloning and merging a state needs to know what a heuristic
is.

## Measurement

None. This is a layering constraint rather than a performance question.

## Do not retry

The triad looks like boilerplate worth collapsing. Collapsing it inverts the
dependency between `search_state` and `heuristic`.
