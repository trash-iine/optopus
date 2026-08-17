# WalkSatForSat

Problem-specific heuristic for [MaxSAT](../problems/sat.md). WalkSAT/SKC keeps
the search **focused**: instead of scanning all `n` variables per step, it
samples a currently unsatisfied clause and flips one variable inside it.

## Algorithm sketch

Every literal of an unsatisfied clause is false, so flipping *any* of its
variables satisfies that clause. Which one is the Selman–Kautz–Cohen rule.
Each `run_once`:

1. **Sample** a uniformly random unsatisfied clause.
2. **Score** each of its variables by *break count* — how many currently
   satisfied clauses that flip would break.
3. **Choose** — if some variable has break count `0`, take it (a free move);
   otherwise flip a random variable of the clause with probability `noise`, and
   the minimum-break one otherwise.
4. **Commit** the flip and update the scratch state in O(degree).

Per-step cost is `O(clause length × variable degree)` and independent of the
total variable count, which is why this heuristic reaches instances the generic
`LocalSearch` / `TabuSearch` (O(n) per move) cannot.

`is_done` stops early once every clause is satisfied — for MaxSAT that is a
global optimum, so there is nothing left to improve.

### Its own flip

The move deliberately does **not** go through `SatFlipNeighbor::apply`, which
refreshes the cached `gain[]` over every neighbor variable — `O(degree²)` and
the dominant cost on dense instances. WalkSAT never reads `gain[]`; it selects
from its own satisfying-literal counts. So it updates only `x`, `n_satisfied`
and its scratch in O(degree), and restores a valid `gain[]` once at the end of
the run. That is the whole speed advantage.

The scratch is three structures rebuilt once per run and maintained
incrementally: per-clause satisfied-literal counts, a dense list of unsatisfied
clauses with O(1) membership, and a variable → clause-occurrence index.

### Adaptive noise

With `adaptive = true`, `noise` is only the starting value and follows Hoos'
schedule: it is lowered by a factor `φ/2` (φ = 0.2) whenever the unsatisfied
count reaches a new low, and raised toward 1 by `φ` after `n_clauses / 6` flips
without improvement.

## Constructor

```rust
WalkSatForSat::new(
    stop_condition: StopCondition,
    noise: f64,        // probability of a random walk step within the clause
    adaptive: bool,    // Hoos' automatic noise adjustment
) -> Self
```

**Panics** if `noise` is outside `[0.0, 1.0]`.

`clear()` drops the scratch and resets the working noise to `noise`, so a fresh
episode starts clean. Multi-restart is composed externally —
[`Restart`](meta.md#restart) around `WalkSat` is the usual form.

## Benchmark config

```toml
[[heuristics]]
kind = "WalkSat"
noise = 0.3              # optional (default shown)
adaptive_noise = false   # optional (default shown)
[heuristics.stop_condition]
max_duration_secs = 30.0
```

## References

- Selman, B., Kautz, H. A., and Cohen, B. "Noise Strategies for Improving Local
  Search." *Proc. AAAI-94*, 337-343, 1994.
- Hoos, H. H. "An Adaptive Noise Mechanism for WalkSAT." *Proc. AAAI-02*,
  655-660, 2002.
