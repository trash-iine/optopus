# BreakoutLocalSearchForMaxCut

Problem-specific heuristic for [MaxCut](../problems/max_cut.md). Alternates a
greedy local search phase with an adaptive perturbation phase, using the
optional `positive_gain` index on `MaxCutSolution` to enumerate only
improving flips in O(|improving|).

## Algorithm sketch

- **Greedy phase**: repeatedly apply the strictly best improving flip,
  updating a tabu map.
- **Perturbation phase**: choose between three perturbation types based on
  the non-improvement counter `omega`:
  - `omega == 0` (just improved or just started): **strong** — apply `l`
    random flips.
  - `omega > 0` (stuck): with probability `p · q` use **weak flip**
    (tabu-guided flip moves), with probability `p · (1 − q)` use
    **weak swap** (tabu-guided swaps), and **strong** otherwise.
  - `p = max(exp(−omega / t), p0)` decays as `omega` grows.
- The perturbation length `l` increases by 1 whenever the solution does not
  change, and resets to `l0` whenever it does.

## Constructor

```rust
BreakoutLocalSearchForMaxCut::new(
    tabu_tenure: (u64, u64),
    stop_condition: StopCondition,
    t: u64,
    l0: u64,
    p0: f64,
    q: f64,
) -> Self
```

| Parameter | Meaning |
|---|---|
| `tabu_tenure` | tabu tenure range `(min, max)` for the LS phase |
| `t` | period of the `omega` counter before it resets |
| `l0` | initial perturbation length |
| `p0` | minimum perturbation probability |
| `q` | fraction of weak perturbations using flip (vs. swap) |

## Reference

Benlic, U. and Hao, J.-K. "Breakout Local Search for the Max-Cut problem."
*Engineering Applications of Artificial Intelligence*, 26(3), 1162-1173, 2013.
