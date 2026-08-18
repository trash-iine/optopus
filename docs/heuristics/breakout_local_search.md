# BreakoutLocalSearchForMaxCut

**API:** [`BreakoutLocalSearchForMaxCut`](../api/optopus/heuristic/struct.BreakoutLocalSearchForMaxCut.html)

Problem-specific heuristic for [MaxCut](../problems/max_cut.md). Alternates a
greedy local search phase with an adaptive perturbation phase, using the
optional `positive_gain` index on `MaxCutSolution` to enumerate only improving
flips in O(|improving|). The descent, the tabu walk and the perturbations it
drives are free functions in `src/heuristic/specific/max_cut/ops/`, shared with
the other MaxCut heuristics; what is BLS's own is the schedule below. All of
them are handed the same `common::TabuLedger`, which is what stops a
perturbation undoing the descent that just ran.

## Algorithm

- **Greedy phase**: repeatedly apply the strictly best improving flip, updating
  a tabu map.
- **Perturbation phase**: `p = max(exp(−omega / t), p0)` is the probability of a
  *directed* (weak) perturbation, and it decays as the non-improvement counter
  `omega` grows:
  - `omega == 0` (the last descent improved the global best): `p = 1`, so a
    **weak** perturbation always runs — **weak flip** with probability `q`,
    **weak swap** with probability `1 − q`. This is the gentle end of the
    schedule, not the strong one.
  - `0 < omega <= t`: the same weak split with probability `p`, **strong**
    (random flips) with probability `1 − p`. As `omega` grows `p` decays toward
    `p0`, so strong perturbations become steadily more likely.
  - `omega > t`: a **strong** perturbation is forced and `omega` resets to 0.
- Both weak perturbations take the **highest-gain move that is not tabu**, and
  admit a tabu move only when it would beat the global best (aspiration). The
  weak swap picks one vertex per partition side, so it tracks a per-side best
  non-tabu vertex plus a per-side best overall for the aspiration test.
- The perturbation length `l` increases by 1 whenever the descent lands on the
  same local optimum as the previous round, and resets to `l0` whenever it
  escapes.

## Differences from the original scheme

Everything here was measured against the cut values Benlic & Hao publish, on
G22 / G27 / G33 / G35 / G39 at one tenth of their budget, five runs each.

- **The tenure parameter is doubled on the way in.** The original tenure is
  added once when a vertex is recorded and once more in the eligibility test, so
  a vertex stays forbidden for twice it. `VecTabuMap` stores a single tenure, so
  `paper_effective_tenure` doubles the caller's range and `tabu_tenure` keeps
  the original meaning, `rand[3, n/10]` on the G-set. Doubling only the upper
  bound does not reproduce it — the whole range has to scale.
- **`l0` scales with the instance: `0.01 · n`, not a constant.** A fixed
  `l0 = 80` is simultaneously ten times too strong at `n = 800` and 2.5 times
  too weak at `n = 20000`.
- **The weak swap selects the best non-tabu move.** An earlier version forced a
  swap of the longest-blocked vertex on each side unless aspiration allowed the
  best swap — a diversification move that degrades with vertex degree: on the
  degree-20 instances the perturbation supplied noise instead of direction.
  With the tenure fix above it closes the gap to the published values from −76
  to −16, and it removed the need for `VecTabuMap::blocked_until`.
- **`l` grows on a repeated *solution*, not a repeated objective.** Every G-set
  weight is ±1, so distinct local optima collide on the same cut value
  constantly; comparing objectives fired on 82.7% of rounds on G11 (measurement
  in `BlsSchedule::prev_local_optimum`).
- **No bucket sort.** The descent narrows its scan with the optional
  `positive_gain` index on `MaxCutSolution`, but still scans that set linearly
  for the maximum, and the tabu walk and both weak perturbations scan **all n**
  flip neighbours per move. The same move is selected either way, so this costs
  only speed — but the constant is large enough that wall-clock comparisons
  against published times are meaningless in either direction (only cut values
  compare), and that a budget in moves is not a budget in work. A caller that
  hands this engine a move budget is really handing it an `O(n)` multiple of
  one, so anything driving it on large instances wants to bound the *work* — a
  budget of `2n` moves is `O(n²)`, which at `n = 20000` is a different order of
  magnitude from what the number suggests.
- **A swap advances the iteration counter by 2**
  (`MaxCutSwapNeighbor::apply_to_iteration`), where BLS counts every move as
  one. That `+2` is a library-wide convention shared by every binary problem's
  swap, so it is not changed here for one heuristic's sake.

## Cases the original scheme leaves open

The eligible sets of the two weak perturbations can come out empty, and BLS as
published does not say what to do then. All three answers below are
**empirically dead on the G-set**: `progress_iteration` is the only thing that
increments `n_rejected`, and all 88 recorded BLS summaries report
`avg_n_rejected = 0.0`. That doubles as a tripwire — if `n_rejected` ever stops
being zero, one of them has started firing.

| Case | Behaviour here | Why it cannot fire on the G-set |
|---|---|---|
| every flip tabu, no aspiration | advance the iteration and skip the move | see above |
| a swap where one partition side has no vertex at all | advance the iteration twice (matching the swap's `+2` accounting) and skip | a side empties only on a degenerate instance |
| a swap where one side has no non-tabu vertex | take that side's best vertex anyway, breaking tabu **without** aspiration | the effective tenure is capped at `2·(n/10) = 0.2n` while a side holds roughly `0.5n` vertices, so a whole side cannot be tabu |

The third is the only one with real algorithmic content, and it *can* fire away
from the G-set: a caller that runs this engine on a small instance while keeping
a tenure tuned for a large one puts the whole of one side inside the tabu
window, which is precisely the regime the G-set never reaches.

## Constructor

```rust
BreakoutLocalSearchForMaxCut::new(
    stop_condition: StopCondition,
    tabu_tenure: (u64, u64),
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

## Benchmark config

```toml
[[heuristics]]
kind = "BreakoutLocalSearch"
tabu_tenure = [3, 80]     # density-scaled; see docs/benchmarks
t = 1000
l0 = 80                   # 0.01 * |V|
p0 = 0.8
q = 0.5

[heuristics.stop_condition]
max_duration_secs = 30
```

`tabu_tenure` is read as Benlic & Hao's `γ`: a vertex stays forbidden for `2γ`
moves. This is the one kind that doubles the key — the same range under
[`TabuSearch`](tabu_search.md) or [`RlBreakoutLocalSearch`](rl_breakout_local_search.md)
prohibits for half as long, so tuned values do not transfer between them.

## Reference

Benlic, U. and Hao, J.-K. "Breakout Local Search for the Max-Cut problem."
*Engineering Applications of Artificial Intelligence*, 26(3), 1162-1173, 2013.
