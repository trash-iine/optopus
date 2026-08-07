# BreakoutLocalSearchForMaxCut

Problem-specific heuristic for [MaxCut](../problems/max_cut.md). Alternates a
greedy local search phase with an adaptive perturbation phase, using the
optional `positive_gain` index on `MaxCutSolution` to enumerate only
improving flips in O(|improving|).

## Algorithm sketch

- **Greedy phase**: repeatedly apply the strictly best improving flip,
  updating a tabu map.
- **Perturbation phase**: `p = max(exp(−omega / t), p0)` is the probability of
  a *directed* (weak) perturbation, and it decays as the non-improvement
  counter `omega` grows:
  - `omega == 0` (the last descent improved the global best): `p = 1`, so a
    **weak** perturbation always runs — **weak flip** with probability `q`,
    **weak swap** with probability `1 − q`. This is the gentle end of the
    schedule, not the strong one.
  - `0 < omega <= t`: the same weak split with probability `p`, **strong**
    (random flips) with probability `1 − p`. As `omega` grows `p` decays
    toward `p0`, so strong perturbations become steadily more likely.
  - `omega > t`: a **strong** perturbation is forced and `omega` resets to 0.
- The perturbation length `l` increases by 1 whenever the descent lands on the
  same local optimum as the previous round, and resets to `l0` whenever it
  escapes.

> The paper's Algorithm 2 reads `if ω = 0 then` *random* perturbation, which
> taken literally means a random restart every time the search improves its
> best — because Algorithm 1 line 19 also sets `ω ← 0` on improvement. That
> contradicts both Formula (2) (`ω = 0` gives `P = 1`, i.e. directed for
> certain) and Section 2.3.1 ("apply more often directed perturbations … as the
> search progresses towards improved new local optima, the non-improving
> consecutive counter ω is small"). The only self-consistent reading is that
> Algorithm 2's `ω = 0` means "just reset by line 26", i.e. the `ω > T` branch,
> which is what is implemented above.

Both weak perturbations implement the paper's eligible sets literally: take the
**highest-gain move that is not tabu**, and admit a tabu move only when it
would beat the global best (aspiration). The weak swap is the paper's `M2`, so
it picks one vertex per partition side and needs a per-side "best non-tabu"
plus a per-side "best overall" for the aspiration test.

## Reproducing the paper's numbers

Two details of Benlic & Hao's description do not carry over literally and cost
a lot when they are missed. Both were found by measuring against the cut values
in the paper's Table 2 (sum of gaps over G22/G27/G33/G35/G39 at one tenth of
the benchmark budget, five runs each):

| | Σ gap to the paper |
|---|---|
| best non-tabu selection missing from the weak swap, single tenure | −76 |
| both corrected | **−16** |

- **The weak swap must select the best non-tabu move.** An earlier version
  applied the best swap only when aspiration allowed it and otherwise forced a
  swap of the longest-blocked vertex on each side. That is a diversification
  move, not the paper's `A2`, and it degrades with vertex degree: on the
  degree-20 random instances it left the perturbation supplying noise instead
  of direction, so the search only improved when `l0` grew towards a random
  restart (the measured optimum reached 64% of the graph). Correcting it also
  removed the need for `VecTabuMap::blocked_until`.
- **The tenure parameter is halved by the paper's own notation.** `H` holds
  "the iteration when the vertex was last moved *plus γ*" while the eligibility
  test asks for `(H_m + γ) < Iter`, so a vertex is really forbidden for `2γ`.
  `VecTabuMap` stores one tenure, so `BreakoutLocalSearch` doubles the caller's
  range on the way in (`paper_effective_tenure`) and `tabu_tenure` keeps the
  paper's meaning, `rand[3, |V|/10]` on the G-set. Doubling only the upper
  bound does not reproduce it — the whole range has to scale.

`l0` is the paper's `0.01 · |V|`, not a constant. A fixed `l0 = 80` across all
size bands is simultaneously ten times too strong at `n = 800` and 2.5 times too
weak at `n = 20000`; on the toroidal instances the paper's rule reproduces its
cut values exactly.

### Deviations that remain

**No bucket sort.** Section 2.2 of the paper keeps vertices in Fiduccia–Mattheyses
buckets — one doubly linked list per gain value, per partition side, plus a
`maxgain` pointer — so selecting the best move is O(1) and a move costs
O(degree(v)) to apply *and* re-bucket. This implementation has no such
structure. The descent narrows its scan with the optional `positive_gain` index
on `MaxCutSolution` (an improving flip must have positive gain, so the set is a
superset of the improving moves and shrinks near a local optimum), but it still
scans that set linearly for the maximum, and the tabu walk and both weak
perturbations scan **all n** flip neighbours on every move.

The consequence is only speed, never solution quality — the same move is
selected either way — but it is a large constant, and two things follow from it:

- **Wall-clock comparisons against the paper are not meaningful in either
  direction.** The published times are for a bucketed C++ implementation on a
  2008 Xeon; ours are for a linear-scan implementation on modern hardware. Only
  cut values compare.
- **Iteration budgets do not mean the same work.** A budget in moves costs
  `O(n)` per move here against `O(degree)` there, which is why the
  MaxCut-specific heuristics that drive this engine bound their tabu phases by
  *work* rather than by move count (see `CorrelationContractionSearch`'s
  `tabu_steps`, clamped by `2·10⁷ / n`).

**Swap iteration accounting.** A swap advances the iteration counter by 2
(`MaxCutSwapNeighbor::apply_to_iteration`) where the paper counts every move as
one iteration. That `+2` is a library-wide convention shared by every binary
problem's swap — it models a swap as two flips' worth of work — so it is not
changed here for one heuristic's sake.

### Cases the paper does not specify

Benlic & Hao define the eligible sets `A1` and `A2` but never say what a
perturbation should do when one of them comes out empty. This file is the
authority on the three answers chosen here; the code carries only a one-line
note and a pointer back to this section, so the reasoning lives in exactly one
place.

All three are also **empirically dead on the G-set**. `progress_iteration` is
the only thing that increments `n_rejected`, and every one of the 88 recorded
BLS summaries reports `avg_n_rejected = 0.0` — so across all 71 instances these
branches have never once been taken. That is what makes documenting them the
right call rather than fixing them, and it doubles as a tripwire: if
`n_rejected` ever stops being zero, one of these cases has started firing.

| Case | Behaviour here | Why it cannot fire on the G-set |
|---|---|---|
| `A1` empty (every flip tabu, no aspiration) | advance the iteration and skip the move | see above |
| `A2` empty because one partition side has no vertex at all | advance the iteration twice (matching the swap's `+2` accounting) and skip | a side empties only on a degenerate instance |
| `A2` has no non-tabu vertex on one side | take that side's best vertex anyway, breaking tabu **without** aspiration | the effective tenure is capped at `2·(n/10) = 0.2n` while a side holds roughly `0.5n` vertices, so a whole side cannot be tabu |

The third is the only one with real algorithmic content, and it *can* fire
outside the G-set: `CorrelationContractionSearch` solves contracted instances
whose cluster count is small and whose tenure is clamped to 25, which is the
regime where a side can run out of free vertices.

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
