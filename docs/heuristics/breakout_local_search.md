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
