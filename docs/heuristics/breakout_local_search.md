# BreakoutLocalSearchForMaxCut

**API:** [`BreakoutLocalSearchForMaxCut`](../api/optopus/heuristic/struct.BreakoutLocalSearchForMaxCut.html)

Problem-specific heuristic for [MaxCut](../problems/max_cut.md). Alternates a
greedy local search phase with an adaptive perturbation phase. The descent is
the generic [`LocalSearch`](local_search.md); the tabu walk and the
perturbations it drives are free functions in
`src/heuristic/specific/max_cut/ops/`, shared with the other MaxCut heuristics.
What is BLS's own is the schedule below. All of them record into — and read —
the tabu memory on the `SearchState` they are handed, which is what stops a
perturbation undoing the descent that just ran.

## Example

```rust
use optopus::prelude::*;

let mut rng = seeded_rng(42);
let mc = MaxCut::new(Graph::erdos_renyi(800, 0.02, &mut rng));
let mut state = SearchState::new_with_seed(&mc, 42);

let mut bls = BreakoutLocalSearchForMaxCut::new(
    StopCondition::iterations(100_000),
    /* tabu_tenure = */ (5, 150),
    /* t           = */ 1_000,
    /* l0          = */ 8,          // 0.01 * |V|
    /* p0          = */ 0.8,
    /* q           = */ 0.5,
);
bls.run(&mut state)?;
println!("cut weight = {}", state.best_solution.objective);
# Ok::<(), optopus::error::OptError>(())
```

`l0` and `tabu_tenure` are instance-dependent, so they are derived from
`|V| = 800` here rather than left at a constant — see
[Benchmark config](#benchmark-config).

## Algorithm sketch

- **Greedy phase**: repeatedly apply the strictly best improving flip, updating
  a tabu map.
- **Perturbation phase**: `p = max(exp(−omega / t), p0)` is the probability of a
  *directed* (weak) perturbation, and it decays as the non-improvement counter
  `omega` grows:
  - `omega == 0` — the last descent improved the global best, or `omega` just
    passed `t` and was reset: a strong perturbation (random flips) runs.
  - `0 < omega <= t`: **weak flip** with probability `p * q`, **weak swap** with
    probability `p * (1 − q)`, **strong** (random flips) with probability
    `1 − p`. As `omega` grows `p` decays toward `p0`, so strong perturbations
    become steadily more likely.
  - `omega > t`: `omega` resets to 0, which the first branch then reads as a
    forced strong perturbation.
- Both weak perturbations take the highest-gain move that is not tabu, and
  admit a tabu move only when it would beat the global best (aspiration rule).
  These weak perturbations apply `l` best flip / swap moves.
- The perturbation length `l` increases by 1 whenever the descent lands on the
  same local optimum as the previous round, and resets to `l0` whenever it
  escapes.

## Differences from the original scheme

Everything here was measured against the cut values Benlic & Hao publish, on
G22 / G27 / G33 / G35 / G39 at one tenth of their budget, five runs each.

- **The tenure parameter is doubled on the way in.** The original tenure is
  added once when a vertex is recorded and once more in the eligibility test, so
  a vertex stays forbidden for twice it. `TabuMemory` stores a single tenure, so
  `paper_effective_tenure` doubles the caller's range and `tabu_tenure` keeps
  the original meaning, `rand[3, n/10]` on the G-set. Doubling only the upper
  bound does not reproduce it — the whole range has to scale.
- **No bucket sort.** The original buckets vertices by gain, so selecting a
  maximum-gain move is O(1) and a move costs only the O(degree(v)) rebucketing
  its gain update already implies. Here every phase selects by linear scan over
  **all n** flip neighbours, O(n) — the descent included, since it is the
  generic `LocalSearch`. The gain update itself is O(degree(v)). The same move
  is selected either way, so this costs only speed. The descent used to narrow
  its scan with the optional `positive_gain` index on `MaxCutSolution`; see
  below for what dropping that cost.
- **A swap advances the iteration counter by 2**
  (`MaxCutSwapNeighbor::apply_to_iteration`), where BLS counts every move as
  one. That `+2` is a library-wide convention shared by every binary problem's
  swap, so it is not changed here for one heuristic's sake.

## The descent is the generic LocalSearch

Once `SearchState::apply` became what records a move into the tabu memory, the
operators stopped needing a shared object — and the question arose whether BLS
could simply drive `LocalSearch` instead of a specialised descent. It can, and
it now does. This is a deliberate trade of objective for one less specialised
implementation, and the price was measured before taking it.

**One descent to a local optimum takes 1.8–2.3× as long.** Ten seeds, medians,
from identical initial solutions on G1 / G22 / G32 / G55 / G70 / G81 — flat
across n from 800 to 20000 and average degree from 2 to 48, with move counts
agreeing to within 1%. Both arms select from the same candidate set
(`is_neighbor_better_than_current` on a flip is `gain > 0`, which is exactly
what `positive_gain` indexes), but `LocalSearch` rescans all n vertices per
move where the specialised descent enumerated only the improving ones.
Decomposed, the scan is **87%** of the gap and `update_best`'s per-move
solution clone the other **13%**.

**Through a whole BLS run that is −40.0 total average cut**, at 30s × 5 runs,
seed 42, over the ten-instance G-set panel G1 / G11 / G22 / G32 / G43 / G55 /
G60 / G63 / G70 / G81 with `l0 = 0.01|V|` and the density-scaled tenure: better
on 3, worse on 5, −21.2 on G81, −12.4 on G63, −8.4 on G70, −6.4 on G55, against
+8.2 on G60. Most sit inside one or two standard deviations; the throughput
behind them does not — 0.74–0.93× as many moves, on every one of the ten. Two
further differences ride along and are not separated by that number: `max_by`
breaks gain ties toward the last candidate where `ops::keep_best` kept the
first, and `LocalSearch` spends one extra `progress_iteration` per descent.

**The 2× is the generic contract's price, not slack in it.** Three
generality-preserving attempts were measured and none of them paid:
`clone_from` in `update_best` changed nothing (`#[derive(Clone)]` does not
specialise it, so it falls back to `*self = source.clone()`); folding the best
update to the end of the descent — sound, because `LocalSearch` only applies
improving moves, so the last solution is the best one — recovers just 8–10% and
coarsens the anytime trajectory to one point per descent; and the identical
traversal written out as a hand loop ran **3.3× slower** than the iterator
chain. Cutting the scan itself means knowing which gains the last move changed,
which is problem knowledge, so it needs a hook on `MoveToNeighbor` that only
MaxCut and QUBO could implement.

Two operators stayed behind. `random_flips` is
free to replace (by the same measurement, `RandomWalk` moved the total by +0.4
with throughput unchanged) but was kept: it deletes no file, since `best_swap`
holds `perturbation.rs` open regardless, and `RandomWalk` fails with
`InvalidState` on the edgeless sub-instances `SubProblemBasedCrossover`
produces, so its guard would only move into the caller.

## Why `best_swap` is not a move type

The obvious next step is to give the directed swap the same treatment: define a
move type whose neighborhood is small enough for a generic `TabuSearch`, and
delete the operator. `TabuSearch<MaxCutSwapNeighbor>` is out because its `iter`
enumerates every cross-side pair, O(n²) — 4·10⁸ against 2·10⁴ per step on G81 —
but that is a property of *that* neighborhood, not of swaps. A
`MaxCutBestSwapNeighbor` yielding only the pairs that touch each side's
highest-gain vertex is O(n), always contains the greedy pair, and needs no new
parameter.

**It was built and measured, and it fails.** Against the integrated descent as
the baseline, on the same ten-instance panel at 30s × 5 runs: **−452.6 total
average cut**, worse on 8 of 10, at **0.05–0.36× the moves**, with `TabuSearch`
reporting no eligible move 3.8× more often than it found one (5.6M times on G1
against 1.5M applied moves).

The cause is structural. `apply_swap_as_two_flips` inverts the sign of both
moved vertices' gains, and the weak swap runs straight after a descent, from a
local optimum where every gain is ≤ 0. So the two vertices a swap just moved
become the highest-gain vertex of their new side — and they are tabu, because
the swap just recorded them. On the next step both centres are forbidden, every
candidate has a forbidden endpoint, and the neighborhood empties. **Ranking a
restricted neighborhood by gain is anti-correlated with a recency-based tabu
list.** Widening to the top `k` per side does not escape it: `k` would have to
exceed the tenure — up to 600 on the sparse instances — and `k²` then overtakes
the `n` it was meant to replace.

What `best_swap` does instead is consult the tabu memory *during* the scan, for
the best **non-tabu** vertex on each side. That is what `MoveToNeighbor::iter`
cannot do: it is handed the problem and the solution, never the search state.
The operator stays.

Integrating the descent also required dropping `LocalSearch::new`'s rewrite of
an unset `max_failed_update` to `Some(1)`. `is_done` reads that field as
`iteration - best_iteration >= 1`, so a descent starting from a solution below
the incumbent best — which is where every kick leaves it — satisfied it before
taking a single move. No meta-heuristic ever saw this, because they all hand
their sub-run a `ClearBest` clone that re-anchors `best_iteration`; BLS
descends on the state directly, and did.

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

`clear()` resets the schedule (`omega` to 0, `l` to `l0`, the remembered local
optimum dropped). The prohibitions are not its to clear: they belong to the
`SearchState`, and a sub-run clone — how every meta-heuristic starts a phase —
already arrives with an empty tabu memory (`state.reset_tabu()` for a state you
are reusing directly).
The remembered local optimum is dropped rather than kept because the same
schedule may be reused on a different instance — a meta-heuristic that rebuilds
its sub-problem every round does exactly that.

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
max_duration_secs = 30.0
```

`tabu_tenure` is read as original `γ`: a vertex stays forbidden for `2γ`
moves. This is the one kind that doubles the key — the same range under
[`TabuSearch`](tabu_search.md) prohibits for half as long, so tuned values do
not transfer between them.

## Driving it yourself

The round is also reachable in halves, for a caller that wants to keep the
descent and the operators but replace the schedule:

- `descend(state)` — the greedy descent, writing the prohibitions the kick reads.
- `kick(state, perturbation, l)` — one [`MaxCutPerturbation`](../api/optopus/heuristic/enum.MaxCutPerturbation.html)
  of length `l`, followed by the round's single `update_best`.
- `externally_driven(stop_condition, tabu_tenure)` — the constructor for that
  use, taking the tenure **literally**: the `2γ` doubling above belongs to the
  paper's schedule, which such a caller replaces.

`run_once` is exactly `descend` + the schedule + `kick`.
[Driving BLS with a learned perturbation policy](../guide/learned_perturbation.md)
walks through a controller that decides between the two halves with a
contextual bandit.

## References

- Benlic, U. and Hao, J.-K. "Breakout Local Search for the Max-Cut problem."
  *Engineering Applications of Artificial Intelligence*, 26(3), 1162-1173,
  2013.
