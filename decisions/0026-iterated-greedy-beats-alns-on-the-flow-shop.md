# Iterated Greedy is its own heuristic, and on the flow shop it beats ALNS on the same Ruinable

- Status: reference
- Area: heuristic, flow_shop
- Date: 2026-10-07
- Code: src/heuristic/iterated_greedy.rs, src/heuristic/specific/flow_shop.rs, src/problem/flow_shop/ruin.rs

## Decision

Iterated Greedy (Ruiz and Stützle 2007) was added as `IteratedGreedy<P: Ruinable>`
rather than as ALNS settings, since ALNS cannot express it. The removal count is a
fraction, the roulette cannot be pinned to random removal and greedy repair,
and the temperature cools from a value derived from the first solution. The
flow shop `Partial` keeps Taillard's heads and tails, so the trait's per-place
`insertion_cost` is O(m) and a scan over all places is O(nm), which avoids the
granularity cost 0015 describes.

The temperature is a function of the instance, resolved on the first iteration
of a run, because `build_special_heuristic` has no instance to read
`Σp / (n·m·10)` from.

## Measurement

Taillard ta051 to ta060 (50 × 20), 30 s by ten runs, seed 42,
`data/benchmarks/flow_shop/taillard_50x20.toml` as committed, nothing else on
the machine.

| | Iterated Greedy | ALNS | TabuSearch (Relocate, [5, 15]) |
|---|---|---|---|
| Mean of the instance averages | 3726.9 | 3733.3 | 3736.1 |
| Instances where it has the best average | 10 of 10 | 0 | 0 |

Both ruin and recreate arms run the same `FlowShopInsertionDescent`. The best
known values of this block average about 3708, which puts Iterated Greedy about
0.5% above them. A 10 s by four runs pilot gave the same order, 3732.5, 3739.6
and 3754.2.

## Do not retry

Do not fold Iterated Greedy back into ALNS as a preset. The difference that
pays is the fixed operator pair with the full descent, and ALNS would need
three new knobs to reproduce it.

Do not move the heads and tails out of the partial into a per-call
recomputation. Each `insertion_cost` would become O(nm) and the scan O(n²m).
