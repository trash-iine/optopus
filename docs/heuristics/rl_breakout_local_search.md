# RlBreakoutLocalSearchForMaxCut

Problem-specific heuristic for [MaxCut](../problems/max_cut.md). Shares the
descent / perturbation machinery of
[BreakoutLocalSearchForMaxCut](breakout_local_search.md) (positive-gain-indexed
greedy descent, flat tabu map, weak-flip / weak-swap / strong perturbations),
but replaces the hand-crafted `omega`-based perturbation rule **and** the
strength schedule with a learned policy: a contextual softmax gradient bandit
(`optopus::heuristic::reinforcement_learning::bandit::SoftmaxBandit`).

## Algorithm sketch

Each outer iteration:

1. **Greedy descent** to a local optimum (identical to BLS).
2. **Reward observation** for the previous decision: change in local-optimum
   objective, normalized by an EMA of its own magnitude, clamped to `[−1, 1]`,
   plus a `+1` bonus when the global best improved. The bandit's per-action
   linear preferences are updated by one-step REINFORCE against an EMA
   baseline.
3. **Action selection**: from 7 context features —
   `[bias, min(ω/t, 1), exp(−ω/t), descent_improved_best, relative_gap,
   reward_ema, budget_progress]` — the bandit picks one of
   `3 × strength_bins.len()` actions: a perturbation type (weak flip / weak
   swap / strong) together with a strength multiplier of `l0`.
4. **Perturbation** with the selected operator and strength.

`exp(−ω/t)` is exactly the probability BLS's hand-crafted rule thresholds on,
so a near-linear policy can imitate BLS quickly before improving on it.

## Constructor

```rust
RlBreakoutLocalSearch::new(
    stop_condition: StopCondition,
    tabu_tenure: (u64, u64),
    t: u64,                    // omega normalization period
    l0: u64,                   // base perturbation length
    strength_bins: Vec<f64>,   // multipliers of l0, e.g. [1.0, 2.0, 4.0]
    learning_rate: f64,        // 0.0 = frozen-policy evaluation
    softmax_temperature: f64,
    exploration: f64,          // ε-uniform floor in [0, 1]
) -> Self
```

`with_policy_weights(Vec<f64>)` (row-major `num_actions × 7`) warm-starts the
bandit; combine with `learning_rate = 0.0` for frozen-policy evaluation.
`policy_weights()` reads the learned weights back for a later warm start.

## Multi-episode learning

`clear()` resets the episode state (omega, tabu map, pending decision, reward
statistics) but **preserves the bandit weights and baseline**, so the policy
keeps improving across [`Restart`](meta.md#restart) /
[`Iterated`](meta.md#iterated) episodes.

## Benchmark config

```toml
[[heuristics]]
kind = "RlBreakoutLocalSearch"
tabu_tenure = [3, 80]          # density-scaled like BLS, but taken literally
                               # (BLS reads the same key as γ and forbids for 2γ)
t = 1000
l0 = 80
strength_bins = [1.0, 2.0, 4.0]  # optional (default shown)
learning_rate = 0.1              # optional (default 0.1)
softmax_temperature = 1.0        # optional (default 1.0)
exploration = 0.05               # optional (default 0.05)
# policy_weights = [...]         # optional warm start

[heuristics.stop_condition]
max_duration_secs = 30
```

## Measurement notes

The action space was five operators rather than three until 2026-08-08: two
objective-preserving *plateau* perturbations (flip a connected cluster / an
independent set of zero-gain vertices) were extra bandit actions, together with
a `plateau_width` context feature.

They were removed, and the A/B says that costs objective. Same config
(`tabu_tenure = [15, 300]`, `t = 1000`, `l0 = 20`), 30 s × 5 seeded runs, mean
cut:

| instance | 5 operators | 3 operators | Δ |
|---|---|---|---|
| G55 (n=5000, seed 42) | 10200.4 | 10104.2 | **−96.2** |
| G55 (seed 7) | 10178.0 | 10115.4 | **−62.6** |
| G60 (n=7000) | 14024.2 | 13913.8 | **−110.4** |
| G63 (n=7000, deg 12) | 26750.2 | 26658.0 | **−92.2** |
| G70 (n=10000) | 9362.0 | 9367.8 | +5.8 |
| G11 (n=800) / G1 (n=800, dense) | 564.0 / 11624.0 | 564.0 / 11624.0 | 0 |

Run-to-run std is 9–28, so the mid-size losses are real. For reference, BLS
averages 10168.0 on G55 at the same budget: with the plateau operators this
heuristic beat it, without them it does not.

The trade taken was objective for structure: three operators instead of five,
one vocabulary shared with BLS, and no "already seen" scratch set inside the
engine. The mechanism is worth restoring if this heuristic ever has to win
rather than to be simple. Bandit weights saved from that version
(`5 × strength_bins.len()` actions × 8 features) no longer load:
`policy_weights` is size-checked at parse time, so they fail loudly rather than
silently misaligning.

The plateau idea itself survives in
[PopulationAnnealingForMaxCut](population_annealing.md) as its non-local cluster
move, which has its own implementation.

