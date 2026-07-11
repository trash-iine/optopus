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
tabu_tenure = [3, 80]          # same density-scaled values as BLS
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

## Results vs BLS (Gset, 30 s × 10 seeded runs)

See `docs/benchmarks/rl_breakout_local_search.md` for the measured comparison
against the tuned BLS baseline on G1 / G11 / G22 / G43.
