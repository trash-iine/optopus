# RLSearch

Online reinforcement learning over move features. At each step:

1. Enumerate all (or a subsample of) neighborhood moves and compute their
   worsening amounts.
2. Score each move with a linear policy over `NUM_FEATURES` hand-crafted
   features (move worsening, approximate rank, step context).
3. Sample one move from the resulting softmax distribution.
4. Apply the move, then update the policy via single-step REINFORCE with
   baseline subtraction.

## Constructor

```rust
RLSearch::<N>::new(
    stop_condition: StopCondition,
    learning_rate: f64,
    discount: f64,
    softmax_temperature: f64,
    reward_shaping: RewardShaping,
    max_candidates: Option<usize>,
) -> Self
```

`N` must satisfy `MoveToNeighbor<P> + Evaluate + Clone`.

`max_candidates`: if set, randomly subsamples the neighborhood down to this
many moves per step (useful for large `O(n²)` neighborhoods).

`with_policy_weights([f64; NUM_FEATURES])` lets you seed the policy with
pre-trained weights.

## Reward shaping

```rust
pub enum RewardShaping {
    Raw,                // -worsening
    Normalized,         // -worsening / step's max |worsening|
    BestImprovement,    // 1.0 if a new best was found, else 0.0
}
```

## Multi-episode learning

`clear()` resets per-episode state but **preserves `policy.weights` and the
running baseline**. Wrap `RLSearch` in [`Restart`](meta.md#restart) or
[`Iterated`](meta.md#iterated) to train across many episodes:

```rust
use optopus::prelude::*;

let rl = RLSearch::<MaxCutFlipNeighbor>::new(
    StopCondition::failed_updates(1_000),
    0.01, 0.99, 1.0,
    RewardShaping::Normalized,
    None,
);
let mut solver = Restart::new(
    StopCondition::iterations(1_000_000),
    Box::new(rl),
    StopCondition::failed_updates(10_000),
);
```

## References

- Williams, R. J. "Simple Statistical Gradient-Following Algorithms for
  Connectionist Reinforcement Learning." *Machine Learning*, 8(3-4), 229-256,
  1992.
