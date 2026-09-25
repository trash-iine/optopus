# ReinforcementLearningSearch

**API:** [`ReinforcementLearningSearch`](../../api/optopus/heuristic/reinforcement_learning/struct.ReinforcementLearningSearch.html)

move の特徴量に対するオンラインの強化学習です。線形の方策が候補の move をスコア付けし、その softmax から一つを引き、
得られた結果から方策を更新します。

## 例 { #example }

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 2.0)]));
let mut state = SearchState::new(&mc);

let mut rl = ReinforcementLearningSearch::<MaxCutFlipNeighbor>::new(
    StopCondition::iterations(10_000),
    /* learning_rate       = */ 0.01,
    /* softmax_temperature = */ 1.0,
    RewardShaping::Normalized,
    /* max_candidates      = */ Some(64),
);
rl.run(&mut state)?;
println!("cut weight = {}", state.best_solution.objective);
```

一回の `run` が一つのエピソードです。エピソードをまたいで引き継がれるのは方策です。
[複数エピソードにわたる学習](#multi-episode-learning) を参照してください。

## アルゴリズムの概要 { #algorithm-sketch }

各ステップで次を行います。

1. 近傍のすべての move (またはその部分標本) を列挙し、それぞれの悪化量を計算します。
2. `NUM_FEATURES` (= 21) 個の交互作用特徴量に対する線形の方策で各 move をスコア付けします。
   特徴量は 3 個の move レベルの特徴 (正規化した gain、改善するかどうか、おおよその順位) のそれぞれ単独のものと、
   それに 6 個の状態レベルの特徴 (進捗、停滞、改善率、近傍の統計) を掛けたものです。これにより探索の状態が move の好みを調整します。
3. 得られた softmax 分布から move を一つ引きます。
4. move を適用し、ベースラインを引いた1ステップの REINFORCE で方策を更新します。

## コンストラクタ { #constructor }

```rust
ReinforcementLearningSearch::<N>::new(
    stop_condition: StopCondition,
    learning_rate: f64,
    softmax_temperature: f64,
    reward_shaping: RewardShaping,
    max_candidates: Option<usize>,
) -> Self
```

`N` は `MoveToNeighbor<P> + Evaluate + Clone` を満たす必要があります。

`max_candidates` を設定すると、評価する前に遅延的な近傍イテレータからこの数だけの move をリザーバサンプリングします。
そのためステップごとの評価と特徴量のコストは `O(近傍)` ではなく `O(max_candidates)` になります。
ステップの統計 (したがって近傍レベルの特徴量) は標本だけから計算されます。

特徴量の交互作用を入れる前に学習した `policy_weights` のファイル (9 要素) は拒否されます。現在の 21 要素の配置で学習し直してください。
以前の `discount` パラメータは削除されました。1ステップの REINFORCE には割引率がないからです (TOML のキーはまだ受け付けますが、警告を出して無視します)。

`with_policy_weights([f64; NUM_FEATURES])` を使うと、学習済みの重みで方策を初期化できます。

## ベンチマーク設定 { #benchmark-config }

```toml
[[heuristics]]
kind = "ReinforcementLearningSearch"
neighbor = "Flip"            # 必須。使える値は問題ごとに決まる
learning_rate = 0.01         # 任意 (値は既定値)。0.0 = 評価モード
softmax_temperature = 1.0    # 任意 (値は既定値)
reward_shaping = "Normalized"  # 任意。Raw | Normalized | BestImprovement
max_candidates = 200         # 任意。未設定なら近傍全体を評価する
policy_weights = []          # 任意。21 要素、下を参照
[heuristics.stop_condition]
max_duration_secs = 30.0
```

`discount` は設定の互換性のためにまだ受け付けますが、構築時に警告を出して無視します。

## 報酬の整形 { #reward-shaping }

```rust
pub enum RewardShaping {
    Raw,                // -worsening
    Normalized,         // -worsening / そのステップの max |worsening|
    BestImprovement,    // 新しい最良解が見つかれば 1.0、そうでなければ 0.0
}
```

## 複数エピソードにわたる学習 { #multi-episode-learning }

`clear()` はエピソードごとの状態をリセットしますが、`policy.weights` と移動平均のベースラインは保持します。
多くのエピソードにわたって学習させるには、`ReinforcementLearningSearch` を [`Restart`](meta.md#restart) や
[`Iterated`](meta.md#iterated) で包みます。

```rust
use optopus::prelude::*;

let rl = ReinforcementLearningSearch::<MaxCutFlipNeighbor>::new(
    StopCondition::failed_updates(1_000),
    0.01, 1.0,
    RewardShaping::Normalized,
    Some(64),
);
let mut solver = Restart::new(
    StopCondition::iterations(1_000_000),
    Box::new(rl),
    StopCondition::failed_updates(10_000),
);
```

## 参考文献 { #references }

- Williams, R. J. "Simple Statistical Gradient-Following Algorithms for
  Connectionist Reinforcement Learning." Machine Learning, 8(3-4), 229-256,
  1992.
