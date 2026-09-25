# LateAcceptanceHillClimbing

**API:** [`LateAcceptanceHillClimbing`](../../api/optopus/heuristic/struct.LateAcceptanceHillClimbing.html)

move を適用した結果のスコアが、`history_length` 反復前に記録したスコアより悪くなければ受理します。
この遅れたスコアが適応的なしきい値になり、調整すべき温度なしに探索が局所最適から抜け出せるようにします。

## 例 { #example }

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]));
let mut state = SearchState::new(&mc);
let mut lahc = LateAcceptanceHillClimbing::<MaxCutFlipNeighbor>::new(
    StopCondition::iterations(100_000),
    5_000,
);
lahc.run(&mut state)?;
println!("cut weight = {}", state.best_solution.objective);
```

## アルゴリズムの概要 { #algorithm-sketch }

各 `run_once` で次を行います。

1. 一様ランダムに近傍を一つ引きます。
2. それをスコア付けします。実行中のスコアは常に大きいほど良い値です (最小化では `- score` を使います)。
   そのため候補は `current − minimized()` で、もとの目的関数の向きは `Evaluate` が扱います。
3. 候補が現在のスコアより悪くないか、`history_length` ステップ前のスコアである `history[i mod history_length]`
   より悪くなければ受理します。棄却した move は反復カウンタを進めるだけです。
4. (変わらなかったかもしれない) 現在のスコアを同じスロットに記録し、`i` を進めます。

実行中のスコアも履歴のバッファも `0.0` から始まるので、バッファが持つのは目的関数値そのものではなく、初期解に対する各ステップの相対的なスコアです。
すべてのエントリに同じオフセットがかかっているので、ステップ 3 の比較には影響しません。

## コンストラクタ { #constructor }

```rust
LateAcceptanceHillClimbing::<N>::new(
    stop_condition: StopCondition,
    history_length: usize,
) -> Self
```

`N` は `MoveToNeighbor<P> + Evaluate` を満たす必要があります。

`history_length == 0` なら panic します。

`history_length` は活用と探索のトレードオフを決めます。

| `history_length` | 振る舞い |
|---|---|
| `1` | ほぼ山登り (1 ステップ前より悪くない move だけを受理)。 |
| `5_000` | 多くの問題で妥当な既定値。 |
| もっと大きい値 | 多様化が強まり、収束は遅くなる。 |

`clear()` は履歴のバッファを空にします。バッファは `run` の後の最初の `run_once` で初期化し直されます。

## ベンチマーク設定 { #benchmark-config }

```toml
[[heuristics]]
kind = "LateAcceptanceHillClimbing"
neighbor = "Flip"        # 必須。使える値は問題ごとに決まる
history_length = 5_000   # 必須。1 以上
[heuristics.stop_condition]
max_iteration = 100_000
```

## 参考文献 { #references }

- Burke, E. K. and Bykov, Y. "The Late Acceptance Hill-Climbing Heuristic."
  European Journal of Operational Research, 258(1), 70-78, 2017.
