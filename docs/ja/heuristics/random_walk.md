# RandomWalk

**API:** [`RandomWalk`](../../api/optopus/heuristic/struct.RandomWalk.html)

一様ランダムに近傍を一つ引き、無条件に適用します。受理判定も比較もありません。
それでも、歩く途中で出会った最良解は `state.best_solution` に記録されます。

## 例 { #example }

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]));
let mut state = SearchState::new(&mc);
let mut rw = RandomWalk::<MaxCutFlipNeighbor>::new(StopCondition::iterations(10));
rw.run(&mut state)?;
println!("cut weight = {}", state.best_solution.objective);
```

## アルゴリズムの概要 { #algorithm-sketch }

各 `run_once` で次を行います。

1. 一様ランダムに近傍を一つ引きます。
2. それを無条件に適用します。歩く途中で出会った最良解は、state が記録し続けます。
3. 近傍が空なら、反復のカウンタだけを進めます。空の近傍は失敗ではなく、歩行に渡されうる状態の一つです。
   カウンタを進めることで、外側の予算で歩行を終えられるようになります。

## コンストラクタ { #constructor }

```rust
RandomWalk::<N>::new(stop_condition: StopCondition) -> Self
```

`stop_condition` は実行を終える条件です。[停止条件](../guide/stop_conditions.md) を参照してください。

`N` は `MoveToNeighbor<P> + Rankable` を満たす必要があります。

## 振る舞い { #behavior }

`RandomWalk` 単体で役に立つことはまれです。主な役割は [`Iterated`](meta.md#iterated) の摂動フェーズです。
いくつかのランダムな move で探索を局所最適から押し出し、次の貪欲なフェーズが別の谷を登れるようにします。

## ベンチマーク設定 { #benchmark-config }

```toml
[[heuristics]]
kind = "RandomWalk"
neighbor = "Flip"        # 必須。使える値は問題ごとに決まる
[heuristics.stop_condition]
max_iteration = 200      # 必ず与える。ランダムウォークは自分では止まらない
```

空の `stop_condition` では終了しません。これが最も効いてくるのは、`RandomWalk` の普段の使い方である入れ子の摂動ステップです。
[ILS の例](../guide/benchmarking.md#nested-example-ils-in-toml) を参照してください。
