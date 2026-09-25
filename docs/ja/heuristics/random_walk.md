# RandomWalk

**API:** [`RandomWalk`](../../api/optopus/heuristic/struct.RandomWalk.html)

一様ランダムに近傍を一つ引き、無条件に適用します。受理判定も比較もありません。
それでも、歩く途中で出会った最良解は `state.best_solution` に記録されます。

move が見つからない反復は、カウンタを進めて戻ります。空の近傍は失敗ではなく、歩行に渡されうる状態の一つです。
カウンタを進めることで、外側の予算で終了できるようになります。

## 例 { #example }

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]));
let mut state = SearchState::new(&mc);
let mut rw = RandomWalk::<MaxCutFlipNeighbor>::new(StopCondition::iterations(10));
rw.run(&mut state)?;
println!("cut weight = {}", state.best_solution.objective);
```

## コンストラクタ { #constructor }

```rust
RandomWalk::<N>::new(stop_condition: StopCondition) -> Self
```

`N` は `MoveToNeighbor<P> + Rankable` を満たす必要があります。

## 使いどころ { #when-to-use }

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
