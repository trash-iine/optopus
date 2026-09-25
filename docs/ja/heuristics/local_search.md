# LocalSearch

**API:** [`LocalSearch`](../../api/optopus/heuristic/struct.LocalSearch.html)

貪欲な最良改善の山登りです。各ステップで近傍のすべての move を評価し、厳密に最良のものを適用し、改善する move がなくなった時点 (局所最適) で止まります。

## 例 { #example }

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]));
let mut state = SearchState::new(&mc);
let mut ls = LocalSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(1_000));
ls.run(&mut state)?;
println!("cut weight = {}", state.best_solution.objective);
```

## アルゴリズムの概要 { #algorithm-sketch }

各 `run_once` で次を行います。

1. 遅延的な `N::iter` で近傍を列挙し、現在の解より厳密に良い move だけを残します。
2. その中から `rank_cmp` による `max_by` で最良のものを選びます。何も集めないので、一歩にメモリ確保のコストはかかりません。
   同点のときはイテレータが最後に出したものになります。これは恣意的ですが、山登りでは害はありません。
3. それを適用します。フィルタで何も残らなかったときは、`is_done` が読む局所最適のフラグを立て、反復カウンタを進めます。

## コンストラクタ { #constructor }

```rust
LocalSearch::<N>::new(stop_condition: StopCondition) -> Self
```

`N` は `MoveToNeighbor<P> + Rankable` を満たす必要があります。

`clear()` は `is_done` が読む局所最適のフラグを下ろします。そのため二回目の `run` はすぐに終了を報告せず、もう一度登ります。
これによって `LocalSearch` をメタヒューリスティクスの探索フェーズとして再利用できます。

## 振る舞い { #behavior }

`StopCondition` については次のとおりです。

- `max_failed_update` は `Some(1)` に強制されます。ほかの値を渡すと上書きされ、警告がログに出ます。
  局所最適での一歩は定義上、更新に失敗したステップだからです。
- 停止条件 (反復回数や時間) は引き続き有効です。一回の山登りより大きな予算を使うには、`LocalSearch` を `Restart` や `Iterated` と組み合わせてください。

## ベンチマーク設定 { #benchmark-config }

```toml
[[heuristics]]
kind = "LocalSearch"
neighbor = "Flip"        # 必須。使える値は問題ごとに決まる
[heuristics.stop_condition]
max_iteration = 100_000
```

設定に何が書かれていても `max_failed_update` は `1` に強制されます ([振る舞い](#behavior) を参照)。
そのためここで意味のある予算のキーは `max_iteration` と `max_duration_secs` で、それも一回の山登りを抑えるだけです。
本当の予算を与えるには、`LocalSearch` を [`Restart` や `Iterated`](meta.md) の中に入れてください。

## 参考文献 { #references }

- Aarts, E. and Lenstra, J. K. (eds.) *Local Search in Combinatorial
  Optimization*. Princeton University Press, 2003.
