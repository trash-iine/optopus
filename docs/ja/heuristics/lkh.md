# LinKernighanHelsgaunForTsp

**API:** [`LinKernighanHelsgaunForTsp`](../../api/optopus/heuristic/struct.LinKernighanHelsgaunForTsp.html)

[TSP](../problems/tsp.md) 専用のヒューリスティクスです。各都市から始めて、可変深さの辺交換探索 (最大 k-opt) を行います。

## 例 { #example }

```rust
use optopus::prelude::*;

let tsp = Tsp::new(
    "demo".to_string(),
    vec![
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.5),
        (2.0, 1.5),
        (1.0, 2.0),
        (0.0, 1.0),
    ],
);
let mut state = SearchState::new(&tsp);

let mut lkh = LinKernighanHelsgaunForTsp::new(
    StopCondition::iterations(10_000),
    /* num_neighbors = */ 5,
    /* max_depth     = */ 5,
);
lkh.run(&mut state)?;

let sol = &state.best_solution;
println!("tour length = {}", sol.objective);
println!("visiting order = {:?}", sol.tour);
```

`LocalSearch` と同じく局所最適で止まるので、一回の降下より大きな予算が生きるのは [`Restart` や `Iterated`](meta.md) の中だけです。

## アルゴリズムの概要 { #algorithm-sketch }

各開始都市について、アルゴリズムは辺交換の連鎖を伸ばしていきます。

1. 現在の連鎖の端点に近い候補都市を選びます。
2. move を閉じてみて、巡回路が短くなればそれを適用します。
3. そうでなければ連鎖をより深く伸ばします。

枝刈りは次のとおりです。

- 候補リスト。各端点で最も近い `num_neighbors` 個の都市だけを考えます。
- 正の gain の基準。部分的な gain はどのステップでも正のままでなければなりません。
- 最大深さ。`max_depth` 段 (k-opt の k) で探索を打ち切ります。

最初に見つかった改善 move を適用します。どの開始都市からも改善 move が見つからなくなるか、停止条件が発火すると探索は終わります。

## コンストラクタ { #constructor }

```rust
LinKernighanHelsgaunForTsp::new(
    stop_condition: StopCondition,
    num_neighbors: usize,
    max_depth: usize,
) -> Self
```

既定値は `num_neighbors = 5`、`max_depth = 5` です。

`clear()` は `is_done` が読む局所最適のフラグを下ろすので、二回目の `run` はすぐに終了を報告せずもう一度探索します。
インスタンスから導いた候補リストはほかの何にも依存しないので、`clear()` の後も残ります。

## ベンチマーク設定 { #benchmark-config }

```toml
[[heuristics]]
kind = "LinKernighanHelsgaun"
num_neighbors = 5        # 任意 (値は既定値)
max_depth = 5            # 任意 (値は既定値)
[heuristics.stop_condition]
max_duration_secs = 30.0
```

自前の move 集合を持つので `neighbor` は取りません。局所最適で止まるので、長い予算が生きるのは
[`Restart` や `Iterated`](meta.md) の中だけです。

## 参考文献 { #references }

- Lin, S. and Kernighan, B. W. "An Effective Heuristic Algorithm for the
  Traveling-Salesman Problem." Operations Research, 21(2), 498-516, 1973.
- Helsgaun, K. "An Effective Implementation of the Lin-Kernighan Traveling
  Salesman Heuristic." European Journal of Operational Research, 126(1),
  106-130, 2000.
