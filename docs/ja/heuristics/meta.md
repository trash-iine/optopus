# Sequential / Iterated / VariableNeighborhoodSearch / Restart

**API:** [`Sequential`](../../api/optopus/heuristic/struct.Sequential.html)

サブランのクローンとマージのパターンでほかのヒューリスティクスを組み合わせる、四つのメタヒューリスティクスです。

```rust
let mut sub = state.clone_for_new_run(SearchStateCloneType::ClearBest);
inner.run(&mut sub)?;
state.update_state(sub);   // 最良解を戻し、反復回数を加算する
```

全体の反復カウンタはすべてのフェーズを通して単調に進みます。トレイトの詳細と三つの `SearchStateCloneType` の種類は
[基本概念](../concepts.md#sub-run-clonemerge-pattern) と
[SearchState API](../search_state.md#searchstateclonetype-variants) にあります。

どれを使うかの目安は次のとおりです。

- `Sequential`。決まった順番に並べたフェーズのパイプラインです (たとえばランダムな開始点を `LocalSearch` で整えてから `TabuSearch`)。
- `Iterated`。探索と摂動を交互に行って局所最適から抜け出します (ILS)。一つの探索が停滞するときの標準的な選択です。
- `VariableNeighborhoodSearch`。同じ考え方で、強さの異なる複数の shake 近傍を持ち、今の近傍が失敗したときだけ強めていきます。
- `Restart`。進歩が止まったら現在の解を捨ててランダムな解を引き直します。全体の最良解は残ります。

一つの現在解ではなく集団を使うなら [GeneticAlgorithm](genetic_algorithm.md) を参照してください。

## 例 { #example }

普通の入口は `Iterated` (ILS) です。停滞する探索フェーズと、停滞した谷からそれを押し出す摂動フェーズからなります。

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]));
let mut state = SearchState::new(&mc);

let mut ils = Iterated::<MaxCut>::new(
    StopCondition::iterations(100_000),
    /* search       = */ Box::new(TabuSearch::<MaxCutFlipNeighbor>::new(
        StopCondition::failed_updates(500),
        (5, 10),
    )),
    /* perturbation = */ Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(
        StopCondition::iterations(5),
    )),
);
ils.run(&mut state)?;
println!("cut weight = {}", state.best_solution.objective);
```

以下の四つの節には、それぞれのコンストラクタと例があります。

どのヒューリスティクスも `StopCondition` で止まります。
反復回数、時間、停滞で止める `StopCondition::iterations`、`duration`、`failed_updates` と、それらの組み合わせ方は [停止条件](../guide/stop_conditions.md) にまとめてあります。

## Sequential

ヒューリスティクスのリストを順番に実行します。それぞれは新しい `ClearBest` のクローン上で動き、ステップの間で結果がマージされます。

```rust
Sequential::<P>::new(
    stop_condition: StopCondition,
    heuristics: Vec<Box<dyn Heuristic<P>>>,
) -> Self

// または少しずつ組み立てる:
seq.push_heuristic(Box::new(...));
```

外側の `stop_condition` はサブヒューリスティクスの間で調べられ、内側のヒューリスティクスはそれぞれ自分の停止条件を持ちます。
リストの最後まで来ると、先頭からもう一周します。

```rust
let mut seq = Sequential::<MaxCut>::new(
    StopCondition::iterations(100_000),
    vec![
        Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
            StopCondition::failed_updates(1),
        )),
        Box::new(TabuSearch::<MaxCutFlipNeighbor>::new(
            StopCondition::failed_updates(500),
            (5, 10),
        )),
    ],
);
seq.run(&mut state)?;
```

## Iterated

[`Iterated`](../../api/optopus/heuristic/struct.Iterated.html) は Iterated Local Search (ILS) のパターンです。
`search` フェーズと `perturbation` フェーズを交互に行います。

```rust
Iterated::<P>::new(
    stop_condition: StopCondition,
    search: Box<dyn Heuristic<P>>,
    perturbation: Box<dyn Heuristic<P>>,
) -> Self
```

サイクルは `search` → 外側の `stop_condition` を確認 → `perturbation` → 繰り返し、です。
どちらのフェーズも `ClearBest` のクローン上で動き、全体の最良解は残ります。

典型的な組み合わせは `search = LocalSearch`、`perturbation` を数反復の `RandomWalk` にするものです。

```rust
let mut ils = Iterated::<MaxCut>::new(
    StopCondition::iterations(100_000),
    Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
        StopCondition::failed_updates(1),
    )),
    Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(
        StopCondition::iterations(5),
    )),
);
ils.run(&mut state)?;
```

## VariableNeighborhoodSearch

[`VariableNeighborhoodSearch`](../../api/optopus/heuristic/struct.VariableNeighborhoodSearch.html)
は基本的な Variable Neighborhood Search (VNS) です。順序付きの shake ヒューリスティクスのリスト `N_1..N_kmax`
(典型的には予算を増やしていく `RandomWalk`) と局所的な `search` を持ちます。

```rust
VariableNeighborhoodSearch::<P>::new(
    stop_condition: StopCondition,
    search: Box<dyn Heuristic<P>>,
    shakes: Vec<Box<dyn Heuristic<P>>>,   // 空であってはならない
) -> Self
```

サイクルは次のとおりです。現在解のスナップショットを取り、`N_k` で shake し、`search` を行います。
結果が現在解を改善すればそれを採用して `k` をリセットし、そうでなければ現在解を戻して `k` を進めます (最後の近傍の次は最初に戻ります)。
どちらの場合も全体の最良解は残ります。

```rust
let mut vns = VariableNeighborhoodSearch::<MaxCut>::new(
    StopCondition::iterations(100_000),
    Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
        StopCondition::failed_updates(1),
    )),
    vec![
        Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(
            StopCondition::iterations(5),
        )),
        Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(
            StopCondition::iterations(20),
        )),
        Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(
            StopCondition::iterations(50),
        )),
    ],
);
vns.run(&mut state)?;
```

## Restart

[`Restart`](../../api/optopus/heuristic/struct.Restart.html) は内側のヒューリスティクスを実行し、
`restart_condition` (典型的には `max_failed_update`) を満たすたびに `state.solution` を新しいランダムな解に置き換えます。
`state.best_solution` はリスタートをまたいで保持されます。

```rust
Restart::<P>::new(
    stop_condition: StopCondition,
    heuristic: Box<dyn Heuristic<P>>,
    restart_condition: StopCondition,
) -> Self
```

内側には任意の `Heuristic<P>` を入れられるので、よくある形は `Iterated` を `Restart` で囲むものです。

```rust
let ils = Iterated::<MaxCut>::new(
    StopCondition::iterations(10_000),
    Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(StopCondition::failed_updates(1))),
    Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(StopCondition::iterations(5))),
);

let mut solver = Restart::new(
    StopCondition::iterations(100_000),
    Box::new(ils),
    StopCondition::failed_updates(1_000),
);
solver.run(&mut state)?;
```

## 実行できる合成例 { #runnable-compositions }

どれも `examples/` にある完結したプログラムで、上のヒューリスティックから定番のアルゴリズムを組み立て、同じ予算で素の手法と比べます。
`cargo run --release --example` に名前を付けて実行します。

| 例 | アルゴリズム | 合成 |
|---|---|---|
| [`compose_iterated_local_search.rs`](https://github.com/trash-iine/optopus/blob/main/examples/compose_iterated_local_search.rs) | Iterated Local Search | `LocalSearch` と `RandomWalk` の `Iterated` |
| [`compose_iterated_tabu_search.rs`](https://github.com/trash-iine/optopus/blob/main/examples/compose_iterated_tabu_search.rs) | Iterated Tabu Search | `TabuSearch` と `RandomWalk` の `Iterated` |
| [`compose_multi_start.rs`](https://github.com/trash-iine/optopus/blob/main/examples/compose_multi_start.rs) | multi-start 降下 | 降下のたびに再出発する `LocalSearch` の `Restart` |
| [`compose_sa_reheating.rs`](https://github.com/trash-iine/optopus/blob/main/examples/compose_sa_reheating.rs) | 再加熱つき焼きなましと再出発つき焼きなまし | `SimulatedAnnealing` 一つの `Sequential` と、その `Restart` |
| [`compose_vns.rs`](https://github.com/trash-iine/optopus/blob/main/examples/compose_vns.rs) | Variable Neighborhood Search | 三段階の shake を持つ `VariableNeighborhoodSearch` |
| [`compose_vnd_pipeline.rs`](https://github.com/trash-iine/optopus/blob/main/examples/compose_vnd_pipeline.rs) | Variable Neighborhood Descent | 近傍の違う二つの `LocalSearch` の `Sequential` |
| [`compose_memetic.rs`](https://github.com/trash-iine/optopus/blob/main/examples/compose_memetic.rs) | memetic algorithm | mutation に `TabuSearch` を使う `GeneticAlgorithm` |
| [`compose_kernel_then_search.rs`](https://github.com/trash-iine/optopus/blob/main/examples/compose_kernel_then_search.rs) | 厳密縮約のあとの探索 | `MaxCutKernel` と、カーネル上の Breakout Local Search |
| [`compose_population_annealing_vs_sa.rs`](https://github.com/trash-iine/optopus/blob/main/examples/compose_population_annealing_vs_sa.rs) | population annealing と焼きなましの比較 | 同じ予算の `PopulationAnnealing` と `SimulatedAnnealing` |

## ベンチマーク設定 { #benchmark-config }

四つとも入れ子の `steps` 配列を持つ一つの `kind` で、違うのは各位置の意味だけです。

```toml
[[heuristics]]
kind = "Iterated"            # Sequential | Iterated | VariableNeighborhoodSearch | Restart
[heuristics.stop_condition]
max_duration_secs = 30.0

[[heuristics.steps]]         # steps[0]
kind = "LocalSearch"
neighbor = "Flip"
[heuristics.steps.stop_condition]
max_failed_update = 1

[[heuristics.steps]]         # steps[1]
kind = "RandomWalk"
neighbor = "Flip"
[heuristics.steps.stop_condition]
max_iteration = 200
```

| `kind` | `steps` | 追加のフィールド |
|---|---|---|
| `Sequential` | 順番に実行し、外側の停止条件まで繰り返す | |
| `Iterated` | `[0]` = search、`[1]` = perturbation | |
| `VariableNeighborhoodSearch` | `[0]` = search、`[1..]` = shake `N_1..N_kmax` | |
| `Restart` | `[0]` = 内側のヒューリスティクス | `restart_condition` (必須、`stop_condition` と同じ形) |

`Restart` は上のフィールドに加えて、独自の表を一つ持ちます。

```toml
[heuristics.restart_condition]     # 必須。いつランダムな解で初期化し直すか
max_failed_update = 1_000
```

steps はいくらでも深く入れ子にできます。`Iterated` を `Restart` で囲むには上の二つのブロックを互いの中に書き、
内側の `steps` を一段深い `[[heuristics.steps.steps]]` にします。完全な
[ILS の例](../guide/benchmarking.md#nested-example-ils-in-toml) を参照してください。

## 参考文献 { #references }

- Lourenco, H. R., Martin, O. C., and Stutzle, T. "Iterated Local Search."
  In Glover, F. and Kochenberger, G. A. (eds.), Handbook of Metaheuristics,
  pp. 320-353. Springer, 2003.
