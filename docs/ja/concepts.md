# 基本概念

## 設計方針 { #design-philosophy }

Optopus は互いに独立した三つの関心事を分離しています。

- 問題。何を最適化するか (MaxCut, TSP, ...)。
- ヒューリスティクス。どう最適化するか (Local Search, SA, ...)。
- 探索状態。反復回数、時間計測、現在の解と最良解。

どのヒューリスティクスも探索状態の上で動き、その振る舞いが記録されます。そのため、同じ条件でヒューリスティクス同士を比較できます。

## 三つのユースケース { #three-use-cases }

1. 既存の問題と既存のヒューリスティクスを組み合わせる。`use optopus::prelude::*` だけで、
   MaxCut や TSP などに `LocalSearch` や `SimulatedAnnealing` などを数行で適用できます。
2. 既存のヒューリスティクスを新しい問題に適用する。三つのコアトレイト
   (問題に `ProblemTrait`、解と move に `Evaluate`、問題上の解の move に `MoveToNeighbor`)
   を実装すれば、`LocalSearch`、`RandomWalk`、`BeamSearch`、`SimulatedAnnealing`
   とすべてのメタヒューリスティクスがそのまま動きます。`Rankable` は自分で書くものではなく、
   `Evaluate` から自動的に付きます。残りはトレイトを一つ足すごとに一つずつ使えるようになります。
   Tabu Search には `EnabledTabu`、Genetic Algorithm には `Distance` と `Crossover` です。
   完全なシグネチャとヒューリスティクスごとの要件表は
   [コアトレイト](traits.md#core-trait-reference) にあります。
3. ヒューリスティクスを組み合わせてベンチマークする。`Sequential`、`Iterated`、
   `VariableNeighborhoodSearch`、`Restart`、`GeneticAlgorithm` でアルゴリズムを組み合わせられます。
   比較の内容を TOML に書いて CLI を実行すると、集計された統計が得られます。

## `SearchState`

`SearchState<'a, P>` はすべてのヒューリスティクスを通して受け渡される共有の作業領域です。
現在の解、全体の最良解、反復カウンタ、時間計測を持ちます。ヒューリスティクスは問題を直接調べることはなく、
`SearchState` を書き換えます。

構造体の全体、メソッド、クローン方式は
[SearchState API](search_state.md) を参照してください。

### サブランのクローンとマージのパターン { #sub-run-clonemerge-pattern }

どのメタヒューリスティクスも、一つのフェーズをクローンした状態の上で隔離して走らせ、
最良解だけを元に戻します。これにより、反復回数を単調に保ったまま
`Sequential`、`Iterated`、`VariableNeighborhoodSearch`、`Restart`、`GeneticAlgorithm`
を自由に組み合わせられます。

```rust
let mut sub = state.clone_for_new_run(SearchStateCloneType::ClearBest);
inner_heuristic.run(&mut sub)?;
state.update_state(sub);   // 最良解を戻し、反復回数を加算する
```

三つのクローン方式 (`Simple` / `ClearBest` / `StartBest`) の違いは
[SearchState API](search_state.md#searchstateclonetype-variants) の表にまとめてあります。
