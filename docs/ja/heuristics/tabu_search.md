# TabuSearch

**API:** [`TabuSearch`](../../api/optopus/heuristic/struct.TabuSearch.html)

各ステップで、現在タブーでない move のうち厳密に最良のものを選び、それを `tabu_tenure = (min, max)` から一様に引いたテニュアの間タブーにします。

タブーの move でも、aspiration 基準を満たせば選べます。適用した結果の解が現在の全体の最良解より厳密に良くなる場合です。

## 例 { #example }

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]));
let mut state = SearchState::new(&mc);
let mut ts = TabuSearch::<MaxCutFlipNeighbor>::new(
    StopCondition::iterations(10_000),
    /* tabu_tenure = */ (5, 10),
);
ts.run(&mut state)?;
println!("cut weight = {}", state.best_solution.objective);
```

## コンストラクタ { #constructor }

```rust
TabuSearch::<N>::new(
    stop_condition: StopCondition,
    tabu_tenure: (u64, u64),
) -> Self
```

`N` は `MoveToNeighbor<P> + Clone + EnabledTabu + Rankable` を満たす必要があります。

`tabu_tenure.0 > tabu_tenure.1` なら panic します。

## タブーマップの置き場所 { #where-the-tabu-map-lives }

マップはこのヒューリスティクスではなく [`SearchState`](../search_state.md) にあります。
move を適用するのは状態なので、それを記録するのも状態です。`apply` / `apply_move_only` は反復が進む前に move をタブーメモリに書き込みます。
タブーを有効にするには、最初に `start_record_tabu(tenure)` を呼ぶ必要があります。例は `TabuSearch` の実装を参照してください。

## タブー方策の抽象化 { #tabu-policy-abstraction }

各近傍の型は、どのキーが空いていなければならないか、そして move を適用すると何が禁止されるかというタブー方策を `EnabledTabu` トレイトで自分で持ち、
`MoveToNeighbor::tabu_policy` を `Some(self)` でオーバーライドする1行で状態に渡します。
`TabuSearch` は何がキーになっているかを知りません。これにより QUBO、MaxCut、SAT は変数のインデックスを、TSP は辺の組を、
Job Shop は swap の位置を、といった具合にキーにできます。二つが一致している必要はありません。
VRP の relocate は、ある顧客が移動先のルートに入ってよいかを問い合わせ、出てきたルートを禁止します。

`tabu_policy` を既定の `None` のままにした move にはタブー方策がまったくなく、それを使って `state.record_tabu` を呼ぶことはできません。
`EnabledTabu` を実装したのに1行のオーバーライドを忘れた move 型は、ここでタブーリストなしに何も言わずに動いてしまいます。
そのため `trait_defs/tabu.rs` はすべての組み込み move についてまさにその点を検査しています。

`run_once` は反復ごとに一度 `state.start_record_tabu(tenure)` を呼びます。テニュアとモードは同じ呼び出しで決まります。
記録は新しい状態でもすべてのサブランでもオフなので、タブーリストを手法の中心とする探索はそのことを明示しなければなりません。
それも、どこか別の場所で一度だけではなく、それに依存するループのすぐそばで明示します。

`common::TabuMemory` は唯一のストアで、`TabuKey` の形ごとに分かれています。密なインデックスには `Var(i)`、それ以外には `Pair` と `Triple` です。
同じ形を使う二つの move 型は禁止を共有します (MaxCut の flip と swap はどちらも `Var` で、Breakout Local Search
がこの探索を weak-flip の摂動として動かすときにはこれを当てにしています)。形が違えば衝突することはありません
(JobShop の swap は `Var`、relocate は `Pair` です)。

## ベンチマーク設定 { #benchmark-config }

```toml
[[heuristics]]
kind = "TabuSearch"
neighbor = "Flip"        # 必須。使える値は問題ごとに決まる
tabu_tenure = [5, 10]    # 必須。(min, max) で、move ごとに一様に引く
[heuristics.stop_condition]
max_duration_secs = 30.0
```

テニュアは文字どおりに解釈されます。move はその反復回数だけ禁止されたままで、このキーはそれを受け取るどの kind でも同じ意味です。

## 参考文献 { #references }

- Glover, F. "Future Paths for Integer Programming and Links to Artificial
  Intelligence." Computers & Operations Research, 13(5), 533-549, 1986.
- Glover, F. "Tabu Search, Part I." ORSA Journal on Computing, 1(3),
  190-206, 1989.
