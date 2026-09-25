# 独自の問題を定義する

**API:** [`ProblemTrait`](../../api/optopus/trait_defs/trait.ProblemTrait.html)

三つのトレイトを実装すると、局所探索系のヒューリスティクスとすべてのメタヒューリスティクスがあなたの問題で動きます。
残りのヒューリスティクスは、任意のトレイトを一つ足すごとに一つずつ使えるようになります。下の
[どのヒューリスティクスに何が必要か](#which-heuristic-needs-what) を参照してください。

実行できる完全な例は
[`examples/custom_problem.rs`](https://github.com/trash-iine/optopus/blob/main/examples/custom_problem.rs)
にあります (`cargo run --example custom_problem`)。

問題のすべての変数が決まった範囲の整数なら、代わりに [`IntegerProblem`](../problems/integer.md) を実装してください。
範囲と目的関数だけを求め、近傍は最初から用意されています。

## 必須のトレイト { #required-traits }

| トレイト | 実装先 | 必須のメソッド |
|---|---|---|
| [`Evaluate`](../traits.md#core-trait-reference) | `Solution` | `evaluate(&self) -> Evaluable<f64>` |
| [`ProblemTrait`](../traits.md#core-trait-reference) | 問題の構造体 | `type Solution`, `new_solution(rng) -> Solution` |
| [`MoveToNeighbor<P>`](../traits.md#core-trait-reference) | 近傍の型 | `iter`, `apply_to_solution`, `move_to_be_better_than` |
| [`Evaluate`](../traits.md#core-trait-reference) | 近傍の型 | `evaluate(&self) -> Evaluable<f64>`。上とは別の、二つ目の impl |

`Evaluate` は本当に二回実装します。解に対しては目的関数値を返し、問題が最大化か最小化かに応じて
`Evaluable::Maximize` または `Evaluable::Minimize` で包みます。move に対しては、その move を適用したときの変化量を同じように包んで返します。

どちらの impl からも [`Rankable`](../traits.md#core-trait-reference) が得られます。
`LocalSearch`、`RandomWalk`、`BeamSearch`、`TabuSearch` が move を選ぶのも、どのヒューリスティクスでも解が改善したかを判断するのもこれを通してです。
これは書くものではなく導出されるものです。`is_better_than` は向きを反映したうえで二つの `evaluate` の値を比べるので、実装することは何もありません。

## 骨組み { #skeleton }

```rust
use optopus::prelude::*;
use optopus::error::OptError;

struct MyProblem { /* ... */ }

#[derive(Clone)]
struct MySolution { /* ... */ }

impl Evaluate for MySolution {
    // 問題の向きに合わせて Maximize か Minimize。
    fn evaluate(&self) -> Evaluable<f64> { todo!() }
}

impl ProblemTrait for MyProblem {
    type Solution = MySolution;
    fn new_solution(&self, rng: &mut impl rand::Rng) -> Self::Solution { todo!() }
}

struct MyMove { /* move の座標 */ }

impl MoveToNeighbor<MyProblem> for MyMove {
    fn iter(prob: &MyProblem, sol: &MySolution) -> impl Iterator<Item = Self> + Send {
        std::iter::empty() // move を遅延的に列挙する
    }
    fn apply_to_solution(&self, prob: &MyProblem, sol: &mut MySolution) -> Result<(), OptError> {
        todo!()
    }
    fn move_to_be_better_than(&self, prob: &MyProblem, src: &MySolution, other: &MySolution) -> bool {
        // 既定の impl は src をクローンして適用する。O(1) の gain 判定にするならオーバーライドする
        let mut cloned = src.clone();
        self.apply_to_solution(prob, &mut cloned).expect("apply ok");
        cloned.is_better_than(other)
    }
}

impl Evaluate for MyMove {
    // この move を適用したときの変化量。キャッシュした gain をここで返す。
    // `LocalSearch` と `TabuSearch` は `max_by(rank_cmp)` で選び、
    // 導出された `Rankable` を通してこれを読む。
    fn evaluate(&self) -> Evaluable<f64> { todo!() }
}
```

gain をキャッシュする形は `examples/custom_problem.rs` にあります。

## どのヒューリスティクスに何が必要か { #which-heuristic-needs-what }

以下はすべて任意です。そのヒューリスティクスを使いたいときだけ、その行を実装してください。完全なシグネチャは
[コアトレイト一覧](../traits.md#core-trait-reference) にあります。

| ヒューリスティクス | 必要なトレイト |
|---|---|
| `LocalSearch`, `RandomWalk`, `BeamSearch` | なし |
| `Sequential`, `Iterated`, `VariableNeighborhoodSearch`, `Restart` | なし |
| `SimulatedAnnealing`, `BangBangSimulatedAnnealing`, `LateAcceptanceHillClimbing` | move に [`Evaluate<f64>`](../traits.md#core-trait-reference) |
| `ReinforcementLearningSearch` | move に [`Evaluate<f64>`](../traits.md#core-trait-reference) と `Clone` |
| `TabuSearch` | move に [`EnabledTabu`](../traits.md#core-trait-reference) と `Clone`、加えてその `MoveToNeighbor` impl に `fn tabu_policy(&self) -> Option<&dyn EnabledTabu> { Some(self) }`。この1行が、メモリを持つ [`SearchState`](../search_state.md#remembering-tabu-moves) に方策を渡します |
| `GeneticAlgorithm` | 解に [`Distance`](../traits.md#core-trait-reference) (`DistantTopK` に限らずどの親選択でも必要) と [`Crossover<P>`](../traits.md#core-trait-reference) の impl (`SubProblemBasedCrossover` を使う場合だけ問題に [`SubProblemExtractable`](../traits.md#core-trait-reference)) |
| CLI ベンチマーク (TOML 設定) | 上のすべて |

最後の行は「そのほうが何でも便利になる」という近道ではありません。ベンチマークのファクトリは実行時にヒューリスティクスを選ぶので、境界をまとめて要求します
(`ConfigNeighbor = MoveToNeighbor + Rankable + Evaluate + EnabledTabu + Clone`、
および `ConfigurableProblem::Solution: Distance + Evaluate`)。Rust から直接動かすだけの問題なら、
使うヒューリスティクスが必要とするトレイトまでで止めてかまいません。ベンチマークに登録する問題は、部分的には登録できません。

## 性能についての注意 { #performance-note }

既定の `move_to_be_better_than` は解をクローンして move を適用します。変数ごとの gain をキャッシュして
O(1) で判定するようにオーバーライドしてください。1行で書くなら
`self.evaluate().improves_over(src.evaluate(), other.evaluate())` です。これは最適化の向きを比較演算子として書き直すのではなく、
`Evaluate` の impl から読み取ります。参照実装は `MaxCutFlipNeighbor` と `QuboFlipNeighbor` です。

## 次に読むもの { #next-reading }

- [コアトレイト一覧](../traits.md#core-trait-reference)
- [独自のヒューリスティクス](custom_heuristic.md)
