# 独自の問題を定義する

**API:** [`ProblemTrait`](../../api/optopus/trait_defs/trait.ProblemTrait.html)

三つのトレイトを実装すると、局所探索系のヒューリスティクスとすべてのメタヒューリスティクスがあなたの問題で動きます。
残りのヒューリスティクスは、任意のトレイトを一つ足すごとに一つずつ使えるようになります。下の
[どのヒューリスティクスに何が必要か](#which-heuristic-needs-what) を参照してください。

実行できる完全な例は
[`examples/custom_problem.rs`](https://github.com/trash-iine/optopus/blob/main/examples/custom_problem.rs)
にあります (`cargo run --example custom_problem`)。

問題のすべての変数が決まった範囲の整数か、解が巡回路のような順列なら、代わりに [`IntegerProblem`](../problems/integer.md) を作ってください。
範囲とクロージャで書いた目的関数を渡すだけで実装するものはなく、近傍は最初から用意されています。

## 必須のトレイト { #required-traits }

| トレイト | 実装先 | 必須のメソッド |
|---|---|---|
| [`Evaluate`](../traits.md#core-trait-reference) | `Solution` | `evaluate(&self) -> Evaluable<f64>`。または `#[derive(Evaluate)]` で、値を持つフィールドに `#[objective(minimize)]` を付ける |
| [`ProblemTrait`](../traits.md#core-trait-reference) | 問題の構造体 | `type Solution`, `new_solution(rng) -> Solution` |
| [`MoveToNeighbor<P>`](../traits.md#core-trait-reference) | 近傍の型 | `iter`, `apply_to_solution`, `move_to_be_better_than` |
| [`Evaluate`](../traits.md#core-trait-reference) | 近傍の型 | `evaluate(&self) -> Evaluable<f64>`。上とは別の、二つ目の impl。同じように derive できる |
| [`Rankable`](../traits.md#core-trait-reference) | `Solution` と近傍の型 | `is_better_than`。`#[derive(Rankable)]` で書ける |

`Evaluate` は本当に二回実装します。解に対しては目的関数値を返し、問題が最大化か最小化かに応じて
`Evaluable::Maximize` または `Evaluable::Minimize` で包みます。move に対しては、その move を適用したときの変化量を同じように包んで返します。

変化量とは「move を適用した後の目的関数値 − 適用前の目的関数値」で、適用後の値そのものではありません。
最小化問題の move は `Evaluable::Minimize(after - before)` を返すので負なら改善、最大化問題の move は `Evaluable::Maximize(after - before)` を返すので正なら改善です。
最大値や絶対値のように目的関数が独立な項の和になっていないときは、move の後に目的関数がとる値を計算して、今の値を引きます。
どちらの `evaluate` も問題を見られないので、必要な値は move を作るときにキャッシュしておきます。
`examples/custom_problem.rs` の `FlipMove::new` が gain をキャッシュしているのと同じやり方です。

```rust
#[derive(Evaluate, Rankable)]
struct MyMove {
    index: usize,
    #[objective(minimize)] // 最小化問題。負なら改善
    delta: f64,
}

impl MyMove {
    fn new(prob: &MyProblem, sol: &MySolution, index: usize) -> Self {
        let after = todo!(); // この move の後に `sol` がとる目的関数値
        MyMove { index, delta: after - sol.objective }
    }
}
```

`#[derive(Evaluate)]` は印を付けたフィールドを、属性が示す向きで包んで返します。
二つのフィールドの和のように、一つのフィールドで表せない数値は `impl Evaluate` を手で書きます。

`#[derive(Rankable)]` で move に [`Rankable`](../traits.md#core-trait-reference) が付きます。解にも同じように derive します。
`LocalSearch`、`RandomWalk`、`BeamSearch`、`TabuSearch` が move を選ぶのも、どのヒューリスティクスでも解が改善したかを判断するのもこれを通してです。
`rank` 属性がなければ、向きを反映したうえで二つの `evaluate` の値を比べます。
順序が目的関数値そのものでない問題、たとえば違反をコストより先に比べる問題は、`#[rank(min = violation(), min = cost)]` のように比べるキーを書き、`Evaluate` は焼きなまし系のヒューリスティクスが読む数値として残します。
書き方の一覧は [`Rankable` の rustdoc](../../api/optopus/trait_defs/trait.Rankable.html) にあります。

## 骨組み { #skeleton }

```rust
use optopus::prelude::*;
use optopus::error::OptError;
use optopus::rand; // optopus がビルドに使っている rand

struct MyProblem { /* ... */ }

#[derive(Clone, Rankable)]
struct MySolution { /* ... */ }

impl Evaluate for MySolution {
    // 問題の向きに合わせて Maximize か Minimize。
    fn evaluate(&self) -> Evaluable<f64> { todo!() }
}

impl ProblemTrait for MyProblem {
    type Solution = MySolution;
    fn new_solution(&self, rng: &mut impl rand::Rng) -> Self::Solution { todo!() }
}

#[derive(Rankable)]
struct MyMove { /* move の座標 */ }

impl MoveToNeighbor<MyProblem> for MyMove {
    fn iter(prob: &MyProblem, sol: &MySolution) -> impl Iterator<Item = Self> + Send {
        std::iter::empty() // move を遅延的に列挙する
    }
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        None // まだタブー方策はない。「タブーを付ける」を参照
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
    // `Rankable` を通してこれを読む。
    fn evaluate(&self) -> Evaluable<f64> { todo!() }
}
```

gain をキャッシュする形は `examples/custom_problem.rs` にあります。

`new_solution` は `impl rand::Rng` を受け取り、optopus は自身がビルドに使っている `rand` を再エクスポートしています。
`use optopus::rand;` でその版がスコープに入るので、問題を定義するクレートに `rand` の依存を別に足す必要はありません。

## どのヒューリスティクスに何が必要か { #which-heuristic-needs-what }

以下はすべて任意です。そのヒューリスティクスを使いたいときだけ、その行を実装してください。完全なシグネチャは
[コアトレイト一覧](../traits.md#core-trait-reference) にあります。

| ヒューリスティクス | 必要なトレイト |
|---|---|
| `LocalSearch`, `RandomWalk`, `BeamSearch` | なし |
| `Sequential`, `Iterated`, `VariableNeighborhoodSearch`, `Restart` | なし |
| `SimulatedAnnealing`, `BangBangSimulatedAnnealing`, `LateAcceptanceHillClimbing` | move に [`Evaluate<f64>`](../traits.md#core-trait-reference) |
| `ReinforcementLearningSearch` | move に [`Evaluate<f64>`](../traits.md#core-trait-reference) と `Clone` |
| `TabuSearch` | move に [`EnabledTabu`](../traits.md#core-trait-reference) と `Clone`、加えてその `MoveToNeighbor` impl の `tabu_policy` が `None` でなく `Some(self)` を返すこと。これが、メモリを持つ [`SearchState`](../search_state.md#remembering-tabu-moves) に方策を渡します。[タブーを付ける](#adding-tabu) を参照 |
| `GeneticAlgorithm` | 解に [`Distance`](../traits.md#core-trait-reference) (`DistantTopK` に限らずどの親選択でも必要) と [`Crossover<P>`](../traits.md#core-trait-reference) の impl (`SubProblemBasedCrossover` を使う場合だけ問題に [`SubProblemExtractable`](../traits.md#core-trait-reference)) |
| CLI ベンチマーク (TOML 設定) | 上のすべて |

最後の行は「そのほうが何でも便利になる」という近道ではありません。ベンチマークのファクトリは実行時にヒューリスティクスを選ぶので、境界をまとめて要求します
(`ConfigNeighbor = MoveToNeighbor + Rankable + Evaluate + EnabledTabu + Clone`、
および `ConfigurableProblem::Solution: Distance + Evaluate`)。Rust から直接動かすだけの問題なら、
使うヒューリスティクスが必要とするトレイトまでで止めてかまいません。ベンチマークに登録する問題は、部分的には登録できません。

## タブーを付ける { #adding-tabu }

`TabuSearch` が move のタブーリストを持つのは、その move が「適用したら何を禁止するか」を述べているときだけです。
そのためには `Clone` と、move が触るキーを書いた `#[tabu(..)]` を付け、骨組みの `None` を `Some(self)` に替えます。

```rust
#[derive(Clone, Evaluate, Rankable, EnabledTabu)]
#[tabu(dense(index))] // `index` の禁止が解けていれば許可し、適用したらそれを禁止する
struct MyMove {
    index: usize,
    #[objective(minimize)]
    delta: f64,
}

impl MoveToNeighbor<MyProblem> for MyMove {
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }
    // iter、apply_to_solution、move_to_be_better_than は骨組みのとおり
}
```

move は書いたキーがすべて空いているときに許可され、適用するとそのすべてを `TabuSearch::new` に渡した範囲から引いた tenure の間禁止します。
`dense(i)` は変数のようにインスタンスで上限が決まる添字に向き、`var(i)` は大きさに上限のない添字です。
二つの添字にまたがる move は `dense(i), dense(j)` と両方を書くか、組を `pair(i, j)` で禁止します。`(i, j)` と `(j, i)` が同じ move なら `unordered(i, j)` です。
`triple(a, b, c)` は三つ組のキーです。書き方の一覧は [`EnabledTabu`](../../api/optopus/trait_defs/trait.EnabledTabu.html) の rustdoc にあります。
VRP の relocate のように確認するキーと禁止するキーが違う move は、`optopus::building_blocks::search` の [`TabuKey`](../../api/optopus/building_blocks/search/enum.TabuKey.html) と `TabuMemory` を使って `EnabledTabu` を手で実装します。

`tabu_policy` には既定がないので、move は方策があるかどうかを必ず書きます。
`examples/custom_problem.rs` はこの方策で OneMax 問題に `TabuSearch` をかけます。

## 性能についての注意 { #performance-note }

既定の `move_to_be_better_than` は解をクローンして move を適用します。変数ごとの gain をキャッシュして
O(1) で判定するようにオーバーライドしてください。1行で書くなら
`self.evaluate().improves_over(src.evaluate(), other.evaluate())` です。これは最適化の向きを比較演算子として書き直すのではなく、
`Evaluate` の impl から読み取ります。参照実装は `MaxCutFlipNeighbor` と `QuboFlipNeighbor` です。

## 次に読むもの { #next-reading }

- [コアトレイト一覧](../traits.md#core-trait-reference)
- [独自のヒューリスティクス](custom_heuristic.md)
