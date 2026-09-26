# Formula

**API:** [`FormulaProblem`](../../api/optopus/problem/integer/struct.FormulaProblem.html)

`FormulaProblem` は、整数変数の算術式で書ける目的関数のための問題です。目的関数を式として宣言し、最大化か最小化かを選び、
ペナルティ重み付きの制約をいくつでも追加できます。変数は [`IntegerProblem`](integer.md) と同じ `IntVars` なので、
binary でも任意の整数範囲でもよく、変数全体を順列にすることもできます。

差分を書く必要はありません。問題を作るときに式を単項式にコンパイルし、一つの変数の変化はその変数が現れる単項式と制約だけから求めます。
さらに各解は、どの変数をどの値に変えたら何が変わるかの表を持っています。近傍を選ぶときはその表を読み、
適用したときは影響しうる変数の分だけを計算し直します。

```text
最大化:  objective(x) − Σ_c penalty_weight_c · violation_c(x)
最小化:  objective(x) + Σ_c penalty_weight_c · violation_c(x)
```

## 例 { #example }

```rust
use optopus::prelude::*;

// binary の x について x[0] + 2*x[1] + 3*x[2] を最大化する。制約は x[0] + x[1] + x[2] <= 2
let vars: IntVars = (0..3).map(|_| IntVar::binary()).collect();
let objective = Expr::Var(0) + 2.0 * Expr::Var(1) + 3.0 * Expr::Var(2);
let prob = FormulaProblem::maximize(vars, objective).with_constraint(Constraint::Comparison {
    lhs: Expr::Var(0) + Expr::Var(1) + Expr::Var(2),
    rel: ConstraintRel::Le,
    rhs: Expr::Const(2.0),
    penalty_weight: 10.0,
});

let mut state = SearchState::new(&prob);
TabuSearch::<IntChangeNeighbor>::new(StopCondition::iterations(1_000), (1, 2))
    .run(&mut state)
    .unwrap();

let values = state.best_solution.values();
println!("assignment = {values:?}");
println!("objective = {}", prob.eval_objective(values)); // 式の値。ペナルティを引く前
println!("penalty = {}", prob.eval_penalty(values));
```

ファイルローダはありません。上のようにコードで問題を作ります。

## 解 { #solution }

`FormulaSolution` は各変数の値を持ち、`values()` で読めます。`Evaluate` は問題の向き付きのペナルティ込みの目的値を返します。
最大化なら `Evaluable::Maximize(objective − penalty)`、最小化なら `Evaluable::Minimize(objective + penalty)` で、探索はこれで順位を付けます。
二つの部分はそれぞれ `eval_objective` と `eval_penalty` で得られます。

## 式 { #expressions }

目的関数と制約の両辺は `Expr` です。`Expr × Expr` と `Expr × f64` について算術演算子をオーバーロードしていて、普通はそれを使って組み立てます。

```rust
use optopus::problem::Expr;

// 線形結合 2*x[0] + x[1] - 3
let linear = 2.0 * Expr::Var(0) + Expr::Var(1) - 3.0;

// 積。binary 変数では AND
let and_of_two = Expr::Var(0) * Expr::Var(1);

// 整数変数では冪がそのまま残る
let square = Expr::Var(2) * Expr::Var(2);
```

`Expr::Var(i)` は変数 `i` の値として評価されます。`Add` と `Mul` は組み立てるときに平たくされます。除算は定数で割る場合だけ対応しています。
`IntVars` の外の変数を読む式で問題を作ると panic します。

## 制約 { #constraints }

`Constraint` は違反に `violation * penalty_weight` のペナルティをかけます。`violation` は違反している量で、満たしていれば `0` です。

```rust
use optopus::problem::{Constraint, ConstraintRel, Expr};

// x[0] + x[1] + x[2] <= 2。違反 1 単位あたり重み 10.0 のペナルティ
let constraint = Constraint::Comparison {
    lhs: Expr::Var(0) + Expr::Var(1) + Expr::Var(2),
    rel: ConstraintRel::Le,
    rhs: Expr::Const(2.0),
    penalty_weight: 10.0,
};
```

`Constraint::Clamp` は式を `lo..=hi` に収めます。`Lt` と `Gt` は等号のときに小さなペナルティをかけるので、狭義の関係では等号が無料になりません。

## 近傍 { #moves }

近傍は整数変数の問題すべてに付いてくるものと同じです。

| 近傍 | 何をするか |
|---|---|
| `IntChangeNeighbor` | 一つの変数を範囲内の別の値にする。binary 変数なら Flip |
| `IntSwapNeighbor` | 二つの変数の値を交換する |
| `IntReverseNeighbor` | 変数の区間の値の並びを反転する |

`IntChangeNeighbor` は解の表から値を読みます。Swap はどちらかの変数を読む単項式と制約だけから求めます。
Reverse は解の複製を評価します。式の問題で Reverse を使うことはほとんどないためです。

## 交叉と部分問題 { #crossover-and-sub-problems }

`IntCrossover` は各変数をどちらかの親からとり、`FormulaSolution` は `Distance` を実装しているので、`FormulaProblem` で `GeneticAlgorithm` が動きます。

`FormulaProblem` は `SubProblemExtractable` も実装しています。親どうしで値が違う変数だけが小さな `FormulaProblem` になり、
それ以外の変数は式の中で値に置き換えられるので、`SubProblemBasedCrossover` も動きます。
ただし変数が順列であってはいけません。順列の一部の位置を固定すると、残りは順列でなくなるためです。
