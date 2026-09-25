# Formula

**API:** [`FormulaProblem`](../../api/optopus/problem/binary_optimization/struct.FormulaProblem.html)

`FormulaProblem` は、上の名前付きの問題のどれにも当てはまらない目的関数のための、設定可能な二値最適化問題です。
`n` 個の二値変数 `x ∈ {0,1}^n` 上の任意の算術式を目的関数として宣言し、`Maximize` か `Minimize` かを選び、
ペナルティ重み付きの制約をいくつでも追加できます。内部ではどちらの向きも、大きいほど良い一つの `score` の最大化として最適化されるので、
ヒューリスティクスは利用者がどちらの向きを選んだかを知る必要がありません。

```text
Maximize:  score(x) = objective(x) − Σ_c penalty_weight_c · violation_c(x)
Minimize:  score(x) = −objective(x) − Σ_c penalty_weight_c · violation_c(x)
```

## 例 { #example }

探索を実行し、割り当て、その生の目的関数値、順位付けに使う内部の `score` を読み出します。

```rust
use optopus::prelude::*;

// x[0] + 2*x[1] + 3*x[2] を最大化する。制約は x[0] + x[1] + x[2] <= 2
let objective = Expr::Var(0) + 2.0 * Expr::Var(1) + 3.0 * Expr::Var(2);
let constraint = Constraint::Comparison {
    lhs: Expr::Var(0) + Expr::Var(1) + Expr::Var(2),
    rel: ConstraintRel::Le,
    rhs: Expr::Const(2.0),
    penalty_weight: 10.0,
};
let prob = FormulaProblem::new(3, objective, OptDirection::Maximize, vec![constraint]);

let mut state = SearchState::new(&prob);
LocalSearch::<FormulaFlipNeighbor>::new(StopCondition::iterations(10_000))
    .run(&mut state)
    .unwrap();

let sol = &state.best_solution;
println!("assignment = {:?}", sol.x);
println!("objective value = {}", prob.eval_objective(&sol.x)); // 利用者が宣言した式の、ペナルティを引く前の値
println!("score = {}", sol.score); // 大きいほど良い内部の順位付けの値。`evaluate` が返すのはこれ
```

ファイルローダはありません。上のように `Expr` の AST からプログラムで問題を作ります。

## 解 { #solution }

[`FormulaSolution`](../../api/optopus/problem/binary_optimization/struct.FormulaSolution.html)
は上の定義の割り当て `x` (`x ∈ {0,1}^n`)、変数ごとの `gain` (その変数を反転したときの `score` の変化)、
そして上で定義した `score(x)` である `score` を持ちます。`score` は `objective(x)` そのものではありません。
フィールドの一覧は rustdoc を参照してください。
`Evaluate` はこれを `Evaluable::Maximize(score)` として返します。`score` には式自身の向きがすでに畳み込まれているので、ここでは常に大きいほど良い値です。

## 式 { #expressions }

目的関数と制約の両辺は [`Expr`](../../api/optopus/problem/binary_optimization/enum.Expr.html)
の AST から作ります。正確な種類の一覧は rustdoc を参照してください。`Expr` は標準の算術演算子 (`+ - * /`) を
`Expr × Expr` と `Expr × f64` の両方についてオーバーロードしていて、普通はそれを使って組み立てます。

```rust
use optopus::problem::Expr;

// 線形結合 2*x[0] + x[1] - 3
let linear = 2.0 * Expr::Var(0) + Expr::Var(1) - 3.0;

// 二つの二値変数の AND ({0,1} の値では Mul ≡ AND)
let and_of_two = Expr::Var(0) * Expr::Var(1);
```

`Add` と `Mul` は自動的に平たくされます。除算は定数で割る場合だけ対応しています。

## 制約 { #constraints }

制約は [`Constraint`](../../api/optopus/problem/binary_optimization/enum.Constraint.html)
から作ります (正確な種類の一覧は rustdoc を参照してください)。違反には `violation * penalty_weight` のペナルティがかかります。
`violation` は制約に違反している量です (満たしていれば `0`)。

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

`Lt` と `Gt` は小さな `STRICT_EPSILON` を使うので、等号は違反として数えられます。

## 近傍 { #neighbors }

| 型 | move |
|---|---|
| `FormulaFlipNeighbor` | 変数を一つ反転する。 |
| `FormulaSwapNeighbor` | 二つの変数を入れ替える。 |

どちらも `Evaluate<f64>` と `Evaluate<i32>` (整数版はスコアを離散化します。係数がすべて整数値のときに向いています)、
そして `EnabledTabu` を実装しています。gain の差分更新の仕組みと反復コストは
[`FormulaFlipNeighbor`](../../api/optopus/problem/binary_optimization/struct.FormulaFlipNeighbor.html) /
[`FormulaSwapNeighbor`](../../api/optopus/problem/binary_optimization/struct.FormulaSwapNeighbor.html)
の rustdoc を参照してください。

## 交叉 { #crossover }

- `FormulaUniformCrossover`。変数ごとにランダムに親を選びます。

## 任意のトレイト { #optional-traits }

- `Distance`。`x` 上のハミング距離です。
- `Evaluate<f64>` と `Evaluate<i32>`。`Evaluable` の両方の向きに対応します。

## 補足 { #notes }

`CompiledPoly` と `interaction_neighbors` は非公開 (`pub(super)`) の実装の詳細なので、公開される rustdoc には出てきません。
利用者向けに説明しているのはここだけです。

- 事前にコンパイルした多項式の形 (`CompiledPoly`) により、反転一回あたりの gain の差分が O(d) で求まります。d は反転した変数を含む単項式の数です。
- `interaction_neighbors[i]` は `i` を反転したときに gain が変わりうる変数を並べたものです。
  目的関数で単項式を共有する変数と、どれかの制約式に一緒に現れる変数からなります。gain の更新はそれ以外の変数をすべて飛ばします。

(ソースを読むコントリビュータへ。どちらも `src/problem/binary_optimization/problem.rs` に説明があります。)
