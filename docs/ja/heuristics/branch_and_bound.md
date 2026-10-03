# BranchAndBound

**API:** [`BranchAndBound`](../../api/optopus/heuristic/struct.BranchAndBound.html)

[整数変数](../problems/integer.md)上の branch-and-bound です。解は任意のヒューリスティクスが見つけ、まだ調べていない部分に残りうる値は緩和が抑えます。最後まで走った実行は、見つけた最良解が最適であることを証明しています。途中で止めた実行は、最適からどれだけ離れうるかを返します。

## 例 { #example }

```rust
use optopus::prelude::*;

// maximize x0 + 2 x1 + 3 x2 with at most two of them set
let vars: IntVars = (0..3).map(|_| IntVar::binary()).collect();
let objective = Expr::Var(0) + 2.0 * Expr::Var(1) + 3.0 * Expr::Var(2);
let prob = FormulaProblem::maximize(vars, objective).with_constraint(Constraint::Comparison {
    lhs: Expr::Var(0) + Expr::Var(1) + Expr::Var(2),
    rel: ConstraintRel::Le,
    rhs: Expr::Const(2.0),
    penalty_weight: 10.0,
});

let mut bnb = BranchAndBound::new(
    StopCondition::new(None, None, None),
    Box::new(LocalSearch::<IntChangeNeighbor>::new(StopCondition::iterations(100))),
    IntervalRelaxation,
);
let mut state = SearchState::new_with_seed(&prob, 1);
bnb.run(&mut state)?;
assert!(bnb.is_proven_optimal(&state.best_solution));
```

`examples/integer_branch_and_bound.rs` は、Dantzig の分数詰めを上界にして 40 品目のナップサックをこの方法で解きます。

## アルゴリズムの概要 { #algorithm-sketch }

ノードは、いくつかの変数の値域を狭めた問題です。未処理のノードは上界の順に保持され、`run_once` のたびに最良のものを 1 つ取り出します。

1. ノードの上界がこれまでの最良解を上回れなければ、そのノードは捨てます。
2. ノードのすべての変数が 1 つの値しか取らなければ、その割り当てを評価して終わりです。
3. そうでなければ、内側のヒューリスティクスがノードを探索します。開始点は最良解の各値をノードの値域に寄せたものです。探索はその値域で作り直した問題の上で走るので、move がノードの外に出ることはありません。
4. 1 つの変数の値域を、最良解でのその値で 2 つに分けます。緩和がそれぞれの半分に上界を与え、最良解を上回りうる半分だけを残します。

実行は、ノードがなくなったとき、またはそれより前に停止条件を満たしたときに終わります。

## 問題に必要なもの { #what-the-problem-needs }

問題は `Branchable` を実装します。`IntegerProblem` と `FormulaProblem` は実装済みです。自前の問題が実装するのは、同じ問題をより狭い値域で返す `restricted` の 1 メソッドだけです。目的関数は値域に依存してはいけません。permutation は変数 1 つずつ狭めることができないので、実行はエラーを返します。

`IntegerProblem` は、クロージャを clone できるときに `Branchable` になります。ノードごとに clone するので、大きな表を読むクロージャには表そのものではなく表への参照を渡すほうがよいです。

## 緩和 { #relaxations }

緩和は `Relaxation<P>` を実装し、与えられた値域について、その中のどの割り当ても上回れない値を問題の向き付きで返します。**そのような上界になっていない値は、エラーを出さずに最適解を刈り取ります。** 新しい緩和は小さなインスタンスで全列挙と照らし合わせておく価値があります。向きの違う上界はエラーとして報告します。

- `IntervalRelaxation` は任意の `FormulaProblem` で使えます。各単項式の変数の値域を掛け合わせ、各制約にはその式が取りうる最小のペナルティを課します。どの式でも妥当ですが、多くの式では弱い上界です。複数の単項式に現れる変数を、それぞれの単項式で最悪の値に取るためです。
- クロージャ `|prob, vars| -> Evaluable<f64>` は緩和になります。問題を知っている上界はこの形で渡します。
- `branch_hint` は次に分ける変数を指定できます。既定では、まだ選択の余地がある変数のうち番号が最小のものを分けます。

## コンストラクタ { #constructor }

```rust
BranchAndBound::new(
    stop_condition: StopCondition,
    heuristic: Box<dyn Heuristic<P>>,
    relaxation: R,
) -> Self
```

`P: Branchable`、`R: Relaxation<P>` です。

## 振る舞い { #behavior }

- 各ノードは、内側のヒューリスティクスが走らせた分に加えて 1 反復と数えます。内側の停止条件はノードごとの予算です。
- `is_proven_optimal(&best)` は、どの未処理ノードも `best` を上回れなくなると真になります。`dual_bound(&best)` はまだ除外されていない割り当てが到達しうる最良の目的関数値で、証明が済むと最良解の値に一致します。
- ノードの中で見つかった改善はノードの探索が戻ったときに記録されるので、trajectory はノードごとに 1 段ずつ進みます。
- 結果の厳密さは緩和の正しさまでです。ペナルティとして書いた制約はペナルティとして最適化されるので、違反を排除するには小さすぎる重みでは、違反した最適解が残ります。

## ベンチマーク設定 { #benchmark-config }

`BranchAndBound` は今のところライブラリ専用です。ベンチマークには、これを走らせる整数問題の種類がありません。

## 参考文献 { #references }

- Land, A. H. and Doig, A. G. "An automatic method of solving discrete
  programming problems". *Econometrica* 28(3), 497-520, 1960.
