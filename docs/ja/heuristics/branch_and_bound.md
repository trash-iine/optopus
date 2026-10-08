# BranchAndBound

**API:** [`BranchAndBound`](../../api/optopus/heuristic/struct.BranchAndBound.html)

[整数変数](../problems/integer.md)と、二値問題である MaxCut、QUBO、MaxSAT、vertex cover の上の branch-and-bound です。解はその問題の任意のヒューリスティクスが見つけ、まだ調べていない部分に残りうる値は緩和が抑えます。最後まで走った実行は、見つけた最良解が最適であることを証明しています。途中で止めた実行は、最適からどれだけ離れうるかを返します。

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

// 時間で止めるなら StopCondition::duration(std::time::Duration::from_secs(10))、
// 停滞で止めるなら StopCondition::failed_updates(1_000)。.with_duration(...) や
// .with_iterations(...) で組み合わせられる。一覧は Stop conditions ガイドにある。
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

二値問題では、内側のヒューリスティクスはその問題自身のもので、上界も用意されています。

```rust
use optopus::prelude::*;

let prob = MaxCut::new(Graph::erdos_renyi(16, 0.3, &mut seeded_rng(1)));
let mut bnb = BranchAndBound::new(
    StopCondition::new(None, None, None),
    Box::new(TabuSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(50), (2, 4))),
    BinaryRelaxation,
);
let mut state = SearchState::new_with_seed(&prob, 1);
bnb.run(&mut state)?;
assert!(bnb.is_proven_optimal(&state.best_solution));
```

`examples/max_cut_branch_and_bound.rs` は、MaxCut の各ノードで Breakout Local Search を走らせ、`EigenvalueRelaxation` か正の重みの和で上界を与えます。

## アルゴリズムの概要 { #algorithm-sketch }

ノードは、いくつかの変数の値域を狭めた問題です。未処理のノードは上界の順に保持され、`run_once` のたびに最良のものを 1 つ取り出します。問題全体を一度探索してから上界を計算するので、最初の上界は探索済みの暫定解に対して計算されます。

1. ノードの上界がこれまでの最良解を上回れなければ、そのノードは捨てます。
2. ノードのすべての変数が 1 つの値しか取らなければ、その割り当てを評価して終わりです。
3. そうでなければ、内側のヒューリスティクスがノードを探索します。開始点は最良解の各値をノードの値域に寄せたものです。探索はノード用に作り直した問題の上で走るので、move がノードの外に出ることはありません。整数問題は狭めた値域で作り直します。二値問題は固定した変数を畳み込んだ、自分と同じ型の小さなインスタンスにします。
4. 1 つの変数の値域を、最良解でのその値で 2 つに分けます。緩和がそれぞれの半分に上界を与え、最良解を上回りうる半分だけを残します。

実行は、ノードがなくなったとき、またはそれより前に停止条件を満たしたときに終わります。

## 問題に必要なもの { #what-the-problem-needs }

探索が要求するのは `BranchSpace` で、以下の問題はすべて実装済みです。

整数問題は `Branchable` を実装します。`IntegerProblem` と `FormulaProblem` は実装済みです。自前の問題が実装するのは、同じ問題をより狭い値域で返す `restricted` の 1 メソッドだけです。目的関数は値域に依存してはいけません。permutation は変数 1 つずつ狭めることができないので、実行はエラーを返します。

`IntegerProblem` は、クロージャを clone できるときに `Branchable` になります。ノードごとに clone するので、大きな表を読むクロージャには表そのものではなく表への参照を渡すほうがよいです。

二値問題は `FixVariables` を実装します。MaxCut、`Qubo`、`Sat`、`VertexCover` は実装済みです。`fix` は、固定した変数を畳み込んだ自由変数だけのインスタンスと、定数のオフセットと、各変数の行き先を返します。探索はそのインスタンスとの間を、`BinaryProblem::solution_from_assignment` で割り当てから解を作り直して行き来するので、解がキャッシュしているものは一から計算されます。自前の二値問題は `FixVariables` を実装し、`BranchSpace` はその隣の `optopus::binary_branch_space!(MyProblem);` で得られます。

| 問題 | 固定の畳み込み方 |
|---|---|
| MaxCut | 固定した頂点を 1 つの基準頂点にまとめ、自由な頂点はそれとの相対で読む。すべての解で厳密。 |
| QUBO | `1` に固定した変数との積は一次の項になる。すべての解で厳密。 |
| MaxSAT | 固定したリテラルで充足した節はオフセットに入れ、偽のリテラルは落とす。すべての解で厳密。 |
| Vertex cover | 被覆の外に固定した頂点に隣接する自由な頂点は被覆に入れる。最良解で厳密。 |

## 緩和 { #relaxations }

緩和は `Relaxation<P>` を実装し、与えられた値域について、その中のどの割り当ても上回れない値を問題の向き付きで返します。**そのような上界になっていない値は、エラーを出さずに最適解を刈り取ります。** 新しい緩和は小さなインスタンスで全列挙と照らし合わせておく価値があります。向きの違う上界はエラーとして報告します。

- `IntervalRelaxation` は任意の `FormulaProblem` で使えます。各単項式の変数の値域を掛け合わせ、各制約にはその式が取りうる最小のペナルティを課します。どの式でも妥当ですが、多くの式では弱い上界です。複数の単項式に現れる変数を、それぞれの単項式で最悪の値に取るためです。
- `BinaryRelaxation` は任意の二値問題で使えます。ノードを畳み込み、畳み込んだインスタンスの `trivial_bound` にオフセットを足します。MaxCut では正の重みの和、QUBO では負の係数の和、MaxSAT では節の数、vertex cover では貪欲な極大マッチングの大きさです。証明できるのは数十変数までで、24 頂点の MaxCut なら 1 秒もかかりませんが、ベンチマークのインスタンスは証明できません。
- `EigenvalueRelaxation` は MaxCut で使えます。ノードを畳み込み、畳み込んだグラフのラプラシアンを `L` として `(n/4) λ_max(L + diag(u)) − ¼ Σu` で抑え、補正 `u` を劣勾配法で改善します。上界は Rump の安全シフト付き Cholesky 分解で浮動小数点のまま証明され、証明できないときや、畳み込んだグラフの頂点数が `with_certify_limit` を超えるときは正の重みの和に戻ります。分解は密なので、コストはその頂点数の 3 乗で増えます。重みが整数なら上界は切り捨てます。正の重みの和では 30 頂点前後が限界のランダムグラフで、60 頂点前後まで証明できます。
- クロージャ `|prob, vars| -> Evaluable<f64>` は緩和になります。問題を知っている上界はこの形で渡します。
- `bound_against` は、これまでの最良解の目的関数値を受け取る `bound` で、既定では無視します。`EigenvalueRelaxation` はこれを劣勾配法の目標値に使い、上界がそこまで下がった時点で止めます。そのノードは、それ以上締めても刈られることに変わりがないためです。
- `branch_hint` は次に分ける変数を指定できます。既定では、まだ選択の余地がある変数のうち番号が最小のものを分けます。

## コンストラクタ { #constructor }

```rust
BranchAndBound::new(
    stop_condition: StopCondition,
    heuristic: Box<dyn Heuristic<P>>,
    relaxation: R,
) -> Self
```

`P: BranchSpace`、`R: Relaxation<P>` です。

## 振る舞い { #behavior }

- 各ノードは、内側のヒューリスティクスが走らせた分に加えて 1 反復と数えます。内側の停止条件はノードごとの予算です。
- `is_proven_optimal(&best)` は、どの未処理ノードも `best` を上回れなくなると真になります。`dual_bound(&best)` はまだ除外されていない割り当てが到達しうる最良の目的関数値で、証明が済むと最良解の値に一致します。
- ノードの中で見つかった改善はノードの探索が戻ったときに記録されるので、trajectory はノードごとに 1 段ずつ進みます。
- 結果の厳密さは緩和の正しさまでです。ペナルティとして書いた制約はペナルティとして最適化されるので、違反を排除するには小さすぎる重みでは、違反した最適解が残ります。

## ベンチマーク設定 { #benchmark-config }

`BranchAndBound` は今のところライブラリ専用です。

## 参考文献 { #references }

- Land, A. H. and Doig, A. G. "An automatic method of solving discrete
  programming problems". *Econometrica* 28(3), 497-520, 1960.
