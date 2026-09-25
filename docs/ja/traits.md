# コアトレイト リファレンス

**API:** [`optopus::trait_defs`](../api/optopus/trait_defs/index.html)

[基本概念](concepts.md) の補足リファレンスです。問題型や move 型が実装できるすべてのトレイトの完全なシグネチャと、
それぞれがどのヒューリスティクスを使えるようにするかを一覧にしています。

これらのトレイトはすべて `optopus::trait_defs` で定義され、`optopus::search_state` と prelude から再エクスポートされています。
最初の三つは入場券です。これがなければどのヒューリスティクスも動きませんが、これだけで十分というわけでもありません。
下の表で `Rankable` が解と move の二か所に別々の impl として現れること、そしてそれ以外のトレイトはそれぞれ特定のアルゴリズム群を使えるようにすることに注意してください。
「X を動かすには何を実装すればよいか」の正式な答えは「必要とするもの」の列です。
最後の行の `ProblemReduction` は例外で、何も使えるようにはせず、問題ではなく縮約が実装します。表の下の節を参照してください。

## コアトレイト一覧 { #core-trait-reference }

| トレイト | 必要とするもの | 主なシグネチャ |
|---|---|---|
| `ProblemTrait` | すべてのヒューリスティクス | `type Solution: Clone + Rankable; fn new_solution(&self, rng) -> Solution` |
| `Evaluate<f64>` (`Solution` に実装) | すべてのヒューリスティクス | `fn evaluate(&self) -> Evaluable<f64>`。解自身の目的関数値を `Evaluable::Maximize` または `Minimize` で包んだもの。解に `Rankable` を与えるのもこれです。下を参照。 |
| `MoveToNeighbor<P>` | すべてのヒューリスティクス | `fn iter(prob, sol) -> impl Iterator<Self> + Send`<br>`fn apply_to_solution(&self, prob, sol) -> Result<()>`<br>`fn random_neighbor(prob, sol, rng) -> Option<Self>` (既定は `iter` からのリザーバサンプリング)<br>`fn move_to_be_better_than(&self, prob, src, other) -> bool` (既定はクローンして適用)<br>`fn apply_to_iteration(&self, iter) -> u64` (既定は `iter + 1`)<br>`fn tabu_policy(&self) -> Option<&dyn EnabledTabu>` (既定は `None` で、タブー方策なし) |
| `Rankable` (両方に実装) | `LocalSearch`、`BeamSearch`、`RandomWalk`、`TabuSearch`、および解が改善したかを問うすべてのヒューリスティクス | `fn is_better_than(&self, other: &Self) -> bool`。手では実装しません。ブランケット impl が `Evaluate` から導出し、二つの `minimized()` の値を比べます。`Evaluate` を実装しない型は、一つの数の比較ではない順序のためにこれを直接実装することもできますが、その場合 `Evaluate` とそれを必要とするすべてのヒューリスティクスを諦めることになります。 |
| `Evaluate<T>` (move に実装) | `SimulatedAnnealing`、`BangBangSimulatedAnnealing`、`LateAcceptanceHillClimbing`、`ReinforcementLearningSearch` (最後のものは move に `Clone` も必要) | `fn evaluate(&self) -> Evaluable<T>` (既定は `T = f64`)。move を適用したときの変化量です。`Evaluable::Maximize(T)` / `Minimize(T)` が最適化の向きを持ち、`Evaluable<f64>::minimized()` がそれを適用して小さいほど良い値にします。これは差分にとっての悪化量であり、目的関数にとってのエネルギーでもあります。 |
| `EnabledTabu` | `TabuSearch` (move に `Rankable` と `Clone` も必要) | `fn is_move_enabled(&self, tabu: &TabuMemory, iter) -> bool;`<br>`fn add_to_tabu_map(&self, tabu: &mut TabuMemory, iter, rng: &mut SmallRng)`<br>加えて move の `MoveToNeighbor` impl に1行。`fn tabu_policy(&self) -> Option<&dyn EnabledTabu> { Some(self) }` |
| `Crossover<P>` | `GeneticAlgorithm` (`Distance` も必要、下を参照) | `fn crossover(&mut self, prob, sol1, sol2, rng: &mut SmallRng) -> Result<P::Solution, OptError>` (`&mut self` なので状態を持つオペレータがサブヒューリスティクスを走らせられる) |
| `SubProblemExtractable` | `SubProblemBasedCrossover` | `fn extract_sub_problem(&self, sol1, sol2) -> Self;`<br>`fn lift_solution(&self, sol1, sol2, sub_solution) -> Self::Solution` |
| `Distance` (`Solution` に実装) | `GeneticAlgorithm`。`ParentSelection::DistantTopK` に限らずどの選択戦略でも必要 | `fn distance(&self, other: &Self) -> usize` |
| `BinaryProblem` | `common::binary` にある共通の二値機構 | `type Flip;`<br>`fn variable_indices(&self) -> Range<usize>;`<br>`fn variable(sol, i) -> bool;`<br>`fn flip_move(sol, i) -> Self::Flip` |
| `Ruinable` | `AdaptiveLargeNeighborhoodSearch` | `type Element: Copy + Eq; type Partial;`<br>`to_partial` / `finish`, `elements` / `remove_all`,<br>`removal_gain` / `relatedness` (破壊),<br>`num_buckets` / `num_places` / `insertion_cost` / `insert` (修復),<br>`num_elements` / `partial_energy` (どちらも既定あり) |
| `LocalRepair<P: Ruinable>` | `AdaptiveLargeNeighborhoodSearch` (任意) | `fn repair_around(&mut self, prob: &P, partial: &mut P::Partial, anchors: &[P::Element], rng: &mut SmallRng)` |
| `ProblemReduction` | なし。要件ではなく道具です | `type Source: ProblemTrait; type Target: ProblemTrait;`<br>`fn target(&self) -> &Self::Target;`<br>`fn project(&self, sol: &SourceSolution) -> TargetSolution;`<br>`fn lift(&self, source: &Self::Source, base: &SourceSolution, sol: &TargetSolution) -> SourceSolution` |

上の `SmallRng` は `rand::rngs::SmallRng` です。乱数を必要とするトレイトメソッドはすべて、スレッド RNG を使わずに引数として受け取ります。
呼び出し側は `&mut state.rng` を渡します。これによって、タブーテニュアでも交叉でも、シード付きの実行がビット単位で再現されます。

最後の行の `SourceSolution` と `TargetSolution` は
`<Self::Source as ProblemTrait>::Solution` と
`<Self::Target as ProblemTrait>::Solution` の略で、トレイトではそう書かれています。
どちらにも別名はありません。

QUBO では gain の値が整数なので、対応する評価は
`Evaluate<i32>` (と `Evaluable<i32>`) です。

## コアの三つだけで得られるもの { #what-the-core-three-alone-buy-you }

`ProblemTrait`、`Rankable` (解と move の両方)、`MoveToNeighbor` があり、ほかに何もない場合は次のようになります。

- `LocalSearch`、`RandomWalk`、`BeamSearch` が動きます。
- すべてのメタヒューリスティクス (`Sequential`、`Iterated`、`VariableNeighborhoodSearch`、
  `Restart`) が動きます。これらは `P: ProblemTrait` についてジェネリックで、要件は内側のヒューリスティクスのものを引き継ぐだけだからです。
- `TabuSearch`、`SimulatedAnnealing`、`LateAcceptanceHillClimbing`、`ReinforcementLearningSearch`
  と `GeneticAlgorithm` は動きません。それぞれ上の表の行に書かれたトレイトが必要です。

## `ProblemReduction`

これはここまでのものと違い、問題がヒューリスティクスを使えるようにするために実装するものではありません。
ある問題インスタンスから別のインスタンスへの写像で、解は両方向に行き来します。探索する小さな対象、
ウォームスタートのための入口、そして戻るための出口です。`MaxCutKernel` がこれを実装しています
([kernelization](problems/max_cut_kernel.md)、`Source = Target = MaxCut`)。
`Source` と `Target` が別々の関連型になっているのは、縮約が同じ問題の中にとどまるとは限らないからです。
ペナルティ項が二次の目的関数は `Qubo` に縮約できます。

`lift` が引数を二つ余分に取る理由は、実装する前に知っておく価値があります。`base` は写像が落としたものを補います。
写した先がもとの変数インデックス空間の全体を覆っていない場合 (孤立頂点を削除した kernelization など)、
覆われていない位置は `base` の値を保ちます。`source` がフィールドではなく引数なのは、解が持つ差分計算用のキャッシュはインスタンスがなければ再構築できず、
しかも `&Self::Source` を保持することはすべての実装に許されるわけではないからです。ヒューリスティクスの中に保存された縮約は、どの一回の呼び出しよりも長く生きます。

厳密さはトレイトの一部ではありません。近似的な縮約も同じ形をしています。利用する側がそれぞれ必要なものを要求します。
`MaxCutKernel` は最適解に限らずすべての `y` について `kernel_cut(y) + offset == original_cut(lift(y))` を保証しています。
そのため、ヒューリスティクスをどの時点で止めても持ち上げられます。

トレイトは写像だけです。写した先でヒューリスティクスを走らせて結果を戻すのは探索状態の操作なので、そちらにあります。
[`SearchState::open_reduction` と
`close_reduction`](search_state.md#crossing-a-reduction) です。これを手で書くと、複製同士が目的関数値ではなく
`iteration` / `n_accepted` / `best_iteration` で、気づかないうちにずれていきます。
