# ヒューリスティクス

**API:** [`optopus::heuristic`](../../api/optopus/heuristic/index.html)

どのヒューリスティクスも `Heuristic<P>` (`clear` / `is_done` / `run_once` / `run`) を実装しています。
ヒューリスティクスは問題に依存しません。近傍の型にトレイトを要求するだけなので、それを満たす組み込みの問題や独自の問題はそのまま組み込めます。

## 基本 { #base }

| アルゴリズム | 設定の `kind` | 近傍に必要なトレイト | 補足 |
|---|---|---|---|
| [LocalSearch](local_search.md) | `LocalSearch` | `MoveToNeighbor`, `Rankable` | 貪欲な最良改善。局所最適で止まる。 |
| [SimulatedAnnealing](simulated_annealing.md) | `SimulatedAnnealing` | `MoveToNeighbor`, `Evaluate<f64>` | Boltzmann 受理と乗算的な冷却。 |
| [LateAcceptanceHillClimbing](late_acceptance.md) | `LateAcceptanceHillClimbing` | `MoveToNeighbor`, `Evaluate<f64>` | `history_length` ステップ前のスコアと比較する。 |
| [TabuSearch](tabu_search.md) | `TabuSearch` | `MoveToNeighbor`, `Rankable`, `EnabledTabu` | タブーでない最良の近傍と aspiration。 |
| [RandomWalk](random_walk.md) | `RandomWalk` | `MoveToNeighbor`, `Rankable` | 一様ランダムな move。摂動として便利。 |
| [BeamSearch](beam_search.md) | `BeamSearch` | `MoveToNeighbor`, `Rankable` | 上位 `k` 個の候補を保持する。 |
| [ReinforcementLearningSearch](rl_search.md) | `ReinforcementLearningSearch` | `MoveToNeighbor`, `Evaluate<f64>`, `Clone` | move の特徴量に対するオンライン REINFORCE。 |

## メタ { #meta }

| アルゴリズム | 設定の `kind` | 説明 |
|---|---|---|
| [Sequential / Iterated / VariableNeighborhoodSearch / Restart](meta.md) | 同じ名前 | サブランのクローンとマージのパターンで内側のヒューリスティクスを組み合わせる。 |
| [GeneticAlgorithm](genetic_algorithm.md) | `GeneticAlgorithm` | 親選択 → `Crossover` → 突然変異 → 置き換え。`BiasedFitness` 選択は、さらに集団をコストと多様性で順位付けする。 |

## 交叉オペレータ { #crossover-operators }

`GeneticAlgorithm` が使います。

- `*UniformCrossover`。変数ごとにランダムに親を選びます (問題ごとに一つ)。
- `TspOrderCrossover`。置換のための Order Crossover (OX)。
- `JobShopPpxCrossover`。重複を許す置換のための Precedence-Preserving Crossover。
- [`SubProblemBasedCrossover`](genetic_algorithm.md#subproblembasedcrossover)
  は任意の `P: SubProblemExtractable` に使える汎用の交叉です。

## 問題でパラメータ化されるもの { #parameterized-by-the-problem }

これらは設定で `neighbor` を取りません。複数の move を同時に動かすので、必要なものは一つの move 型ではなく問題の側に書かれます。

| アルゴリズム | 設定の `kind` | 問題に必要なトレイト | 補足 |
|---|---|---|---|
| [PopulationAnnealing](population_annealing.md) | `PopulationAnnealing` | 解と move の両方に `Evaluate` | β で冷却するレプリカ集団を、ステップごとにリサンプリングし、Metropolis でスイープする。任意の問題。 |
| [BreakoutLocalSearch](breakout_local_search.md) | `BreakoutLocalSearch` | `ProblemTrait` 以外はなし。スケジュールがさらに要求することはある | 貪欲な降下と、ヒューリスティクスの集まりに対する適応的な摂動。MaxCut に登録済み。 |
| [AdaptiveLargeNeighborhoodSearch](alns.md) | `AdaptiveLargeNeighborhoodSearch` | `Ruinable`、解に `Evaluate` | 適応的なオペレータ重みと SA 受理による ruin-and-recreate。任意でアンカー付きの `LocalRepair`。CVRP と TSP に登録済み。 |

## 問題専用 { #problem-specific }

これらも設定で `neighbor` を取りませんが、それぞれが自前の move 集合を持っています。

| アルゴリズム | 設定の `kind` | 問題 | 補足 |
|---|---|---|---|
| [LinKernighanHelsgaunForTsp](lkh.md) | `LinKernighanHelsgaun` | 2 次元の TSP | 候補リストを使う可変深さの k-opt。 |
| [WalkSatForSat](walksat.md) | `WalkSat` | MaxSAT | ランダムな非充足節の中での集中的な SKC flip。任意で適応的なノイズ。 |
| [HybridGeneticSearchForVrp](hgs.md) | `HybridGeneticSearch` | CVRP | giant tour の GA、最適な Split、granular な局所探索。実行可能と実行不能の部分集団に対する biased fitness。 |
