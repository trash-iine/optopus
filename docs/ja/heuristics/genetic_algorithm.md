# GeneticAlgorithm

**API:** [`GeneticAlgorithm`](../../api/optopus/heuristic/struct.GeneticAlgorithm.html)

集団に基づく探索です。解の集団を `Crossover<P>` オペレータで二つずつ組み換え、`Heuristic<P>` を突然変異オペレータとして使います。

## 例 { #example }

HEA 風のハイブリッド GA です。`SubProblemBasedCrossover` で組み換え、`TabuSearch` を突然変異オペレータとし、
ランダムな初期個体はすべて最初に改善しておきます。

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]));
let mut state = SearchState::new(&mc);

let mut ga = GeneticAlgorithm::new(
    StopCondition::iterations(10_000),
    /* population_size  = */ 50,
    SubProblemBasedCrossover {
        sub_heuristic: Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
            StopCondition::failed_updates(1),
        )),
    },
    /* mutation         = */ Box::new(TabuSearch::<MaxCutFlipNeighbor>::new(
        StopCondition::failed_updates(100),
        (5, 10),
        None,
    )),
    ParentSelection::Tournament,
)
.with_init_improvement(Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
    StopCondition::failed_updates(1),
)));
ga.run(&mut state)?;
println!("cut weight = {}", state.best_solution.objective);
```

## アルゴリズムの概要 { #algorithm-sketch }

各反復で次を行います。

1. 親を二つ選びます。
2. オペレータ `C` で交叉させます。
3. 子に突然変異をかけます。
4. 集団に挿入します。定員に達していれば最悪の個体を追い出します。

## コンストラクタ { #constructor }

```rust
GeneticAlgorithm::<P, C>::new(
    stop_condition: StopCondition,
    population_size: usize,
    crossover: C,
    mutation: Box<dyn Heuristic<P>>,
    parent_selection: ParentSelection,
) -> Self
```

`C: Crossover<P>` と `P::Solution: Distance` が必要です (型の境界が `Heuristic<P>` の impl にあるので、
`Tournament` 選択を使う場合でも距離の impl は必要です)。

`population_size < 2` なら panic します。

## HEA 風の初期化 { #hea-style-init }

```rust
.with_init_improvement(op: Box<dyn Heuristic<P>>)
```

ランダムな初期個体はすべて、サブランのクローンとマージのパターンで `op` にも通されます。
これを `TabuSearch` の突然変異オペレータと組み合わせると、Galinier と Hao の Hybrid Evolutionary Algorithm (HEA) を再現できます。
集団は最初の `run_once` まで作られないので、それより前ならいつ設定してもかまいません。

`clear()` は集団を捨てます。集団は `run` の後の最初の `run_once` で作り直されます。

## 親選択 { #parent-selection }

コンストラクタの最後の引数で、次のどれかです。

```rust
pub enum ParentSelection {
    Tournament,                          // 二回のバイナリトーナメント。設定での既定
    DistantTopK { top_k: usize },        // A をランダムに選び、B を距離の上位 k 個から選ぶ
    BiasedFitness { n_elite: usize, n_closest: usize },  // コストと多様性で順位付けする
}
```

`BiasedFitness` は、コストの順位と多様性の順位を Vidal の方法で混ぜた値で集団を順位付けし、その順位でバイナリトーナメントを行います。
[HybridGeneticSearch](hgs.md) が使うのと同じ方式です。生き残りの選択も変わり、同じ順位で刈り込み、クローンを先に追い出します。
この二つはセットです。親を多様性で順位付けしながらコストだけで追い出すと、集団は一回の追い出しごとに収束していってしまうからです。
解に `Evaluate` が必要ですが、ここにあるすべての問題が実装しています。

`n_closest` は、ある個体の多様性への寄与を求めるときに平均する最近傍の個体数で、Vidal の `nbClose` です。
`n_elite` はコストの順位だけで生かしておく個体数で、Vidal の `nbElite` です。これは `1 - n_elite / N` として混合に入るので、
**`n_elite` 以下の大きさの集団はコストだけで順位付けされます**。多様性の半分に 0 が掛かり、この戦略は本来置き換えるはずのコストだけの選択になってしまいます。
そのためベンチマークは `n_elite >= population_size` の設定を拒否します。Rust から使う場合は同じ組み合わせが合法で、何も言わずに退化します。
enum の種類が両方を持つので既定値はありません。設定では `4` と `5` が既定です。

`DistantTopK` は `P::Solution: Distance` を必要とし、遠い親を好むことで多様性を促します。

## 置き換え { #replacement }

最悪個体の置き換えです。集団が満員のとき、子が最悪の個体より厳密に良い場合に限りそれを置き換えます。
最良の個体は集団を一回走査して求めます。これは一世代の交叉と突然変異に比べれば小さなコストです。

## Crossover トレイト { #crossover-trait }

```rust
pub trait Crossover<P: ProblemTrait> {
    fn crossover(
        &mut self,
        prob: &P,
        sol1: &P::Solution,
        sol2: &P::Solution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Result<P::Solution, OptError>;
}
```

`&mut self` なので、状態を持つオペレータ (内側のヒューリスティクスを走らせる
[`SubProblemBasedCrossover`](../../api/optopus/heuristic/struct.SubProblemBasedCrossover.html) など)
が呼び出しをまたいで可変の状態を保てます。RNG は明示的に渡されるので、シード付きの実行は再現可能なままです。

## SubProblemBasedCrossover

任意の `P: SubProblemExtractable` に使える汎用の交叉です。

1. `extract_sub_problem(sol1, sol2)`。両親で一致している変数を固定し、一致しない変数で部分インスタンスを作ります。
2. `sub_heuristic.run(...)` が部分インスタンスを一から解きます。
3. `lift_solution(sol1, sol2, sub_solution)` が全体の解を組み立て直します。

```rust
let crossover = SubProblemBasedCrossover {
    sub_heuristic: Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
        StopCondition::failed_updates(1),
    )),
};
```

MaxCut、QUBO、SAT、Vertex Cover、Formula が実装しています。

## ベンチマーク設定 { #benchmark-config }

```toml
[[heuristics]]
kind = "GeneticAlgorithm"
population_size = 20         # 必須、2 以上
crossover_kind = "Uniform"   # 任意、既定値は問題ごと (下を参照)
parent_selection = "Tournament"  # 任意、Tournament (既定) | DistantTopK | BiasedFitness
parent_top_k = 5             # parent_selection = "DistantTopK" のとき必須
n_elite = 4                  # 任意 (値は既定値)、BiasedFitness のみ、population_size 未満
n_closest = 5                # 任意 (値は既定値)、BiasedFitness のみ
[heuristics.stop_condition]
max_duration_secs = 30.0

[[heuristics.steps]]         # steps[0] = 突然変異 (必須)
kind = "TabuSearch"
neighbor = "Flip"
tabu_tenure = [5, 150]
[heuristics.steps.stop_condition]
max_iteration = 2_000

[[heuristics.steps]]         # steps[1] = init_improvement (任意、HEA のパターン)
kind = "LocalSearch"
neighbor = "Flip"
[heuristics.steps.stop_condition]
max_failed_update = 1
```

`crossover_kind` の既定は `"Uniform"` で、TSP と CVRP では `"Order"`、JobShop では `"Ppx"` です。
MaxCut はさらに `"SubProblem"` を受け付けます。これは一致しない変数からなる部分 MaxCut を内部の上限付き BLS で解く memetic な組み換えです
([SubProblemBasedCrossover](#subproblembasedcrossover) を参照)。

## 参考文献 { #references }

- Holland, J. H. *Adaptation in Natural and Artificial Systems*. University of
  Michigan Press, 1975.
- Goldberg, D. E. *Genetic Algorithms in Search, Optimization, and Machine
  Learning*. Addison-Wesley, 1989.
- Galinier, P. and Hao, J.-K. "Hybrid Evolutionary Algorithms for Graph
  Coloring." Journal of Combinatorial Optimization, 3(4), 379-397, 1999.
- Vidal, T. et al. "A Hybrid Genetic Algorithm for Multidepot and Periodic
  Vehicle Routing Problems." *Operations Research*, 60(3), 611-624, 2012.
  (Biased fitness です。)
