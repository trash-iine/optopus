# HybridGeneticSearchForVrp

**API:** [`HybridGeneticSearchForVrp`](../../api/optopus/heuristic/struct.HybridGeneticSearchForVrp.html)

[CVRP](../problems/vrp.md) 専用のヒューリスティクスです。Hybrid Genetic Search は、知られている中で最も強力な汎用の CVRP メタヒューリスティクスです。

## 例 { #example }

```rust
use optopus::prelude::*;

let vrp = Vrp::load_file("data/instances/vrp/demo16.vrp")?;
let mut state = SearchState::new(&vrp);

let mut hgs = HybridGeneticSearchForVrp::new(
    StopCondition::iterations(10_000),
    /* min_population_size = */ 25,   // μ
    /* generation_size     = */ 40,   // λ
    /* granularity         = */ 20,   // Γ、n - 1 で頭打ち
    /* target_feasible     = */ 0.2,
    /* restart_generations = */ Some(20_000),
);
hgs.run(&mut state)?;

let sol = &state.best_solution;
println!("total distance = {}", sol.distance);
for (vehicle, route) in sol.routes.iter().enumerate() {
    println!("vehicle {vehicle}: depot -> {route:?} -> depot");
}
```

自前の move 集合を持つので、`neighbor` の型パラメータは取りません。

## アルゴリズムの概要 { #algorithm-sketch }

表現は giant tour なので、個体は顧客の置換です。これを `split_giant_tour` でルートに復号します。
これはその置換に対して距離が最適な区切り位置を求める動的計画法です。復号が厳密なので、遺伝的オペレータは顧客の順序さえ正しくすればよいことになります。

各 `run_once` は子を一つ作ります。

1. 選択。両方の部分集団を合わせたものに対し、biased fitness でバイナリトーナメントを行います。
2. 交叉。二つの親の giant tour に Order Crossover (OX) をかけます。
3. 復号。現在の容量ペナルティの下で `split_giant_tour` を行います。
4. 局所探索。granular な降下で局所最適まで進めます。
5. 修復。実行不能な子は、確率 `repair_probability` (既定 0.5) でペナルティを 10 倍、次に 100 倍にして二回目の降下を受けます。
   それが成功すれば、元の子に加えて実行可能な複製も挿入されます。
6. 生き残り。子は実行可能か実行不能の部分集団に加わります。
   それぞれ `min_population_size + generation_size` まで増えたら `min_population_size` まで間引かれます。

### Biased fitness

コストだけで選ぶと、数百世代のうちに集団が一つの谷に潰れてしまいます。代わりに各個体を次で順位付けします。

```text
fitness = rank_cost / (N−1) + (1 − n_elite/N) · rank_diversity / (N−1)
```

`rank_diversity` は寄与の大きい順に個体を並べたもので、寄与とは部分集団の中の最も近い `n_closest` 個の個体との broken-pairs 距離の平均です
(既定は `n_elite = 4`、`n_closest = 5`)。
したがって解は、安いか、ほかと似ていないかのどちらかで居場所を得ます。クローン (ほかのメンバーとの距離が `0`) は常に最初に追い出されます。

broken-pairs 距離は、ルート上の隣が異なる顧客の割合で、ルートのラベルの付け替えや反転に対して不変です。
`VrpSolution` の [`Distance`](../../api/optopus/trait_defs/trait.Distance.html) impl は同じ数え方を対称化して使いますが、
ここでの biased fitness は Vidal が定義した形である方向付きの数え方で順位付けします。

### 二つの部分集団と適応的なペナルティ { #two-sub-populations-and-the-adaptive-penalty }

実行可能な個体と実行不能な個体は別々の部分集団に保たれ、容量ペナルティは 100 個の子ごとに調整し直されて、実行可能な割合を
`target_feasible` (既定 0.2) の近くに保ちます。実行可能な子が少なすぎれば 1.2 倍に上げ、多すぎれば 0.85 倍に下げ、両側に 0.05 の不感帯を持ちます。
この四つの数はすべて `with_penalty_adaptation` で設定します。

あえて実行可能な割合を低くして探索することが要点です。二つの良い実行可能解の間の最短経路は、たいてい実行不能な領域を通るからです。
ペナルティはインスタンスの需要 1 単位あたりの平均距離から始まるのでスケールに依存せず、その両側数桁の範囲に制限されます。

そのため HGS は `VrpSolution` ではなく自前の個体を持ちます。`Vrp::penalty_weight()` は、どの最適解も実行可能になるほど大きな固定定数だからです。
探索状態に書き込むときは `Vrp::solution_from_routes` で変換し直すので、報告される目的関数値はほかのどのヒューリスティクスとも比較できます。

### Granular な局所探索 { #granular-local-search }

各顧客 `u` について、最も近い `granularity` 人の顧客だけを move の相手として考えます。move 集合は次のとおりです。

| move | 範囲 |
|---|---|
| 1〜2 人の顧客からなる区間を移す (反転してもよい) | ルート間とルート内 |
| 1〜2 人の顧客からなる区間を交換する | ルート間 |
| 2-opt (部分経路を反転する) | ルート内 |
| 2-opt\* (ルートの末尾を交換する) | ルート間 |
| 区間を使っていない車両に移す | |

どの move も端点の距離から O(1) で評価し、最初に見つかった改善 move を適用します。降下はドライバが与えるペナルティで
`distance + penalty · overload` を対象に走ります。`VrpRelocateNeighbor` の系統を再利用しないのはこのためです。

## コンストラクタ { #constructor }

```rust
HybridGeneticSearchForVrp::new(
    stop_condition: StopCondition,
    min_population_size: usize,        // μ
    generation_size: usize,            // λ
    granularity: usize,                // Γ
    target_feasible: f64,
    restart_generations: Option<u64>,
) -> Self
```

妥当な既定値は `μ = 25`、`λ = 40`、`Γ = 20`、`target_feasible = 0.2`、
`restart_generations = Some(20_000)` です。

Vidal が調整するそれ以外のものはすべて既定値が公開された builder なので、`new` の引数は六つのままです。

| builder | 設定するもの | 既定値 |
|---|---|---|
| `with_elite(n_elite, n_closest)` | biased fitness における Vidal の `nbElite` と `nbClose` | `4` / `5` |
| `with_penalty_adaptation(period, increase, decrease, tolerance)` | ペナルティ更新の間の子の数、二つの係数、`target_feasible` の周りの不感帯 | `100` / `1.2` / `0.85` / `0.05` |
| `with_repair_probability(p)` | 実行不能な子を修復もする確率 | `0.5` |
| `with_initial_population(factor, nearest_neighbor_every)` | μ の倍数で表した初期個体数と、そのうち何個に一個を最近傍巡回路にするか | `4` / `4` |
| `with_descent_passes(p)` | granular な降下が一つの子に使ってよいパス数 | `64` |

既定値は探索の関連定数 (`HybridGeneticSearchForVrp::DEFAULT_N_ELITE` など) です。builder は Rust からしか使えず、TOML からは使えません。

最初の個体は `state.solution` から作られるので、HGS を `Sequential`、`Iterated`、`Restart` の中で組み合わせると現在解が引き継がれます。
一方リスタート (`restart_generations` 世代改善がなかった後) は一から作り直します。抜け出せなかった谷に戻ってしまわないためです。
`state.best_solution` はリスタートをまたいで保持されます。

`clear()` は両方の部分集団を空にし、実行可能な割合と停滞のカウンタをリセットし、容量ペナルティを初期値に戻します。
そのため新しいエピソードは、調整済みのペナルティを引き継がずに導き直します。

## ベンチマーク設定 { #benchmark-config }

```toml
[[heuristics]]
kind = "HybridGeneticSearch"
min_population_size = 25
generation_size = 40
granularity = 20
target_feasible = 0.2
restart_generations = 20000
[heuristics.stop_condition]
max_duration_secs = 30.0
```

フィールドはすべて任意です。レポートの受理カウンタには子の実行可能な割合が入ります。適応的なペナルティが操作しているのはこの値です。

## 計測した品質 { #measured-quality }

1 回 30 秒、3 回実行、シード 42、`μ=25 λ=40 Γ=20` で、CVRPLIB の最良既知解 (`data/instances/scripts/fetch_cvrp.sh`) と比較しました。
ALNS は同じ予算の `AdaptiveLargeNeighborhoodSearch` です。

| インスタンス | BKS | ALNS の最良 | HGS の最良 | ALNS の差 | HGS の差 |
|---|---|---|---|---|---|
| X-n101-k25 | 27591 | 27597 | 27597 | +0.02% | +0.02% |
| X-n195-k51 | 44225 | 44334 | 44506 | +0.25% | +0.64% |
| X-n502-k39 | 69226 | 69872 | 70025 | +0.93% | +1.15% |

この予算では HGS と [ALNS](alns.md) は互角で、X インスタンス 10 本でどの差も 0.7% 未満です。CVRP ではどちらを既定にしても妥当です。

再現には `data/benchmarks/vrp/hgs_{small,medium,large}.toml` を使います。

## 参考文献 { #references }

- Vidal, T., Crainic, T. G., Gendreau, M., Lahrichi, N., and Rei, W. "A Hybrid
  Genetic Algorithm for Multidepot and Periodic Vehicle Routing Problems."
  Operations Research, 60(3), 611-624, 2012.
- Vidal, T. "Hybrid Genetic Search for the CVRP: Open-Source Implementation and
  SWAP\* Neighborhood." Computers & Operations Research, 140, 105643, 2022.
- Prins, C. "A Simple and Effective Evolutionary Algorithm for the Vehicle
  Routing Problem." Computers & Operations Research, 31(12), 1985-2002, 2004.
