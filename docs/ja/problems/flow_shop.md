# Permutation Flow Shop

**API:** [`FlowShop`](../../api/optopus/problem/flow_shop/struct.FlowShop.html)

Permutation Flow Shop は、`n_jobs` 個のジョブを `n_machines` 台の機械でスケジュールする問題です。どのジョブも機械 0 から同じ順番で機械を回ります。
すべての機械はジョブを一つの共通の順序で処理し、その順序が解です。機械は一度に一つのジョブしか処理できず、ジョブは一度に一台の機械にしか載りません。
ジョブ `j` が機械 `i` で費やす時間を `p(i, j)`、順序 `π` の `k` 番目のジョブが機械 `i` を離れる時刻を `C(i, k)` として、メイクスパンを最小化します。

```text
C(i, k) = max(C(i-1, k), C(i, k-1)) + p(i, π_k)      C(·, 0) = C(0, ·) = 0
minimize  C(m, n)
```

機械が 3 台以上で強 NP 困難です。標準のベンチマークは Taillard の 120 インスタンスで、難しいのは最大のものではなく中間の 50 ジョブ 20 機械と 100 ジョブ 20 機械の組です。

## 例 { #example }

```rust
use optopus::prelude::*;

let fs = FlowShop::load_file("data/instances/flow_shop/taillard/ta051.txt")?;
println!("NEH = {}", fs.neh().objective);

let mut state = SearchState::new_with_seed(&fs, 42);
iterated_greedy_for_flow_shop(
    StopCondition::duration(std::time::Duration::from_secs(10)),
    /* removal_count      = */ 4,
    /* temperature_factor = */ 0.4,
)
.run(&mut state)?;
println!("makespan = {}", state.best_solution.objective);
println!("sequence = {:?}", state.best_solution.sequence);
```

この問題は [Iterated Greedy](../heuristics/iterated_greedy.md) のために作られています。
`fs.neh()` は Nawaz、Enscore、Ham の構築ヒューリスティクスで、単独でも使えます。

## 解 { #solution }

[`FlowShopSolution`](../../api/optopus/problem/flow_shop/struct.FlowShopSolution.html) は、最初に処理するジョブを先頭にしたジョブの置換 `sequence` と、そのメイクスパン `objective` を持ちます。
自前の順序は `FlowShop::solution_from_sequence` で解に包めます。

## head と tail { #heads-and-tails }

この問題のあらゆる価格付けは Taillard の高速化から来ています。順序を任意の位置で前半と後半に分けます。
前半の head は、その最後のジョブが各機械を離れる時刻です。後半の tail は、その最初のジョブが各機械で始まってから、その機械が働き続ける時間です。
メイクスパンは機械ごとの head と tail の和の最大値で、どちらの側も反対側に依存しません。
すべての分割位置について両方を O(nm) で表にしておけば、任意の位置へのジョブの挿入や除去は O(m) で、`n + 1` 箇所すべてへの挿入を試すのは O(nm) で済みます。
これが NEH を O(n²m) にし、Iterated Greedy の 1 反復を安くしています。

## 近傍 { #neighbors }

| 型 | 設定 | 移動 | 走査コスト |
|---|---|---|---|
| `FlowShopInsertNeighbor` | `Relocate` | ジョブを一つ取り出して別の位置に戻す。 | `(n−1)²` 手すべてで O(n²m) |
| `FlowShopSwapNeighbor` | `Swap` | 二つの位置のジョブを入れ替える。 | head と tail から 1 手 O((j−i)·m) |

フローショップの文献が頼りにするのは挿入近傍です。どちらの移動もランダムな近傍を O(nm) で引きます。

## Ruin and recreate { #ruin-and-recreate }

`FlowShop` は、ジョブを要素、順序を唯一のコンテナとして [`Ruinable`](../traits.md) を実装しています。TSP の巡回路と同じ形です。
部分解が head と tail を保持しているので、トレイトの位置ごとの挿入コストはスケジュール全体ではなく O(m) です。
[Iterated Greedy](../heuristics/iterated_greedy.md) と [ALNS](../heuristics/alns.md) はこの上で、`iterated_greedy_for_flow_shop` と `alns_for_flow_shop` を通して動きます。
どちらも探索を `FlowShopInsertionDescent` と組みます。これは Ruiz と Stützle の全体挿入降下で、すべてのジョブをランダムな順に取り出してメイクスパンが最小の位置に戻すことを、一周で何も変わらなくなるまで繰り返します。

## 交叉 { #crossover }

`FlowShopOrderCrossover` は Order Crossover (OX) です。第 1 親の順序の一区間をその位置に残し、残りのジョブを第 2 親が処理する順で埋めます。

## ファイル形式 { #file-format }

```text
n_jobs n_machines
p(0,0) p(0,1) ... p(0,n-1)        機械ごとに処理時間を 1 行
...
```

Taillard の形式から seed と上下界を除いたものです。Taillard の 120 インスタンスは `data/instances/flow_shop/taillard/ta001.txt` から `ta120.txt` として同梱しており、
`data/instances/scripts/fetch_pfsp.sh` が公開されている seed から再生成します。

## 参考文献 { #references }

- Taillard, E. "Some efficient heuristic methods for the flow shop sequencing
  problem." *European Journal of Operational Research*, 47(1), 65-74, 1990.
  (head と tail。)
- Taillard, E. "Benchmarks for basic scheduling problems." *European Journal
  of Operational Research*, 64(2), 278-285, 1993.
- Nawaz, M., Enscore, E. E. and Ham, I. "A heuristic algorithm for the
  m-machine, n-job flow-shop sequencing problem." *Omega*, 11(1), 91-95, 1983.
- Ruiz, R. and Stützle, T. "A simple and effective iterated greedy algorithm
  for the permutation flowshop scheduling problem." *European Journal of
  Operational Research*, 177(3), 2033-2049, 2007.
- インスタンスの出典とライセンスは [`data/instances/README.md`](https://github.com/trash-iine/optopus/blob/main/data/instances/README.md) を参照してください。
