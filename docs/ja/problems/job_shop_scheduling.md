# Job Shop Scheduling

**API:** [`JobShopScheduling`](../../api/optopus/problem/job_shop_scheduling/struct.JobShopScheduling.html)

Job Shop Scheduling は、最もよく研究されている強 NP 困難なスケジューリング問題の一つです。

`n_jobs` 個のジョブと `n_machines` 台の機械が与えられます。各ジョブ `j` は作業 `(machine, duration)` の決まった順序付きの列で、
作業はその順番どおりに指定の機械で実行しなければなりません。作業は同じジョブの直前の作業が終わるまで始められず、
機械は一度に一つの作業しか処理できません。ジョブ `j` の `k` 番目の作業の完了時刻を `C_{j,k}`、処理時間を `p_{j,k}` とします。
メイクスパンは、どこかで最後の作業が終わる時刻です。メイクスパンを最小化します。

```text
minimize  max_j C_{j,last}
subject to  C_{j,k} ≥ C_{j,k-1} + p_{j,k}                 (ジョブ内の先行関係)
            同じ機械上の作業は重ならない                    (機械の容量)
```

## 例 { #example }

探索を実行し、復号したスケジュールを読み出します。

```rust
use optopus::prelude::*;

let inst = JobShopScheduling::new(
    "tiny".to_string(),
    /* n_machines = */ 2,
    vec![
        vec![(0, 2), (1, 3)],   // ジョブ 0: M0(2) → M1(3)
        vec![(1, 1), (0, 4)],   // ジョブ 1: M1(1) → M0(4)
    ],
);
let mut state = SearchState::new(&inst);
LocalSearch::<JobShopSwapNeighbor>::new(StopCondition::iterations(10_000))
    .run(&mut state)
    .unwrap();

let sol = &state.best_solution;
println!("makespan = {}", sol.objective);
println!("operation order = {:?}", sol.operations); // 復号前の重複を許す置換
println!("completion times = {:?}", sol.completion_times); // 上の各位置の完了時刻
```

## 解 { #solution }

解は長さ `n_jobs * n_machines` の重複を許す置換として符号化されます。列の中でジョブ `j` が `k` 回目に現れる位置が、
そのジョブの `k` 番目の作業を表します。これを左詰めの semi-active スケジューリングで復号し、上の定義の完了時刻 `C_{j,k}` を得ます。
[`JobShopSolution`](../../api/optopus/problem/job_shop_scheduling/struct.JobShopSolution.html)
はその符号化を `operations` として、位置ごとに復号した完了時刻を `completion_times` として
(`completion_times[pos]` は位置 `pos` の作業の `C_{j,k}`)、最小化するメイクスパン `max_j C_{j,last}` を
`objective` として持ちます。

## 近傍 { #neighbors }

| 型 | move | 反復コスト |
|---|---|---|
| `JobShopSwapNeighbor` | `operations[i]` と `operations[i+1]` を入れ替える。 | `iter + 1` |
| `JobShopRelocateNeighbor` | `operations[i]` を取り除き、別の位置に挿入し直す。 | `iter + 1` |

## 交叉 { #crossover }

交叉 `JobShopPpxCrossover` は Precedence-Preserving Crossover (PPX) です。子の各位置で親をランダムに選び、
その親のまだ使っていない作業のうち最も左のものを追加します。両親は同期して進むので、子も先行関係を満たす重複を許す置換のままです。

## ファイル形式 (Taillard / OR-Library) { #file-format-taillard-or-library }

```text
n_jobs n_machines
m d m d m d ...      (ジョブごとに n_machines 組の (machine, duration)、1 行に 1 ジョブ)
m d m d m d ...
...
```

- 機械のインデックスは 0 始まりです。
- 空行と `#` で始まるコメント行は無視されます。
- ファイルは行単位で厳密に読むのではなくトークンに分けて読むので、行内や行間の空白は自由です。

```rust
use optopus::prelude::*;

let inst = JobShopScheduling::load_file("data/instances/jssp/ft06.txt")?;
```

## 参考文献 { #references }

- Fisher, H. and Thompson, G. L. "Probabilistic Learning Combinations of
  Local Job-Shop Scheduling Rules." In Industrial Scheduling, pp. 225-251.
  Prentice-Hall, 1963. (古典的な `ft06` インスタンスの出典です。)
- Taillard, E. "Benchmarks for Basic Scheduling Problems." *European Journal
  of Operational Research*, 64(2), 278-285, 1993.
- インスタンスの出典とライセンスは
  [`data/instances/README.md`](https://github.com/trash-iine/optopus/blob/main/data/instances/README.md) を参照してください。
