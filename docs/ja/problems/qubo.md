# QUBO

**API:** [`Qubo`](../../api/optopus/problem/qubo/struct.Qubo.html)

Quadratic Unconstrained Binary Optimization (QUBO) は、`n` 個の二値変数 `x ∈ {0,1}^n` 上の二次多項式を最小化する問題で、
対称な係数行列 `Q` で表されます。このクレートは上三角 (`i ≤ j`) だけを保持します。
二値では `x[i]² = x[i]` なので、対角成分 `Q[i][i]` は `x[i]` の一次の項になります。

```text
E(x) = Σ_{i ≤ j} Q[i][j] · x[i] · x[j]    (x ∈ {0,1}^n)
```

QUBO は量子および古典のイジングマシン型アニーラが受け付ける標準的な入力形式です。また制約がないので、
ほかの多くの組合せ問題 (MaxCut, Graph Coloring, ...) が帰着される形でもあります。
このクレートでは `Qubo` をそうした帰着の特別扱いはせず、独立した汎用の問題型として扱います。

## 例 { #example }

探索を実行し、最小化する割り当てを読み出します。

```rust
use optopus::prelude::*;

let qubo = Qubo::from_entries([
    (0, 0, -1), // 対角 = 一次の項
    (0, 1, 1),
    (1, 2, 2),
    (0, 2, 3),
]);
let mut state = SearchState::new(&qubo);
LocalSearch::<QuboFlipNeighbor>::new(StopCondition::iterations(10_000))
    .run(&mut state)
    .unwrap();

let sol = &state.best_solution;
println!("energy = {}", sol.objective);
println!("assignment = {:?}", sol.x); // sol.x[i] は見つかった最小点での x[i] の値
```

`Qubo::from_entries` はインスタンスの作り方の一つです。`Qubo::new()` から始めて、`set_q` (上書き) や `add_q` (加算)
を少しずつ呼んでもかまいません。

## 解 { #solution }

[`QuboSolution`](../../api/optopus/problem/qubo/struct.QuboSolution.html) は
上の定義の割り当て `x` (`x ∈ {0,1}^n`) を持ちます。

## 近傍 { #neighbors }

| 型 | TOML の `neighbor` | move |
|---|---|---|
| `QuboFlipNeighbor` | `"Flip"` | 変数を一つ反転する。`iter + 1`。 |
| `QuboSwapNeighbor` | `"Swap"` | 値の異なる二つの変数を入れ替える。`iter + 2`。 |

## 交叉 { #crossover }

- `QuboUniformCrossover`。変数ごとにランダムに親を選びます。
- `Qubo` は `SubProblemExtractable` を実装しています。両親で一致している変数は固定され、
  その寄与は部分 QUBO の一次の項に畳み込まれるので、部分問題はそれだけで完結します。

## ファイル形式 { #file-format }

```text
N M
i j v
i j v
...
```

- `N` は変数の数、`M` はエントリの数です。
- インデックスは 1 始まりで、内部で 0 始まりに変換されます。
- `i == j` の行は一次 (対角) の係数を表します。
- 重複したエントリは `set_q` と同じく、最後に書いたものが残ります。

```rust
use optopus::prelude::*;

let qubo = Qubo::load_file("data/instances/qubo/sample.qubo")?;
```

## 参考文献 { #references }

- Kochenberger, G., Hao, J.-K., Glover, F., Lewis, M., Lü, Z., Wang, H., and
  Wang, Y. "The Unconstrained Binary Quadratic Programming Problem: A Survey."
  Journal of Combinatorial Optimization, 28(1), 58-81, 2014.
- Beasley, J. E. "Obtaining Test Problems via Internet." *Journal of Global
  Optimization*, 8(4), 429-433, 1996. (OR-Library。同梱の `bqp` インスタンス集合の出典です。)
