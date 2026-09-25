# MaxSAT

**API:** [`Sat`](../../api/optopus/problem/sat/struct.Sat.html)

`n` 個のブール変数 `x ∈ {0,1}^n` 上の命題論理式が連言標準形 (CNF) で与えられます。これは節 `C_1, ..., C_m`
の連言で、各節はリテラル (変数またはその否定) の選言です。`x` が充足する節の数を最大化します。

```text
maximize  Σ_{k=1}^{m} [C_k(x) = true]           (x ∈ {0,1}^n, clauses C_1..C_m)
```

MaxSAT は古典的な (判定問題としての) SAT を緩めたものです。すべての節が成り立つことを要求するのではなく
(制約が多すぎる式ではそれは不可能かもしれません)、できるだけ多くの節を充足する割り当てを求めます。
このクレートは一貫して MaxSAT を実装しています。たまたま完全に充足可能なインスタンスも、充足節の数を `m` まで最大化することで解かれるだけです。

## 例 { #example }

探索を実行し、割り当てと、それがいくつの節を充足するかを読み出します。

```rust
use optopus::prelude::*;

let mut sat = Sat::new(3);
sat.add_clause([1, -2, 3]); // (x1 ∨ ¬x2 ∨ x3)。リテラルは符号付きの 1 始まり
sat.add_clause([-1, 2]);
sat.add_clause([1, 2, 3]);

let mut state = SearchState::new(&sat);
LocalSearch::<SatFlipNeighbor>::new(StopCondition::iterations(10_000))
    .run(&mut state)
    .unwrap();

let sol = &state.best_solution;
println!("{} / {} clauses satisfied", sol.n_satisfied, sat.n_clauses());
for (i, &v) in sol.x.iter().enumerate() {
    println!("x{} = {v}", i + 1); // DIMACS と add_clause の慣習に合わせて 1 始まり
}
```

## 解 { #solution }

[`SatSolution`](../../api/optopus/problem/sat/struct.SatSolution.html) は
上の定義の割り当て `x` (`x ∈ {0,1}^n`) と、`Σ_{k=1}^{m} [C_k(x)=true]` である
`n_satisfied` を持ちます。

## 近傍 { #neighbors }

| 型 | move | 反復コスト |
|---|---|---|
| [`SatFlipNeighbor`](../../api/optopus/problem/sat/struct.SatFlipNeighbor.html) | 変数を一つ反転する。 | `iter + 1` |
| [`SatSwapNeighbor`](../../api/optopus/problem/sat/struct.SatSwapNeighbor.html) | 二つの変数を入れ替える。 | `iter + 2` |

## 交叉 { #crossover }

- `SatUniformCrossover`。変数ごとにランダムに親を選びます。
- `Sat` は `SubProblemBasedCrossover` のために `SubProblemExtractable` を実装しています。

## ファイル形式 (DIMACS CNF) { #file-format-dimacs-cnf }

インデックスの慣習に注意してください。`add_clause` とファイル形式は符号付きの 1 始まりのリテラルを使います
(正は肯定リテラル、負は否定)。

```text
c optional comment lines
p cnf N M
1 -2 3 0
-1 2 0
...
```

- `N` は変数の数、`M` は節の数です。
- 各節の行は空白区切りの符号付き整数の列で、`0` で終わります。符号が極性を、絶対値が変数のインデックス (1 始まり) を表します。

```rust
use optopus::prelude::*;

let sat = Sat::load_file("data/instances/sat/example.cnf")?;
```

## 参考文献 { #references }

- "Satisfiability Suggested Format." DIMACS Challenge technical report, 1993.
  (DIMACS CNF のファイル形式を定義しています。)
- Hoos, H. H. and Stützle, T. "SATLIB: An Online Resource for Research on
  SAT." In SAT 2000, pp. 283-292. IOS Press, 2000. (`uf` インスタンス集合の出典です。)
