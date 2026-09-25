# クイックスタート

このページでは、このライブラリの最小の使い方を順に説明します。

## インストール { #install }

`Cargo.toml` に Optopus を追加します。

```toml
[dependencies]
optopus = { git = "https://github.com/trash-iine/optopus" }
```

## サンプルを実行する { #run-an-example }

```bash
cargo run --example max_cut
```

## メモリ上の MaxCut { #in-memory-maxcut }

```rust
use optopus::prelude::*;

// 1. 辺のリストから問題インスタンスを作る。
let mc = MaxCut::new(Graph::from_edges([
    (0, 1, 1.0),
    (0, 2, 1.0),
    (1, 2, 1.0),
]));

// 2. 探索状態を初期化する (初期解はランダム)。
let mut state = SearchState::new(&mc);

// 3. ヒューリスティクスを設定して実行する。
let mut ls = LocalSearch::<MaxCutFlipNeighbor>::new(
    StopCondition::iterations(1_000_000),
);
ls.run(&mut state).unwrap();

// 4. 最良の結果を読み出す。
println!("best cut = {}", state.best_solution.objective);
```

## ファイルからインスタンスを読み込む { #loading-instances-from-files }

各問題には `Result<Self, optopus::error::OptError>` を返すローダが付いています。

```rust
use optopus::prelude::*;

// MaxCut と Vertex Cover は共通の Graph ローダを使う (形式は `N M / i j w`)。
let mc = MaxCut::new(Graph::load_from_file("data/instances/max_cut/G1")?);

// QUBO ローダ (形式は `N M / i j v`、1 始まり):
let qubo = Qubo::load_file("data/instances/qubo/sample.txt")?;

// MaxSAT ローダ (DIMACS CNF):
let sat = Sat::load_file("data/instances/sat/example.cnf")?;

// TSP ローダ (TSPLIB):
let tsp = Tsp::load_file("data/instances/tsp/burma14.tsp")?;

// Job Shop Scheduling ローダ (Taillard / OR-Library):
let jssp = JobShopScheduling::load_file("data/instances/jssp/ft06.txt")?;

// CVRP ローダ (CVRPLIB):
let vrp = Vrp::load_file("data/instances/vrp/X-n101-k25.vrp")?;
```

各ローダのファイル形式は、対応する問題のページに書かれています。

## 次に読むもの { #what-to-read-next }

- [基本概念](concepts.md)。設計方針、三つのユースケース、主要なパターン。
- [問題](problems/README.md)。組み込みの各問題が提供するもの。
- [ヒューリスティクス](heuristics/README.md)。アルゴリズムの選び方。
- [独自の問題](guide/custom_problem.md)。三つのトレイトを実装すれば、独自の問題が
  `LocalSearch` とメタヒューリスティクスで動きます。残りは任意のトレイトを一つ足すごとに一つずつ使えるようになります。
