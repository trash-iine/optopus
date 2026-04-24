# Profiling workflow

`samply` を使って `examples/prof_*.rs` を実行し、パフォーマンス解析用の
トレースを取得するためのディレクトリ。

各 `prof_*.rs` は 5 秒前後で完了するよう `StopCondition::duration` または
`Restart` でラップされており、サンプル密度が十分取れる構成になっている。

## ベースライン取得

初回および最適化前の基準点として、9 本すべてを一括キャプチャする:

```sh
cargo build --profile profiling --examples

mkdir -p profiles/baseline
for ex in prof_ls_maxcut_flip prof_ts_maxcut_flip prof_sa_maxcut \
         prof_beam_maxcut prof_ls_tsp prof_lkh_tsp \
         prof_ls_sat prof_ls_formula prof_ga_maxcut; do
  samply record --save-only \
    -o "profiles/baseline/${ex}.profile.json.gz" \
    "./target/profiling/examples/${ex}"
done
```

## 差分取得（最適化後）

最適化検証用トレースは `profiles/current/` に出力する:

```sh
mkdir -p profiles/current
samply record --save-only \
  -o profiles/current/<example>.profile.json.gz \
  ./target/profiling/examples/<example>
```

## 閲覧

```sh
samply load profiles/baseline/<example>.profile.json.gz
```

## 対応表

| 例                       | 対象                                         |
|--------------------------|----------------------------------------------|
| `prof_ls_maxcut_flip`    | LocalSearch × MaxCutFlip  (G22)              |
| `prof_ts_maxcut_flip`    | TabuSearch  × MaxCutFlip  (G22)              |
| `prof_sa_maxcut`         | SA × MaxCut Flip + Swap   (G22)              |
| `prof_beam_maxcut`       | BeamSearch × MaxCutFlip   (G22)              |
| `prof_ls_tsp`            | LS × TSP TwoOpt + Relocate (berlin52)        |
| `prof_lkh_tsp`           | Lin-Kernighan (Helsgaun)  (berlin52)         |
| `prof_ls_sat`            | LS × SAT Flip + Swap      (n=500)            |
| `prof_ls_formula`        | LS × FormulaProblem Flip + Swap (n=60)       |
| `prof_ga_maxcut`         | GeneticAlgorithm × MaxCut (G22)              |

## 備考

- `profiles/baseline/` および `profiles/current/` は `.gitignore` で除外。
  トレース自体はローカル成果物としてのみ扱う。
- `cargo build --profile profiling` は `Cargo.toml` の
  `[profile.profiling]` (release + `debug = "line-tables-only"`) で
  シンボル解決に必要な行情報のみを含むバイナリを生成する。
