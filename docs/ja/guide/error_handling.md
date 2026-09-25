# エラー処理

**API:** [`OptError`](../../api/optopus/error/enum.OptError.html)

Optopus で失敗しうる操作はすべて `Result<_, OptError>` を返します。エラー型は
[`optopus::error`](../../api/optopus/error/index.html) で定義されています。

## 種類 { #variants }

| 種類 | 発生する場面 | よくある原因 |
|---|---|---|
| [`Config(String)`](../../api/optopus/error/enum.OptError.html#variant.Config) | 利用者向けの設定エラー。 | ベンチマーク TOML のフィールドが不正、必須パラメータの欠落、範囲外の値。 |
| [`Io(std::io::Error)`](../../api/optopus/error/enum.OptError.html#variant.Io) | `#[from]` で包まれ、`std::io` の層から上がってくる。 | ファイルが見つからない、権限がない、読み込み途中で EOF。 |
| [`TomlDe(toml::de::Error)`](../../api/optopus/error/enum.OptError.html#variant.TomlDe) | `#[from]` で包まれる。TOML のデシリアライズに失敗した。内側のエラーが行と列の情報を持っている。 | ベンチマーク設定の書式誤り、未知のヒューリスティクス `kind`、タグ付き `HeuristicConfig` の種類に必要なフィールドの欠落。 |
| [`TomlSer(toml::ser::Error)`](../../api/optopus/error/enum.OptError.html#variant.TomlSer) | `#[from]` で包まれる。値の TOML へのシリアライズに失敗した。 | `write_to_dir` で `BenchmarkReport` を書き出すとき。 |
| [`FileLoad { path, line, detail }`](../../api/optopus/error/enum.OptError.html#variant.FileLoad) | 構造化されたファイル読み込みエラー。`line == 0` は特定の行に結びつかないファイル全体のエラーを表す。 | TSPLIB / DIMACS / QUBO のローダが予期しないトークンに当たった。 |
| [`InvalidState(String)`](../../api/optopus/error/enum.OptError.html#variant.InvalidState) | 探索が実行時に矛盾した状態に達した。 | 近傍が空、範囲外のインデックスへの move。 |

## エラーで分岐する { #matching-on-errors }

```rust
use optopus::error::OptError;
use optopus::prelude::*;

match Qubo::load_file("instance.qubo") {
    Ok(prob) => { /* ヒューリスティクスを実行する */ }
    Err(OptError::FileLoad { path, line, detail }) => {
        eprintln!("Failed to parse {path} at line {line}: {detail}");
    }
    Err(OptError::Io(e)) => {
        eprintln!("I/O error: {e}");
    }
    Err(other) => {
        eprintln!("Unexpected error: {other}");
    }
}
```

## 独自のコードでは { #in-custom-code }

独自の問題やヒューリスティクスを実装するときは次のようにします。

- 実行時の不変条件違反には [`OptError::InvalidState(...)`](../../api/optopus/error/enum.OptError.html#variant.InvalidState) を返します
  (たとえば不正なインデックスを受け取った
  [`MoveToNeighbor::apply_to_solution`](../../api/optopus/trait_defs/trait.MoveToNeighbor.html#tymethod.apply_to_solution))。
- 独自のローダからは [`OptError::FileLoad { … }`](../../api/optopus/error/enum.OptError.html#variant.FileLoad) を返します。
  ファイル全体の問題なら `line == 0` にします。
  [`InstanceLines`](../../api/optopus/common/parse/struct.InstanceLines.html) を使うと、パスと行番号が埋まった状態で作れます。
- `std::io::Error` は `?` でそのまま伝播させます。
  [`OptError: From<io::Error>`](../../api/optopus/error/enum.OptError.html#trait-implementations) が導出されています。
