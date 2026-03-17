/// Optopus のエラー型。
///
/// ヒューリスティックや問題定義から発生するエラーを統一した型で表現します。
#[derive(Debug, thiserror::Error)]
pub enum OptError {
    /// IO エラー（ファイル読み込みなど）
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// パースエラー（入力フォーマットが不正）
    #[error("Parse error: {0}")]
    Parse(String),

    /// 探索状態が無効（近傍が空など）
    #[error("Invalid state: {0}")]
    InvalidState(String),
}
