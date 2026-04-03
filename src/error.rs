//! Error types for the optopus library.

/// The unified error type for optopus.
///
/// Represents errors that can occur during heuristic execution or problem parsing.
#[derive(Debug, thiserror::Error)]
pub enum OptError {
    /// A user-facing configuration error (e.g., invalid benchmark settings).
    #[error("Config error: {0}")]
    Config(String),

    /// An I/O error (e.g., file not found or read failure).
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// A parse error (e.g., malformed input format).
    #[error("Parse error: {0}")]
    Parse(String),

    /// An invalid search state (e.g., empty neighborhood).
    #[error("Invalid state: {0}")]
    InvalidState(String),
}
