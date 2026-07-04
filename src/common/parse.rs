//! Shared scaffold for line-oriented instance-file parsers.
//!
//! Every instance loader needs the same plumbing: open the file, iterate lines
//! with 1-based numbering, and wrap IO/token failures in
//! [`OptError::FileLoad`] with path and line context. [`InstanceLines`] holds
//! that plumbing in one place; the per-format parsing stays in each problem's
//! `load_file`.

use crate::error::OptError;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Line-by-line reader for instance files that converts failures into
/// [`OptError::FileLoad`] with path/line context.
#[derive(Debug)]
pub struct InstanceLines {
    path: String,
    lines: std::io::Lines<BufReader<File>>,
    line_num: usize,
}

impl InstanceLines {
    /// Opens `path`. Open failures are reported as `FileLoad` at line 0.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, OptError> {
        let path_display = path.as_ref().display().to_string();
        let file = File::open(path.as_ref()).map_err(|e| OptError::FileLoad {
            path: path_display.clone(),
            line: 0,
            detail: format!("failed to open file: {e}"),
        })?;
        Ok(Self {
            path: path_display,
            lines: BufReader::new(file).lines(),
            line_num: 0,
        })
    }

    /// Builds a `FileLoad` error at the current line.
    pub fn err(&self, detail: impl Into<String>) -> OptError {
        self.err_at(self.line_num, detail)
    }

    /// Builds a `FileLoad` error at an explicit line (for parsers that buffer
    /// tokens with their line numbers before interpreting them).
    pub fn err_at(&self, line: usize, detail: impl Into<String>) -> OptError {
        OptError::FileLoad {
            path: self.path.clone(),
            line,
            detail: detail.into(),
        }
    }

    /// Current 1-based line number (0 before the first `next_line` call).
    pub fn line_num(&self) -> usize {
        self.line_num
    }

    /// Returns the next line (advancing the line counter), or `None` at EOF.
    pub fn next_line(&mut self) -> Result<Option<String>, OptError> {
        self.line_num += 1;
        match self.lines.next() {
            None => Ok(None),
            Some(Ok(line)) => Ok(Some(line)),
            Some(Err(e)) => Err(self.err(format!("failed to read line: {e}"))),
        }
    }

    /// Returns the next line whose content is not blank, or `None` at EOF.
    pub fn next_data_line(&mut self) -> Result<Option<String>, OptError> {
        while let Some(line) = self.next_line()? {
            if !line.trim().is_empty() {
                return Ok(Some(line));
            }
        }
        Ok(None)
    }

    /// Parses the next whitespace token from `tokens` as `T`.
    /// `what` names the token in the error message.
    pub fn parse_next<'s, T>(
        &self,
        tokens: &mut impl Iterator<Item = &'s str>,
        what: &str,
    ) -> Result<T, OptError>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display,
    {
        let token = tokens
            .next()
            .ok_or_else(|| self.err(format!("expected {what}, but it is missing")))?;
        token
            .parse::<T>()
            .map_err(|e| self.err(format!("failed to parse {what}: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp(content: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "optopus_parse_test_{}_{}.txt",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn open_missing_file_reports_line_zero() {
        let err = InstanceLines::open("/nonexistent/optopus/foo.txt").unwrap_err();
        match err {
            OptError::FileLoad { line, .. } => assert_eq!(line, 0),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn next_data_line_skips_blank_lines_and_tracks_numbers() {
        let path = write_tmp("a\n\n  \nb\n");
        let mut lines = InstanceLines::open(&path).unwrap();
        assert_eq!(lines.next_data_line().unwrap().as_deref(), Some("a"));
        assert_eq!(lines.line_num(), 1);
        assert_eq!(lines.next_data_line().unwrap().as_deref(), Some("b"));
        assert_eq!(lines.line_num(), 4);
        assert_eq!(lines.next_data_line().unwrap(), None);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn parse_next_reports_line_number_of_bad_token() {
        let path = write_tmp("1 2\n3 oops\n");
        let mut lines = InstanceLines::open(&path).unwrap();
        lines.next_line().unwrap();
        let line = lines.next_line().unwrap().unwrap();
        let mut tokens = line.split_whitespace();
        let _: usize = lines.parse_next(&mut tokens, "first value").unwrap();
        let err = lines
            .parse_next::<usize>(&mut tokens, "second value")
            .unwrap_err();
        match err {
            OptError::FileLoad { line, detail, .. } => {
                assert_eq!(line, 2);
                assert!(detail.contains("second value"), "{detail}");
            }
            other => panic!("unexpected error: {other}"),
        }
    }
}
