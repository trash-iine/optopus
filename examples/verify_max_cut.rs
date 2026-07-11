//! Independently verifies MaxCut solutions recorded in a benchmark result TOML.
//!
//! Parses the instance file and recomputes every cut in integer arithmetic
//! (`i64`), fully independent of the library's `f32` incremental pipeline.
//! Run this before claiming any best-known update.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example verify_max_cut -- \
//!     data/instances/max_cut/G55 result/record_g55_YYYYMMDD_HHMMSS.toml
//! ```

use serde::Deserialize;

#[derive(Deserialize)]
struct Report {
    results: Vec<ResultEntry>,
}

#[derive(Deserialize)]
struct ResultEntry {
    instance_path: String,
    runs: Vec<Run>,
}

#[derive(Deserialize)]
struct Run {
    run_index: usize,
    status: String,
    best_objective: f64,
    #[serde(default)]
    seed: Option<u64>,
    /// 0-indexed vertices with `x[i] == true` (`encode_as_indices` output).
    solution: Vec<usize>,
}

/// A weighted edge `(u, v, weight)` with 0-indexed endpoints.
type Edge = (usize, usize, i64);

/// Parses a Gset-format instance (`N M` header, then `i j w` 1-indexed edges)
/// with integer weights. Returns `(num_vertices, edges)`.
fn load_edges(path: &str) -> Result<(usize, Vec<Edge>), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines.next().ok_or("empty instance file")?;
    let mut it = header.split_whitespace();
    let n: usize = it
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or("bad header: missing N")?;
    let m: usize = it
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or("bad header: missing M")?;
    let mut edges = Vec::with_capacity(m);
    for line in lines {
        let mut it = line.split_whitespace();
        let i: usize = it
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| format!("bad edge line: {line}"))?;
        let j: usize = it
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| format!("bad edge line: {line}"))?;
        let w: i64 = it
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| format!("non-integer weight in line: {line}"))?;
        if i == 0 || j == 0 || i > n || j > n {
            return Err(format!("edge ({i}, {j}) out of 1..={n}"));
        }
        edges.push((i - 1, j - 1, w));
    }
    if edges.len() != m {
        eprintln!("warning: header says {m} edges, file has {}", edges.len());
    }
    Ok((n, edges))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (instance_path, result_path) = match &args[..] {
        [_, i, r] => (i.as_str(), r.as_str()),
        _ => {
            eprintln!("usage: verify_max_cut <instance_path> <result_toml>");
            std::process::exit(2);
        }
    };

    let (n, edges) = match load_edges(instance_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    let report: Report = match std::fs::read_to_string(result_path)
        .map_err(|e| e.to_string())
        .and_then(|t| toml::from_str(&t).map_err(|e| e.to_string()))
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: parse {result_path}: {e}");
            std::process::exit(1);
        }
    };

    // Match result entries by file name so relative-path differences don't matter.
    let instance_name = std::path::Path::new(instance_path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();

    let mut checked = 0usize;
    let mut mismatches = 0usize;
    for (entry_idx, entry) in report.results.iter().enumerate() {
        let entry_name = std::path::Path::new(&entry.instance_path)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        if entry_name != instance_name {
            continue;
        }
        for run in &entry.runs {
            if run.status != "success" {
                continue;
            }
            let mut x = vec![false; n];
            let mut bad_index = false;
            for &v in &run.solution {
                if v >= n {
                    eprintln!(
                        "entry {entry_idx} run {}: solution vertex {v} out of range 0..{n}",
                        run.run_index
                    );
                    bad_index = true;
                    break;
                }
                x[v] = true;
            }
            if bad_index {
                mismatches += 1;
                continue;
            }
            let cut: i64 = edges
                .iter()
                .map(|&(i, j, w)| if x[i] != x[j] { w } else { 0 })
                .sum();
            checked += 1;
            let ok = run.best_objective == cut as f64;
            if !ok {
                mismatches += 1;
            }
            println!(
                "entry {entry_idx} run {:2} seed {:>20}: recomputed {cut:>8}  reported {:>10}  {}",
                run.run_index,
                run.seed.map_or_else(|| "-".into(), |s| s.to_string()),
                run.best_objective,
                if ok { "OK" } else { "MISMATCH" },
            );
        }
    }

    println!("---");
    println!("{checked} runs checked for {instance_name}, {mismatches} mismatch(es)");
    if checked == 0 || mismatches > 0 {
        std::process::exit(1);
    }
}
