//! Binary entry point: reads a TOML benchmark config file and runs the specified
//! heuristics on the specified instances, writing results to a timestamped TOML file.
//!
//! Usage: `optopus <config_file>`

use std::process::ExitCode;

use optopus::benchmark::{Benchmark, BenchmarkConfig};
use optopus::error::OptError;

const USAGE: &str = "Usage: optopus <config_file>";

enum CliAction {
    Run(String),
    PrintHelp,
}

fn parse_args() -> Result<CliAction, OptError> {
    match std::env::args().nth(1).as_deref() {
        Some("-h") | Some("--help") => Ok(CliAction::PrintHelp),
        Some(path) => Ok(CliAction::Run(path.to_string())),
        None => Err(OptError::Config(USAGE.to_string())),
    }
}

fn run() -> Result<(), OptError> {
    tracing_subscriber::fmt::init();

    let config_file = match parse_args()? {
        CliAction::Run(path) => path,
        CliAction::PrintHelp => {
            println!("{USAGE}");
            return Ok(());
        }
    };

    let config_str = std::fs::read_to_string(&config_file)
        .map_err(|e| OptError::Config(format!("failed to read config file '{}': {}", config_file, e)))?;

    let config: BenchmarkConfig = toml::from_str(&config_str)?;

    let report = Benchmark::run_from_config(config, &config_file)?;

    let toml_str = toml::to_string(&report)?;

    let output_dir = std::path::Path::new("result");
    std::fs::create_dir_all(output_dir).map_err(|e| OptError::FileLoad {
        path: output_dir.display().to_string(),
        line: 0,
        detail: format!("failed to create output directory: {e}"),
    })?;

    let output_file = output_dir.join(
        chrono::Local::now()
            .format("%Y%m%d_%H%M%S.toml")
            .to_string(),
    );

    std::fs::write(&output_file, toml_str).map_err(|e| OptError::FileLoad {
        path: output_file.display().to_string(),
        line: 0,
        detail: format!("failed to write result file: {e}"),
    })?;

    tracing::info!("Results written to {}", output_file.display());
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}
