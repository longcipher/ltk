//! Smoke tests for the `ltk` binary.
//!
//! Each test writes a small log file, runs the real binary through
//! `CARGO_BIN_EXE_ltk`, and asserts on stdout / stderr / exit status. No network
//! and no model files are involved, so the suite is deterministic offline.

use std::{
    error::Error,
    path::{Path, PathBuf},
    process::{Command, Output},
};

/// A log file that removes itself when the test ends.
struct TempLog {
    path: PathBuf,
}

impl TempLog {
    fn new(name: &str, contents: &str) -> Result<Self, Box<dyn Error>> {
        let mut path = std::env::temp_dir();
        path.push(format!("ltk-cli-{}-{name}", std::process::id()));
        std::fs::write(&path, contents)?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempLog {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn ltk(args: &[&str]) -> Result<Output, Box<dyn Error>> {
    Ok(Command::new(env!("CARGO_BIN_EXE_ltk")).args(args).output()?)
}

fn stdout_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// Read the numeric value of a `label:  value` row of the `--stats` report.
fn stat_number(report: &str, label: &str) -> f64 {
    let line = report.lines().find(|l| l.starts_with(label)).unwrap_or_default();
    line[label.len()..].trim().trim_end_matches('%').parse::<f64>().unwrap_or(f64::NAN)
}

/// Three lines that differ only by IP, so all three collapse into one cluster.
const REPEATED: &str = "\
connect to 10.0.0.1:5432 refused
connect to 10.0.0.2:5432 refused
connect to 10.0.0.3:5432 refused
";

#[test]
fn help_lists_the_compression_options() -> Result<(), Box<dyn Error>> {
    let out = ltk(&["--help"])?;
    assert!(out.status.success());
    let stdout = stdout_of(&out);
    assert!(stdout.contains("--format"), "{stdout}");
    assert!(stdout.contains("--window"), "{stdout}");
    assert!(stdout.contains("--threshold"), "{stdout}");
    Ok(())
}

#[test]
fn compresses_a_file_into_one_cluster() -> Result<(), Box<dyn Error>> {
    let log = TempLog::new("repeated.log", REPEATED)?;
    let out = ltk(&[log.path().to_string_lossy().as_ref()])?;
    assert!(out.status.success(), "{}", stderr_of(&out));
    let stdout = stdout_of(&out);
    assert_eq!(stdout.trim(), "[x3] connect to <IP>:<PORT> refused");
    Ok(())
}

#[test]
fn tsv_format_renders_count_and_template() -> Result<(), Box<dyn Error>> {
    let log = TempLog::new("tsv.log", REPEATED)?;
    let out = ltk(&["--format", "tsv", log.path().to_string_lossy().as_ref()])?;
    assert!(out.status.success(), "{}", stderr_of(&out));
    assert_eq!(stdout_of(&out).trim(), "3\tconnect to <IP>:<PORT> refused");
    Ok(())
}

#[test]
fn json_minimal_format_renders_a_json_array() -> Result<(), Box<dyn Error>> {
    let log = TempLog::new("json.log", REPEATED)?;
    let out = ltk(&["--format", "json-minimal", log.path().to_string_lossy().as_ref()])?;
    assert!(out.status.success(), "{}", stderr_of(&out));
    assert_eq!(
        stdout_of(&out).trim(),
        r#"[{"count":3,"template":"connect to <IP>:<PORT> refused"}]"#
    );
    Ok(())
}

#[test]
fn stats_go_to_stderr_and_never_to_stdout() -> Result<(), Box<dyn Error>> {
    let log = TempLog::new("stats.log", REPEATED)?;
    let out = ltk(&["--stats", log.path().to_string_lossy().as_ref()])?;
    assert!(out.status.success(), "{}", stderr_of(&out));
    let stderr = stderr_of(&out);
    assert!(stderr.contains("raw lines:"), "{stderr}");
    assert!(stderr.contains("clusters:"), "{stderr}");
    // The savings percentage must follow from the reported token counts,
    // whichever counter the build uses.
    let raw = stat_number(&stderr, "raw tokens:");
    let stage1 = stat_number(&stderr, "stage1 tokens:");
    let savings = stat_number(&stderr, "stage1 savings:");
    let expected = (1.0 - stage1 / raw) * 100.0;
    assert!((savings - expected).abs() < 0.1, "raw={raw} stage1={stage1} reported={savings}");
    // Nothing was evicted, so no `lost lines` row.
    assert!(!stderr.contains("lost lines"), "{stderr}");
    assert!(!stdout_of(&out).contains("raw lines"), "stats leaked to stdout");
    Ok(())
}

#[test]
fn window_of_one_reports_the_lines_it_dropped() -> Result<(), Box<dyn Error>> {
    let log = TempLog::new("window.log", "alpha beta gamma delta\nepsilon zeta eta theta\n")?;
    let out = ltk(&["--window", "1", "--stats", log.path().to_string_lossy().as_ref()])?;
    assert!(out.status.success(), "{}", stderr_of(&out));
    let stderr = stderr_of(&out);
    assert!(stderr.contains("lost lines:"), "{stderr}");
    Ok(())
}

#[test]
fn an_out_of_range_target_rate_is_rejected() -> Result<(), Box<dyn Error>> {
    let log = TempLog::new("rate.log", REPEATED)?;
    let out = ltk(&["--target-rate", "2.0", log.path().to_string_lossy().as_ref()])?;
    assert!(!out.status.success());
    assert!(stderr_of(&out).contains("--target-rate"), "{}", stdout_of(&out));
    Ok(())
}

#[cfg(not(feature = "llmlingua-onnx"))]
#[test]
fn either_onnx_flag_alone_is_rejected_when_the_build_lacks_support() -> Result<(), Box<dyn Error>> {
    let log = TempLog::new("onnx-unsupported.log", REPEATED)?;
    let path = log.path().to_string_lossy().into_owned();
    for args in [
        vec!["--onnx-model", "model.onnx", "--onnx-tokenizer", "tokenizer.json"],
        vec!["--onnx-tokenizer", "tokenizer.json"],
        vec!["--onnx-model", "model.onnx"],
    ] {
        let mut full: Vec<&str> = args.clone();
        full.push(&path);
        let out = ltk(&full)?;
        assert!(!out.status.success(), "{args:?} should have failed");
        assert!(stderr_of(&out).contains("no ONNX support"), "{args:?}: {}", stdout_of(&out));
    }
    Ok(())
}

#[test]
fn onnx_and_rpc_backends_are_mutually_exclusive() -> Result<(), Box<dyn Error>> {
    let log = TempLog::new("both-backends.log", REPEATED)?;
    let out = ltk(&[
        "--onnx-model",
        "model.onnx",
        "--onnx-tokenizer",
        "tokenizer.json",
        "--rpc-endpoint",
        "http://127.0.0.1:8080/compress",
        log.path().to_string_lossy().as_ref(),
    ])?;
    assert!(!out.status.success());
    assert!(stderr_of(&out).contains("mutually exclusive"), "{}", stdout_of(&out));
    Ok(())
}

#[cfg(feature = "llmlingua-onnx")]
#[test]
fn a_half_specified_onnx_pair_is_rejected() -> Result<(), Box<dyn Error>> {
    let log = TempLog::new("onnx-half.log", REPEATED)?;
    let out = ltk(&["--onnx-model", "model.onnx", log.path().to_string_lossy().as_ref()])?;
    assert!(!out.status.success());
    let stderr = stderr_of(&out);
    assert!(stderr.contains("--onnx-tokenizer"), "{stderr}");
    Ok(())
}

#[cfg(not(feature = "llmlingua-rpc"))]
#[test]
fn rpc_flags_are_rejected_when_the_build_lacks_support() -> Result<(), Box<dyn Error>> {
    let log = TempLog::new("rpc-unsupported.log", REPEATED)?;
    let out = ltk(&[
        "--rpc-endpoint",
        "http://127.0.0.1:8080/compress",
        log.path().to_string_lossy().as_ref(),
    ])?;
    assert!(!out.status.success());
    assert!(stderr_of(&out).contains("no LLMLingua RPC support"), "{}", stdout_of(&out));
    Ok(())
}
