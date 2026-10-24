//! End-to-end tests for the full `ltk-core` cascade:
//! `Normalizer -> Clusterer -> (optional) SemanticCompressor -> Formatter`.

use std::{error::Error, sync::Arc};

use ltk_core::{
    CompactFormatter, Formatter, JaccardClusterer, JsonMinimalFormatter, LtkError, Pipeline,
    RegexNormalizer, SemanticCompressor, Stats, TokenCounter, ToonFormatter, TsvFormatter,
};

/// Three near-identical connection failures plus one unrelated line. The first
/// three differ only in timestamp and IP, so Stage 0 masks them identically and
/// Stage 1 collapses them into a single cluster.
const LINES: [&str; 4] = [
    "2026-07-31T10:00:01.001Z ERROR connect to 10.0.0.1:5432 refused",
    "2026-07-31T10:00:02.002Z ERROR connect to 10.0.0.2:5432 refused",
    "2026-07-31T10:00:03.003Z ERROR connect to 10.0.0.3:5432 refused",
    "worker started",
];

/// Sum the `[xN]` occurrence prefixes of a `compact`-formatted stream; lines
/// without a prefix count once.
fn sum_counts(compact: &str) -> u64 {
    compact.lines().fold(0, |acc, line| {
        let n = line
            .strip_prefix("[x")
            .and_then(|rest| rest.split(']').next())
            .and_then(|digits| digits.parse::<u64>().ok())
            .unwrap_or(1);
        acc + n
    })
}

fn run<F: Formatter>(formatter: F) -> Result<(String, Stats), LtkError> {
    Pipeline::new(RegexNormalizer::new()?, JaccardClusterer::new(), formatter).run(LINES)
}

/// Stage 2 test double: keeps the first two whitespace-separated words.
struct TruncatingCompressor;

impl SemanticCompressor for TruncatingCompressor {
    fn compress(&self, text: &str, _target_rate: f32) -> Result<String, LtkError> {
        Ok(text.split_whitespace().take(2).collect::<Vec<_>>().join(" "))
    }
}

/// Token-counting test double: every span costs the same fixed number of tokens.
struct FixedCounter(u64);

impl TokenCounter for FixedCounter {
    fn count(&self, _text: &str) -> u64 {
        self.0
    }
}

#[test]
fn pipeline_collapses_masked_duplicates() -> Result<(), Box<dyn Error>> {
    let (out, stats) = run(CompactFormatter::new())?;
    assert!(out.contains("[x3]"), "expected a count-3 cluster in:\n{out}");
    assert!(out.contains("<IP>:<PORT>"), "expected masked template in:\n{out}");
    assert!(out.contains("worker started"), "expected the unrelated line in:\n{out}");
    assert_eq!(stats.cluster_count, 2);
    Ok(())
}

#[test]
fn occurrence_counts_sum_to_the_input_line_count() -> Result<(), Box<dyn Error>> {
    let (out, stats) = run(CompactFormatter::new())?;
    assert_eq!(sum_counts(&out), u64::try_from(LINES.len())?);
    assert_eq!(stats.raw_lines, u64::try_from(LINES.len())?);
    assert_eq!(stats.lost_lines, 0);
    Ok(())
}

#[test]
fn stats_reports_raw_bytes() -> Result<(), Box<dyn Error>> {
    let (_, stats) = run(CompactFormatter::new())?;
    let expected: u64 = LINES.iter().map(|l| u64::try_from(l.len()).unwrap_or(u64::MAX)).sum();
    assert_eq!(stats.raw_bytes, expected);
    Ok(())
}

#[test]
fn every_formatter_renders_the_counts() -> Result<(), Box<dyn Error>> {
    let (tsv, _) = run(TsvFormatter::new())?;
    assert!(tsv.contains("3\t"), "expected a count-3 row in:\n{tsv}");
    assert!(tsv.contains("1\t"), "expected a count-1 row in:\n{tsv}");

    let (toon, _) = run(ToonFormatter::new())?;
    assert!(toon.contains("3 | "), "expected a count-3 row in:\n{toon}");

    let (json, _) = run(JsonMinimalFormatter::new())?;
    assert!(json.starts_with('[') && json.ends_with(']'), "expected a JSON array:\n{json}");
    assert!(json.contains(r#""count":3"#), "expected a count-3 object in:\n{json}");
    Ok(())
}

#[test]
fn stage_two_rewrites_templates_and_reports_tokens() -> Result<(), Box<dyn Error>> {
    let (out, stats) =
        Pipeline::new(RegexNormalizer::new()?, JaccardClusterer::new(), CompactFormatter::new())
            .with_compressor(Arc::new(TruncatingCompressor))
            .with_target_rate(0.5)
            .run(LINES)?;

    let stage2 = stats.stage2_tokens.unwrap_or(stats.stage1_tokens);
    assert!(stage2 < stats.stage1_tokens, "stage 2 must shrink the output: {stats:?}");
    // The exemplar is truncated to its first two words.
    assert!(out.lines().all(|l| l.split_whitespace().count() <= 4), "unexpected output:\n{out}");
    Ok(())
}

#[test]
fn stage_two_failure_propagates_as_an_error() -> Result<(), Box<dyn Error>> {
    struct FailingCompressor;
    impl SemanticCompressor for FailingCompressor {
        fn compress(&self, _text: &str, _target_rate: f32) -> Result<String, LtkError> {
            Err(LtkError::Compression("boom".into()))
        }
    }

    let res =
        Pipeline::new(RegexNormalizer::new()?, JaccardClusterer::new(), CompactFormatter::new())
            .with_compressor(Arc::new(FailingCompressor))
            .run(LINES);
    assert!(matches!(res, Err(LtkError::Compression(_))));
    Ok(())
}

#[test]
fn attached_token_counter_overrides_the_heuristic() -> Result<(), Box<dyn Error>> {
    let (_, stats) =
        Pipeline::new(RegexNormalizer::new()?, JaccardClusterer::new(), CompactFormatter::new())
            .with_token_counter(Arc::new(FixedCounter(7)))
            .run(LINES)?;
    // One count per input line plus one for the formatted output.
    assert_eq!(stats.raw_tokens, 7 * u64::try_from(LINES.len())?);
    assert_eq!(stats.stage1_tokens, 7);
    Ok(())
}

#[test]
fn window_of_one_reports_lines_it_could_not_keep() -> Result<(), Box<dyn Error>> {
    let lines = ["alpha beta gamma delta", "epsilon zeta eta theta", "iota kappa lambda mu"];
    let (out, stats) = Pipeline::new(
        RegexNormalizer::new()?,
        JaccardClusterer::with_params(0.7, 1),
        CompactFormatter::new(),
    )
    .run(lines)?;

    assert_eq!(stats.raw_lines, 3);
    assert_eq!(stats.lost_lines, 2, "two evicted clusters had no survivor");
    assert_eq!(sum_counts(&out) + stats.lost_lines, 3);
    Ok(())
}

#[test]
fn empty_input_produces_empty_output() -> Result<(), Box<dyn Error>> {
    let (out, stats) =
        Pipeline::new(RegexNormalizer::new()?, JaccardClusterer::new(), CompactFormatter::new())
            .run(std::iter::empty::<&str>())?;
    assert!(out.is_empty());
    assert_eq!(stats.raw_lines, 0);
    assert_eq!(stats.cluster_count, 0);
    Ok(())
}
