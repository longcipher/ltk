//! Property-based tests for ltk-core normalizer and clusterer invariants.

use ltk_core::{Clusterer, JaccardClusterer, Normalizer, RegexNormalizer};
use proptest::{prelude::*, test_runner::TestCaseError};

fn norm() -> Result<RegexNormalizer, TestCaseError> {
    RegexNormalizer::new().map_err(|e| TestCaseError::fail(e.to_string()))
}

proptest! {
    #[test]
    fn normalizer_never_panics_on_ascii_input(s in "[ -~]{0,500}") {
        let n = norm()?;
        let _ = n.normalize(&s);
    }

    #[test]
    fn normalizer_preserves_raw_ref(s in "[ -~]{1,200}") {
        let n = norm()?;
        let line = n.normalize(&s);
        prop_assert_eq!(line.raw, s.as_str());
    }

    #[test]
    fn normalizer_masked_is_nonempty_for_nonempty_input(s in "[ -~]{1,200}") {
        let n = norm()?;
        let line = n.normalize(&s);
        prop_assert!(
            !line.masked.is_empty(),
            "masked output is empty for non-empty input: {s:?}"
        );
    }

    #[test]
    fn clusterer_never_loses_lines(
        lines in prop::collection::vec("[a-z ]{1,50}", 1..50)
    ) {
        let n = norm()?;
        let mut clusterer = JaccardClusterer::new();
        for line in &lines {
            clusterer.ingest(n.normalize(line));
        }
        let clusters = clusterer.finish();
        let total: u64 = clusters.iter().map(|c| c.count).sum();
        prop_assert_eq!(total, lines.len() as u64);
    }

    #[test]
    fn clusterer_never_exceeds_window(
        lines in prop::collection::vec("[a-z]{4,20}", 0..100)
    ) {
        let n = norm()?;
        let mut clusterer = JaccardClusterer::with_params(1.0, 10);
        for line in &lines {
            clusterer.ingest(n.normalize(line));
        }
        let clusters = clusterer.finish();
        prop_assert!(clusters.len() <= 10);
    }

    #[test]
    fn clusterer_preserves_total_line_count(
        lines in prop::collection::vec("[a-z ]{1,50}", 1..30)
    ) {
        let n = norm()?;
        let mut clusterer = JaccardClusterer::new();
        for line in &lines {
            clusterer.ingest(n.normalize(line));
        }
        let clusters = clusterer.finish();
        let total: u64 = clusters.iter().map(|c| c.count).sum();
        prop_assert_eq!(total, lines.len() as u64);
    }

    #[test]
    fn normalizer_masks_all_ipv4(
        ip in "[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}"
    ) {
        let n = norm()?;
        let input = format!("host {ip} up");
        let line = n.normalize(&input);
        let masked = line.masked.as_ref();
        prop_assert!(
            !masked.contains(ip.as_str()),
            "IP {ip} not masked in: {masked}"
        );
    }

    #[test]
    fn estimate_tokens_is_nonzero_for_nonempty(s in "[ -~]{1,200}") {
        let count = ltk_core::estimate_tokens(&s);
        prop_assert!(count > 0);
    }

    #[test]
    fn normalizer_strips_ansi_escapes(
        text in "[a-z]{1,20}"
    ) {
        let n = norm()?;
        let with_ansi = format!("\x1b[31m{text}\x1b[0m");
        let line = n.normalize(&with_ansi);
        let masked = line.masked.as_ref();
        prop_assert!(
            !masked.contains('\x1b'),
            "ANSI escape byte found in masked output: {masked:?}"
        );
        prop_assert!(
            masked.contains(&text),
            "original text {text:?} not found in masked output: {masked:?}"
        );
    }
}
