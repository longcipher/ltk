//! Stage 3: concrete [`Formatter`] implementations.
//!
//! Each formatter renders a slice of [`LogCluster`]s into a token-efficient
//! output string. All formatters are zero-dependency (no additional crates
//! required beyond what the core `ltk-core` crate already uses).

use crate::{Formatter, LogCluster};

/// Compact format: `[xN] template` — one line per cluster.
///
/// When `count == 1`, the prefix is omitted and only the template is printed.
/// Each cluster is terminated by a newline.
///
/// # Examples
///
/// ```text
/// [x3] connect to <IP>:<PORT> refused
/// unexpected error in worker
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct CompactFormatter;

impl CompactFormatter {
    /// Create a new compact formatter.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Formatter for CompactFormatter {
    fn format(&self, clusters: &[LogCluster]) -> String {
        let mut out = String::new();
        for c in clusters {
            if c.count > 1 {
                out.push_str("[x");
                push_u64(&mut out, c.count);
                out.push_str("] ");
            }
            out.push_str(&c.template);
            out.push('\n');
        }
        out
    }
}

/// TSV format: `count\ttemplate` — tab-separated values.
///
/// Each cluster is rendered on its own line, terminated by a newline.
///
/// # Examples
///
/// ```text
/// 3\tconnect to <IP>:<PORT> refused
/// 1\tunexpected error in worker
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct TsvFormatter;

impl TsvFormatter {
    /// Create a new TSV formatter.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Formatter for TsvFormatter {
    fn format(&self, clusters: &[LogCluster]) -> String {
        let mut out = String::new();
        for c in clusters {
            push_u64(&mut out, c.count);
            out.push('\t');
            out.push_str(&c.template);
            out.push('\n');
        }
        out
    }
}

/// TOON (tagged object-optimized notation) format.
///
/// Each cluster is rendered as a tagged line: `count | template`,
/// terminated by a newline.
#[derive(Debug, Clone, Copy, Default)]
pub struct ToonFormatter;

impl ToonFormatter {
    /// Create a new TOON formatter.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Formatter for ToonFormatter {
    fn format(&self, clusters: &[LogCluster]) -> String {
        let mut out = String::new();
        for c in clusters {
            push_u64(&mut out, c.count);
            out.push_str(" | ");
            out.push_str(&c.template);
            out.push('\n');
        }
        out
    }
}

/// Minimal JSON format: a JSON array of `{"count": N, "template": "..."}`.
///
/// Uses manual string construction (no `serde_json` dependency) with minimal
/// JSON escaping for double-quote and backslash characters.
///
/// # Examples
///
/// ```json
/// [{"count":3,"template":"connect to <IP>:<PORT> refused"}]
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct JsonMinimalFormatter;

impl JsonMinimalFormatter {
    /// Create a new JSON-minimal formatter.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Formatter for JsonMinimalFormatter {
    fn format(&self, clusters: &[LogCluster]) -> String {
        let mut out = String::from('[');
        for (i, c) in clusters.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(r#"{"count":"#);
            push_u64(&mut out, c.count);
            out.push_str(r#","template":""#);
            // Minimal JSON escaping: only double-quote and backslash.
            for ch in c.template.chars() {
                match ch {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    _ => out.push(ch),
                }
            }
            out.push_str("\"}");
        }
        out.push(']');
        out
    }
}

/// Push a `u64` decimal representation into `buf` without heap allocation.
fn push_u64(buf: &mut String, mut n: u64) {
    if n == 0 {
        buf.push('0');
        return;
    }
    // u64::MAX is 20 digits.
    let mut tmp = [0u8; 20];
    let mut i = tmp.len();
    while n > 0 {
        i -= 1;
        tmp[i] = b'0' + u8::try_from(n % 10).unwrap_or(b'0');
        n /= 10;
    }
    // SAFETY: `tmp[i..]` was written entirely from ASCII digit bytes (`b'0'..=b'9'`).
    buf.push_str(unsafe { std::str::from_utf8_unchecked(&tmp[i..]) });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cluster(template: &str, count: u64) -> LogCluster {
        LogCluster::new(template.into(), count, Box::new([]))
    }

    #[test]
    fn compact_omits_prefix_for_count_one() {
        let f = CompactFormatter::new();
        let out = f.format(&[cluster("error in worker", 1)]);
        assert_eq!(out, "error in worker\n");
    }

    #[test]
    fn compact_uses_prefix_for_count_gt_one() {
        let f = CompactFormatter::new();
        let out = f.format(&[cluster("error in worker", 3)]);
        assert_eq!(out, "[x3] error in worker\n");
    }

    #[test]
    fn tsv_separates_count_and_template() {
        let f = TsvFormatter::new();
        let out = f.format(&[cluster("timeout", 2)]);
        assert_eq!(out, "2\ttimeout\n");
    }

    #[test]
    fn toon_uses_pipe_separator() {
        let f = ToonFormatter::new();
        let out = f.format(&[cluster("timeout", 5)]);
        assert_eq!(out, "5 | timeout\n");
    }

    #[test]
    fn json_minimal_produces_array() {
        let f = JsonMinimalFormatter::new();
        let out = f.format(&[cluster("timeout", 2)]);
        assert_eq!(out, r#"[{"count":2,"template":"timeout"}]"#);
    }

    #[test]
    fn compact_multiple_clusters() {
        let f = CompactFormatter::new();
        let out = f.format(&[cluster("alpha", 3), cluster("beta", 1)]);
        assert_eq!(out, "[x3] alpha\nbeta\n");
    }

    #[test]
    fn empty_clusters_produce_empty_output() {
        let f = CompactFormatter::new();
        let out = f.format(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn push_u64_various_values() {
        let mut buf = String::new();
        push_u64(&mut buf, 0);
        assert_eq!(buf, "0");
        buf.clear();
        push_u64(&mut buf, 42);
        assert_eq!(buf, "42");
        buf.clear();
        push_u64(&mut buf, u64::MAX);
        assert_eq!(buf, "18446744073709551615");
    }
}
