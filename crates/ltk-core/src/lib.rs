//! `ltk-core` — Less Token core library.
//!
//! A cascade log-compression pipeline that pre-processes, normalizes, deduplicates,
//! and (optionally) neural-prunes log streams before they are fed into LLM / agent
//! contexts, saving 50%–95% of tokens.
//!
//! # Pipeline
//!
//! ```text
//! raw line ─▶ Normalizer ─▶ Clusterer ─▶ (optional) SemanticCompressor ─▶ Formatter ─▶ stdout
//! ```
//!
//! - **Stage 0 — `Normalizer`**: strips ANSI, masks volatile sub-tokens (IP, UUID, TS, hex,
//!   numbers, paths) into stable placeholders, and tokenizes for similarity.
//! - **Stage 1 — `Clusterer`**: fast rule-based online clustering (Jaccard similarity on token
//!   sets, move-to-front LRU with frozen exemplars, inspired by AMD's `logslop`). Emits
//!   parameterized templates with occurrence counts.
//! - **Stage 2 — `SemanticCompressor`** *(optional, feature-gated)*: LLMLingua-style neural token
//!   pruning. Either a local ONNX Runtime token-classification model (`llmlingua-onnx`) or a remote
//!   HTTP daemon (`llmlingua-rpc`).
//! - **Stage 3 — `Formatter`**: renders clusters into a token-efficient format (`compact` / `tsv` /
//!   `toon` / `json-minimal`).
//!
//! # Feature flags
//!
//! | Feature           | Default | Adds                                                   |
//! |-------------------|---------|--------------------------------------------------------|
//! | `default`         | yes     | Stage 0/1/3 — zero-model, sub-5ms rule-based mode.     |
//! | `tiktoken`        | no      | Accurate OpenAI token counting for savings stats.      |
//! | `llmlingua-onnx`  | no      | Stage 2 local neural pruning via ONNX Runtime.         |
//! | `llmlingua-rpc`   | no      | Stage 2 remote neural pruning via an HTTP daemon.      |
//!
//! Phase 1 ships the trait contracts and pipeline scaffolding; concrete
//! `Normalizer` / `Clusterer` / `Formatter` / `SemanticCompressor` implementations
//! land in later phases.

use std::{borrow::Cow, sync::Arc};

use compact_str::CompactString;

pub mod checkout;
mod clusterer;
mod formatter;
mod normalizer;

pub use clusterer::JaccardClusterer;
pub use formatter::{CompactFormatter, JsonMinimalFormatter, ToonFormatter, TsvFormatter};
pub use normalizer::RegexNormalizer;
#[cfg(feature = "llmlingua-onnx")]
pub use onnx::OnnxCompressor;
#[cfg(feature = "llmlingua-rpc")]
pub use rpc::RpcCompressor;

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

/// Errors produced by the `ltk-core` pipeline.
#[derive(Debug, thiserror::Error)]
pub enum LtkError {
    /// Stage 2 semantic compression failed.
    #[error("semantic compression failed: {0}")]
    Compression(String),
    /// A built-in regex pattern failed to compile (only possible on a bug in
    /// the constant patterns).
    #[error("regex compile failed: {0}")]
    Regex(String),
    /// The `tiktoken` BPE ranks failed to load.
    #[error("tiktoken tokenizer load failed: {0}")]
    TiktokenLoad(String),
    /// A remote LLMLingua daemon returned an error (feature `llmlingua-rpc`).
    #[cfg(feature = "llmlingua-rpc")]
    #[error("llmlingua rpc error: {0}")]
    Rpc(String),
    /// An ONNX Runtime inference error (feature `llmlingua-onnx`).
    #[cfg(feature = "llmlingua-onnx")]
    #[error("onnx inference error: {0}")]
    Onnx(String),
}

// ---------------------------------------------------------------------------------------------
// Stage 0: normalized tokens
// ---------------------------------------------------------------------------------------------

/// The set of volatile log sub-tokens that normalization collapses into stable
/// placeholders, so that near-identical lines cluster together.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, serde::Serialize)]
pub enum Placeholder {
    /// IPv4/IPv6 address, e.g. `192.168.1.10` → `<IP>`.
    Ip,
    /// TCP/UDP port, e.g. `5432` → `<PORT>`.
    Port,
    /// RFC 4122 UUID, e.g. `550e8400-e29b-41d4-a716-446655440000` → `<UUID>`.
    Uuid,
    /// Timestamp (ISO-8601 / epoch / syslog), e.g. `2026-07-31T10:00:01Z` → `<TS>`.
    Timestamp,
    /// Hex literal, e.g. `0x413d12000` / `deadbeef` → `<HEX>`.
    Hex,
    /// Decimal/float number, e.g. `42` / `3.14` → `<NUM>`.
    Number,
    /// Filesystem path, e.g. `/var/log/syslog` → `<PATH>`.
    Path,
}

impl Placeholder {
    /// The canonical rendering used in parameterized templates, e.g. `<IP>`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ip => "<IP>",
            Self::Port => "<PORT>",
            Self::Uuid => "<UUID>",
            Self::Timestamp => "<TS>",
            Self::Hex => "<HEX>",
            Self::Number => "<NUM>",
            Self::Path => "<PATH>",
        }
    }
}

/// A normalized token: either a literal substring or a normalized placeholder.
///
/// Literals use [`CompactString`] (small-string optimization), so tokens up to 24
/// bytes avoid heap allocation. Placeholders are zero-cost `Copy` tags.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Token {
    /// A literal substring of the (possibly masked) input.
    Lit(CompactString),
    /// A normalized placeholder substituting a volatile sub-token.
    Mask(Placeholder),
}

impl Token {
    /// The canonical text rendering of this token.
    #[must_use]
    pub fn render(&self) -> &str {
        match self {
            Self::Lit(s) => s.as_str(),
            Self::Mask(p) => p.as_str(),
        }
    }
}

/// The output of [`Normalizer::normalize`]: a zero-copy view over the input line
/// plus the tokenized representation used for clustering.
#[derive(Debug, Clone)]
pub struct NormalizedLine<'a> {
    /// The original raw line, borrowed unmodified from the input stream.
    pub raw: &'a str,
    /// The masked text with volatile sub-tokens replaced by placeholders.
    /// `Cow::Borrowed` when no masking was needed (true zero-copy); `Cow::Owned`
    /// otherwise.
    pub masked: Cow<'a, str>,
    /// Tokenized view of [`Self::masked`] for Jaccard similarity.
    pub tokens: Vec<Token>,
}

impl<'a> NormalizedLine<'a> {
    /// Construct a normalized line from its parts.
    #[must_use]
    pub const fn new(raw: &'a str, masked: Cow<'a, str>, tokens: Vec<Token>) -> Self {
        Self { raw, masked, tokens }
    }
}

// ---------------------------------------------------------------------------------------------
// Stage 1: cluster output
// ---------------------------------------------------------------------------------------------

/// A cluster of near-identical log lines, represented by a frozen exemplar template.
///
/// Following `logslop`, the exemplar is the *first* line that created the cluster;
/// subsequent matches never mutate it, preventing centroid drift. Unlike `logslop`,
/// `ltk-core` also records the occurrence [`count`](Self::count) and a parameterized
/// [`template`](Self::template) for compact output.
#[derive(Debug, Clone)]
pub struct LogCluster {
    /// The parameterized template (placeholders rendered) shared by all members.
    pub template: String,
    /// Number of raw lines that matched this cluster.
    pub count: u64,
    /// The exemplar's token set, precomputed for similarity matching.
    pub tokens: Box<[Token]>,
}

impl LogCluster {
    /// Construct a cluster from its parts.
    #[must_use]
    pub const fn new(template: String, count: u64, tokens: Box<[Token]>) -> Self {
        Self { template, count, tokens }
    }
}

// ---------------------------------------------------------------------------------------------
// Core traits
// ---------------------------------------------------------------------------------------------

/// Stage 0: normalizes a raw log line into a tokenized, placeholder-masked view.
///
/// Implementations must be cheap (sub-microsecond) and borrow from the input so the
/// hot path stays zero-copy until the clusterer copies what it keeps.
pub trait Normalizer: Send + Sync {
    /// Normalize `raw` into a [`NormalizedLine`] borrowing from `raw`.
    fn normalize<'a>(&self, raw: &'a str) -> NormalizedLine<'a>;
}

/// Stage 1: online clustering of normalized lines into parameterized templates.
///
/// The clusterer owns the live cluster state. [`ingest`](Self::ingest) accepts a
/// borrowed [`NormalizedLine`] of any lifetime and copies what it needs to keep;
/// [`finish`](Self::finish) consumes the clusterer and returns the final clusters.
pub trait Clusterer {
    /// Ingest one normalized line.
    fn ingest(&mut self, line: NormalizedLine<'_>);

    /// Finalize and return all clusters.
    fn finish(self) -> Vec<LogCluster>;
}

/// Stage 2 (optional): neural semantic token pruning (LLMLingua-style).
///
/// `target_rate` is the fraction of information to *keep* (0.0–1.0); e.g. `0.4`
/// keeps ~40% of the tokens by pruning low-information ones based on a small
/// language model's perplexity / token-keep probability.
pub trait SemanticCompressor: Send + Sync {
    /// Compress `text` keeping roughly `target_rate` of the information.
    ///
    /// # Errors
    /// Returns [`LtkError`] if the underlying model or daemon fails.
    fn compress(&self, text: &str, target_rate: f32) -> Result<String, LtkError>;
}

/// Stage 3: renders clusters into a token-efficient output format.
pub trait Formatter {
    /// Format all `clusters` into the final output string.
    fn format(&self, clusters: &[LogCluster]) -> String;
}

impl Formatter for Box<dyn Formatter> {
    fn format(&self, clusters: &[LogCluster]) -> String {
        (**self).format(clusters)
    }
}

// ---------------------------------------------------------------------------------------------
// Output format + token estimation
// ---------------------------------------------------------------------------------------------

/// The output format selected by `-f` / `--format`.
#[derive(Debug, Clone, Copy, Eq, PartialEq, serde::Serialize, Default)]
pub enum OutputFormat {
    /// `[xN] <template>` — one line per cluster (default).
    #[default]
    Compact,
    /// Tab-separated: `count\ttemplate`.
    Tsv,
    /// TOON (tagged object-optimized notation) for LLM-friendly structured logs.
    Toon,
    /// Minimal JSON array of `{count, template}`.
    JsonMinimal,
}

impl OutputFormat {
    /// The canonical CLI name for this format.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Compact => "compact",
            Self::Tsv => "tsv",
            Self::Toon => "toon",
            Self::JsonMinimal => "json-minimal",
        }
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Heuristic token estimate: ~4 bytes per token (matches common char→token ratios
/// for ASCII log text, same heuristic used by `rtk`).
///
/// With the `tiktoken` feature, [`tiktoken::TiktokenCounter`] provides exact OpenAI
/// counts for the `--stats` savings report.
#[must_use]
pub const fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

/// Token-savings statistics reported by `--stats` (raw vs Stage 1 vs Stage 1+2).
#[derive(Debug, Clone, serde::Serialize)]
pub struct Stats {
    /// Number of raw input lines processed.
    pub raw_lines: u64,
    /// Total bytes of raw input.
    pub raw_bytes: u64,
    /// Number of clusters emitted by Stage 1.
    pub cluster_count: usize,
    /// Estimated tokens in the raw input.
    pub raw_tokens: u64,
    /// Estimated tokens in the Stage 1 output (rule-based clustering).
    pub stage1_tokens: u64,
    /// Estimated tokens in the Stage 1+2 output (with neural pruning), if Stage 2 ran.
    pub stage2_tokens: Option<u64>,
    /// Number of lines whose cluster was evicted and could not be merged
    /// (only nonzero when `--window` is set and a cluster has no surviving neighbor).
    pub lost_lines: u64,
}

impl Stats {
    /// Fraction of raw tokens retained after Stage 1 (0.0 = all dropped, 1.0 = no
    /// savings). Returns `1.0` when there was no raw input.
    #[must_use]
    pub fn stage1_retention(&self) -> f64 {
        if self.raw_tokens == 0 {
            return 1.0;
        }
        let retained = f64::from(u32::try_from(self.stage1_tokens).unwrap_or(u32::MAX));
        let raw = f64::from(u32::try_from(self.raw_tokens).unwrap_or(u32::MAX));
        retained / raw
    }

    /// Fraction of raw tokens retained after Stage 2, if it ran.
    #[must_use]
    pub fn stage2_retention(&self) -> Option<f64> {
        self.stage2_tokens.map(|t| {
            if self.raw_tokens == 0 {
                return 1.0;
            }
            let retained = f64::from(u32::try_from(t).unwrap_or(u32::MAX));
            let raw = f64::from(u32::try_from(self.raw_tokens).unwrap_or(u32::MAX));
            retained / raw
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------------------------

/// The cascade compression pipeline:
/// `Normalizer → Clusterer → (optional) SemanticCompressor → Formatter`.
///
/// Construct with [`Pipeline::new`], then optionally
/// [`with_compressor`](Self::with_compressor) /
/// [`with_target_rate`](Self::with_target_rate), then [`run`](Self::run) (which
/// consumes the pipeline, since [`Clusterer::finish`] consumes the clusterer).
///
/// ```
/// use ltk_core::{Pipeline, OutputFormat};
/// # use ltk_core::{Normalizer, Clusterer, Formatter, NormalizedLine, LogCluster};
/// # struct N; impl Normalizer for N {
/// #   fn normalize<'a>(&self, raw: &'a str) -> NormalizedLine<'a> {
/// #     NormalizedLine::new(raw, std::borrow::Cow::Borrowed(raw), Vec::new())
/// #   }
/// # }
/// # struct C; impl Clusterer for C {
/// #   fn ingest(&mut self, _l: NormalizedLine<'_>) {}
/// #   fn finish(self) -> Vec<LogCluster> { Vec::new() }
/// # }
/// # struct F; impl Formatter for F { fn format(&self, _c: &[LogCluster]) -> String { String::new() } }
/// let (output, stats) = Pipeline::new(N, C, F)
///     .with_target_rate(0.4)
///     .run(["error at 10:00", "error at 10:01"]).unwrap();
/// # let _ = (output, stats);
/// ```
pub struct Pipeline<N, C, F> {
    normalizer: N,
    clusterer: C,
    formatter: F,
    compressor: Option<Arc<dyn SemanticCompressor>>,
    target_rate: f32,
}

impl<N, C, F> Pipeline<N, C, F> {
    /// Build a rule-based pipeline (Stage 0 + 1 + 3) from its three components.
    #[must_use]
    pub fn new(normalizer: N, clusterer: C, formatter: F) -> Self {
        Self { normalizer, clusterer, formatter, compressor: None, target_rate: 0.5 }
    }

    /// Attach a Stage 2 neural [`SemanticCompressor`] (e.g. an `OnnxCompressor` or
    /// `RpcCompressor` behind its feature flag).
    #[must_use]
    pub fn with_compressor(mut self, compressor: Arc<dyn SemanticCompressor>) -> Self {
        self.compressor = Some(compressor);
        self
    }

    /// Set the Stage 2 target keep-rate (fraction of information to retain).
    #[must_use]
    pub const fn with_target_rate(mut self, rate: f32) -> Self {
        self.target_rate = rate;
        self
    }

    /// Run the full cascade over `lines` and return `(output, stats)`.
    ///
    /// Consumes the pipeline because [`Clusterer::finish`] consumes the clusterer.
    ///
    /// # Errors
    /// Returns [`LtkError`] only if Stage 2 compression fails.
    pub fn run<I, S>(self, lines: I) -> Result<(String, Stats), LtkError>
    where
        N: Normalizer,
        C: Clusterer,
        F: Formatter,
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let Self { normalizer, mut clusterer, formatter, compressor, target_rate } = self;

        let mut raw_lines: u64 = 0;
        let mut raw_bytes: u64 = 0;
        let mut raw_tokens: u64 = 0;

        for line in lines {
            let raw = line.as_ref();
            raw_lines += 1;
            raw_bytes += u64::try_from(raw.len()).unwrap_or(u64::MAX);
            raw_tokens += u64::try_from(estimate_tokens(raw)).unwrap_or(u64::MAX);
            let normalized = normalizer.normalize(raw);
            clusterer.ingest(normalized);
        }
        // Normalizer is no longer needed; released at scope end.

        let clusters = clusterer.finish();
        let cluster_count = clusters.len();

        let stage1_output = formatter.format(&clusters);
        let stage1_tokens = u64::try_from(estimate_tokens(&stage1_output)).unwrap_or(u64::MAX);

        let (final_output, stage2_tokens) = match &compressor {
            Some(comp) => {
                tracing::debug!(
                    target: "ltk::pipeline",
                    clusters = cluster_count,
                    target_rate,
                    "running stage 2 neural compression"
                );
                let mut compressed = clusters;
                for cluster in &mut compressed {
                    cluster.template = comp.compress(&cluster.template, target_rate)?;
                }
                let out = formatter.format(&compressed);
                let tokens = u64::try_from(estimate_tokens(&out)).unwrap_or(u64::MAX);
                (out, Some(tokens))
            }
            None => (stage1_output, None),
        };

        let stats = Stats {
            raw_lines,
            raw_bytes,
            cluster_count,
            raw_tokens,
            stage1_tokens,
            stage2_tokens,
            lost_lines: 0,
        };
        tracing::trace!(target: "ltk::pipeline", ?stats, "pipeline complete");
        Ok((final_output, stats))
    }
}

impl<N, C, F> std::fmt::Debug for Pipeline<N, C, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("target_rate", &self.target_rate)
            .field("has_compressor", &self.compressor.is_some())
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------------------------
// Feature-gated Stage 2 backends
// ---------------------------------------------------------------------------------------------

/// tiktoken-backed accurate token counting (requires feature `tiktoken`).
#[cfg(feature = "tiktoken")]
pub mod tiktoken {
    use tiktoken_rs::CoreBPE;

    /// Token counter backed by tiktoken's `cl100k_base` BPE (matches LLMLingua's
    /// `gpt-3.5-turbo` encoding used for token-length weighting).
    pub struct TiktokenCounter {
        bpe: CoreBPE,
    }

    impl TiktokenCounter {
        /// Load the `cl100k_base` tokenizer.
        ///
        /// # Errors
        /// Returns [`crate::LtkError::TiktokenLoad`] if the bundled BPE ranks fail
        /// to load.
        pub fn new() -> Result<Self, crate::LtkError> {
            let bpe = tiktoken_rs::cl100k_base()
                .map_err(|e| crate::LtkError::TiktokenLoad(e.to_string()))?;
            Ok(Self { bpe })
        }

        /// Exact token count for `text` (including special tokens).
        #[must_use]
        pub fn count(&self, text: &str) -> usize {
            self.bpe.encode_with_special_tokens(text).len()
        }
    }

    impl std::fmt::Debug for TiktokenCounter {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("TiktokenCounter").finish_non_exhaustive()
        }
    }
}

/// ONNX Runtime neural compression (requires feature `llmlingua-onnx`).
///
/// LLMLingua-2 style: a small token-classification model (e.g.
/// `microsoft/llmlingua-2-xlm-roberta-large-meetingbank`) exported to ONNX, taking
/// `input_ids` + `attention_mask` and emitting per-token keep/drop logits. See
/// [`OnnxCompressor`] for the full inference and pruning pipeline.
#[cfg(feature = "llmlingua-onnx")]
pub mod onnx;

/// Remote LLMLingua daemon client over HTTP (requires feature `llmlingua-rpc`).
///
/// Delegates Stage 2 compression to a local Python LLMLingua sidecar exposing a
/// JSON endpoint `{"text": ..., "target_rate": ...}` → `{"compressed": ...}`. See
/// [`RpcCompressor`] for the full async round-trip.
#[cfg(feature = "llmlingua-rpc")]
pub mod rpc;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_renders_canonical_tags() {
        assert_eq!(Placeholder::Ip.as_str(), "<IP>");
        assert_eq!(Placeholder::Timestamp.as_str(), "<TS>");
        assert_eq!(Placeholder::Number.as_str(), "<NUM>");
    }

    #[test]
    fn token_render_round_trips() {
        assert_eq!(Token::Lit(CompactString::new("hello")).render(), "hello");
        assert_eq!(Token::Mask(Placeholder::Hex).render(), "<HEX>");
    }

    #[test]
    fn estimate_tokens_is_bytes_div_ceil_4() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("ab"), 1);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
    }

    #[test]
    fn output_format_default_is_compact() {
        assert_eq!(OutputFormat::default(), OutputFormat::Compact);
        assert_eq!(OutputFormat::Tsv.as_str(), "tsv");
    }

    #[test]
    fn stats_retention_handles_zero_raw() {
        let stats = Stats {
            raw_lines: 0,
            raw_bytes: 0,
            cluster_count: 0,
            raw_tokens: 0,
            stage1_tokens: 0,
            stage2_tokens: None,
            lost_lines: 0,
        };
        assert!((stats.stage1_retention() - 1.0).abs() < f64::EPSILON);
        assert_eq!(stats.stage2_retention(), None);
    }

    #[test]
    fn stats_retention_computes_ratio() {
        let stats = Stats {
            raw_lines: 100,
            raw_bytes: 400,
            cluster_count: 3,
            raw_tokens: 100,
            stage1_tokens: 30,
            stage2_tokens: Some(12),
            lost_lines: 0,
        };
        assert!((stats.stage1_retention() - 0.3).abs() < f64::EPSILON);
        assert!((stats.stage2_retention().unwrap_or(1.0) - 0.12).abs() < f64::EPSILON);
    }
}
