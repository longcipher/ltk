//! Stage 1: online Jaccard-similarity log-line clusterer.
//!
//! [`JaccardClusterer`] is an incremental, rule-based clusterer inspired by
//! AMD's `logslop`. It groups near-identical normalized log lines into
//! parameterized templates with occurrence counts, using:
//!
//! - **Jaccard similarity** over the per-line token *sets* (bag-of-tokens, order-insensitive).
//! - **Move-to-front LRU** with a count-bounded capacity: head = most recently touched, tail =
//!   eviction victim. No timestamps; recency is purely positional, refreshed on both match and
//!   create.
//! - **Frozen exemplars**: matching a line to a cluster *never* mutates the cluster's tokens or
//!   template — only its count and position. This prevents template drift when a slowly-mutating
//!   log line would otherwise walk a cluster across the token space.
//! - **First-match-wins, head-first scan**: biases toward the most recent pattern (temporal
//!   locality) and avoids a global argmax pass.
//!
//! `jaccard([], []) == 1.0` (two empty-token lines are treated as identical),
//! matching the reference implementation.

use ahash::AHashSet;

use crate::{Clusterer, LogCluster, NormalizedLine, Token};

/// Default Jaccard similarity threshold for matching (per the `ltk-core` spec;
/// `logslop` uses 0.6).
const DEFAULT_THRESHOLD: f64 = 0.7;

/// Default maximum number of live clusters (`0` = unbounded).
const DEFAULT_WINDOW: usize = 0;

/// Online Jaccard-similarity clusterer with move-to-front LRU eviction.
#[derive(Debug)]
pub struct JaccardClusterer {
    /// Jaccard score required for a line to match an existing cluster.
    threshold: f64,
    /// Maximum live clusters; `0` = unbounded.
    max_clusters: usize,
    /// Head (index 0) = most recently touched; tail = LRU victim.
    clusters: Vec<ClusterEntry>,
}

#[derive(Debug)]
struct ClusterEntry {
    /// Parameterized template (placeholders rendered) shared by all members.
    template: String,
    /// Number of raw lines that matched this cluster.
    count: u64,
    /// Frozen exemplar token sequence (never mutated after creation).
    tokens: Box<[Token]>,
    /// Precomputed token set for O(1) membership during Jaccard.
    token_set: Box<AHashSet<Token>>,
}

impl JaccardClusterer {
    /// Build a clusterer with default threshold (0.7) and window (1000).
    #[must_use]
    pub const fn new() -> Self {
        Self::with_params(DEFAULT_THRESHOLD, DEFAULT_WINDOW)
    }

    /// Build a clusterer with a custom `threshold` (clamped to `[0.0, 1.0]`)
    /// and `max_clusters` window (`0` = unbounded).
    #[must_use]
    pub const fn with_params(threshold: f64, max_clusters: usize) -> Self {
        Self { threshold: threshold.clamp(0.0, 1.0), max_clusters, clusters: Vec::new() }
    }

    /// Current number of live clusters.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.clusters.len()
    }

    /// Whether the clusterer currently holds no clusters.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.clusters.is_empty()
    }
}

impl Default for JaccardClusterer {
    fn default() -> Self {
        Self::new()
    }
}

impl Clusterer for JaccardClusterer {
    fn ingest(&mut self, line: NormalizedLine<'_>) {
        // Borrowed token set for the incoming line (zero-clone). The borrow on
        // `line.tokens` ends after the scan, so `line.tokens` can still be
        // moved when creating a new cluster below.
        let new_set: AHashSet<&Token> = line.tokens.iter().collect();

        // Head-first scan; first cluster clearing the threshold wins.
        let matched: Option<usize> = self
            .clusters
            .iter()
            .enumerate()
            .find(|(_, e)| jaccard(&new_set, &e.token_set) >= self.threshold)
            .map(|(i, _)| i);

        if let Some(i) = matched {
            // Move-to-front (preserve order via `remove`), keep the frozen
            // exemplar, bump the count.
            let entry = self.clusters.remove(i);
            self.clusters.insert(0, entry);
            self.clusters[0].count += 1;
        } else {
            // New cluster at the head; evict the LRU tail first if at capacity.
            if self.max_clusters > 0 &&
                self.clusters.len() >= self.max_clusters &&
                let Some(victim) = self.clusters.pop()
            {
                // Merge evicted count into the nearest surviving cluster
                // (best Jaccard match) so that total line count is preserved.
                let victim_set: AHashSet<&Token> = victim.token_set.iter().collect();
                if let Some(best_idx) = self
                    .clusters
                    .iter()
                    .enumerate()
                    .map(|(i, e)| (i, jaccard(&victim_set, &e.token_set)))
                    .max_by(|(_, sa), (_, sb)| {
                        sa.partial_cmp(sb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                {
                    self.clusters[best_idx].count += victim.count;
                }
            }
            let template = line.masked.to_string();
            let tokens: Box<[Token]> = line.tokens.into_boxed_slice();
            let token_set = Box::new(tokens.iter().cloned().collect::<AHashSet<Token>>());
            self.clusters.insert(0, ClusterEntry { template, count: 1, tokens, token_set });
        }
    }

    fn finish(self) -> Vec<LogCluster> {
        self.clusters.into_iter().map(|e| LogCluster::new(e.template, e.count, e.tokens)).collect()
    }
}

/// Jaccard similarity over two token sets: `|A ∩ B| / |A ∪ B|`.
///
/// Both-empty returns `1.0` (two blank lines are treated as identical).
fn jaccard(new_set: &AHashSet<&Token>, cluster_set: &AHashSet<Token>) -> f64 {
    let inter = new_set.iter().filter(|t| cluster_set.contains(*t)).count();
    let union = new_set.len() + cluster_set.len() - inter;
    if union == 0 {
        1.0
    } else {
        f64::from(u32::try_from(inter).unwrap_or(u32::MAX)) /
            f64::from(u32::try_from(union).unwrap_or(u32::MAX))
    }
}

#[cfg(test)]
mod tests {
    use std::{borrow::Cow, error::Error};

    use super::*;
    use crate::{Placeholder, Token};

    /// Build a [`NormalizedLine`] borrowing `raw` as both `raw` and `masked`,
    /// with the given tokens.
    fn nl<'a>(raw: &'a str, tokens: Vec<Token>) -> NormalizedLine<'a> {
        NormalizedLine::new(raw, Cow::Borrowed(raw), tokens)
    }

    fn lit(s: &str) -> Token {
        Token::Lit(compact_str::CompactString::new(s))
    }

    #[test]
    fn identical_lines_collapse_into_one_cluster() {
        let mut c = JaccardClusterer::new();
        c.ingest(nl(
            "error at <TS>",
            vec![lit("error"), lit("at"), Token::Mask(Placeholder::Timestamp)],
        ));
        c.ingest(nl(
            "error at <TS>",
            vec![lit("error"), lit("at"), Token::Mask(Placeholder::Timestamp)],
        ));
        let clusters = c.finish();
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].count, 2);
    }

    #[test]
    fn lines_differing_only_in_ip_cluster() {
        let mut c = JaccardClusterer::new();
        c.ingest(nl(
            "connect <IP> refused",
            vec![lit("connect"), Token::Mask(Placeholder::Ip), lit("refused")],
        ));
        c.ingest(nl(
            "connect <IP> refused",
            vec![lit("connect"), Token::Mask(Placeholder::Ip), lit("refused")],
        ));
        let clusters = c.finish();
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].count, 2);
    }

    #[test]
    fn dissimilar_lines_form_separate_clusters() {
        let mut c = JaccardClusterer::new();
        c.ingest(nl("alpha beta", vec![lit("alpha"), lit("beta")]));
        c.ingest(nl("gamma delta epsilon", vec![lit("gamma"), lit("delta"), lit("epsilon")]));
        let clusters = c.finish();
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].count, 1);
        assert_eq!(clusters[1].count, 1);
    }

    #[test]
    fn threshold_one_requires_exact_token_set() {
        let mut c = JaccardClusterer::with_params(1.0, 100);
        c.ingest(nl("a b c", vec![lit("a"), lit("b"), lit("c")]));
        // Extra token → Jaccard 3/4 = 0.75 < 1.0 → new cluster.
        c.ingest(nl("a b c d", vec![lit("a"), lit("b"), lit("c"), lit("d")]));
        let clusters = c.finish();
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn lru_eviction_merges_count_to_nearest() {
        // Window of 2. Ingest A, B (both at capacity). Touch A → A moves to
        // head, B is LRU. Ingest C → B evicted, its count merged into A
        // (nearest surviving cluster), C at head.
        let mut c = JaccardClusterer::with_params(0.7, 2);
        c.ingest(nl("a line", vec![lit("a"), lit("line")]));
        c.ingest(nl("b line", vec![lit("b"), lit("line")]));
        // Touch A: identical to first → matches, moves to head.
        c.ingest(nl("a line", vec![lit("a"), lit("line")]));
        // New distinct cluster C: evicts LRU tail (B), B's count (1) merged
        // into nearest surviving cluster (A shares "line" token).
        c.ingest(nl("c line", vec![lit("c"), lit("line")]));
        let clusters = c.finish();
        assert_eq!(clusters.len(), 2);
        let templates: Vec<&str> = clusters.iter().map(|x| x.template.as_str()).collect();
        assert!(templates.contains(&"a line"));
        assert!(templates.contains(&"c line"));
        assert!(!templates.contains(&"b line"));
        // A was touched twice (create + match), plus B's count (1) merged.
        let a = clusters.iter().find(|x| x.template == "a line");
        assert_eq!(a.map(|x| x.count), Some(3));
        // C is new.
        let c_cluster = clusters.iter().find(|x| x.template == "c line");
        assert_eq!(c_cluster.map(|x| x.count), Some(1));
    }

    #[test]
    fn move_to_front_promotes_matched_cluster() {
        let mut c = JaccardClusterer::with_params(0.7, 100);
        c.ingest(nl("a line", vec![lit("a"), lit("line")]));
        c.ingest(nl("b line", vec![lit("b"), lit("line")]));
        // Head is now B. Touch A → A should move to head.
        c.ingest(nl("a line", vec![lit("a"), lit("line")]));
        let clusters = c.finish();
        assert_eq!(clusters[0].template, "a line");
        assert_eq!(clusters[0].count, 2);
        assert_eq!(clusters[1].template, "b line");
    }

    #[test]
    fn empty_token_lines_cluster_together() {
        let mut c = JaccardClusterer::new();
        c.ingest(nl("", vec![]));
        c.ingest(nl("", vec![]));
        let clusters = c.finish();
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].count, 2);
    }

    #[test]
    fn finish_preserves_template_and_tokens() -> Result<(), Box<dyn Error>> {
        let mut c = JaccardClusterer::new();
        let toks = vec![lit("error"), Token::Mask(Placeholder::Number)];
        c.ingest(nl("error <NUM>", toks.clone()));
        let clusters = c.finish();
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].template, "error <NUM>");
        assert_eq!(clusters[0].tokens.len(), 2);
        Ok(())
    }
}
