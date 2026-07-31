//! Stage 2 local neural compression via ONNX Runtime (LLMLingua-2 style).
//!
//! [`OnnxCompressor`] runs a small token-classification model (e.g.
//! `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` exported to ONNX) over
//! a HuggingFace [`Tokenizer`], producing per-token keep/drop logits. It then
//! softmaxes the logits to a drop probability and keeps the `target_rate`
//! fraction of *non-special* tokens with the lowest drop probability (most
//! informative), always preserving special tokens (`<s>`, `</s>`, …) so the
//! decoded output stays well-formed.
//!
//! ## Lazy init
//!
//! The ONNX session and tokenizer are heavy; they load lazily on the first
//! [`SemanticCompressor::compress`] call and are reused for every subsequent
//! call. The session lives behind a [`Mutex`](std::sync::Mutex) because
//! [`Session::run`] requires `&mut self`; the tokenizer is immutable after
//! load and sits in a [`OnceLock`].

use std::{
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

use ort::{session::Session, value::Tensor};
use tokenizers::Tokenizer;

use crate::{LtkError, SemanticCompressor};

/// Default max sequence length (xlm-roberta context window used by LLMLingua-2).
///
/// Inputs longer than this are truncated to the first `MAX_SEQ_LEN` tokens
/// before inference; longer log lines are rare after Stage 0/1 normalization.
const MAX_SEQ_LEN: usize = 512;

/// Index of the "drop" class in the model's per-token 2-class logits. LLMLingua-2
/// labels `1 = drop`, `0 = keep`.
const DROP_CLASS: usize = 1;

/// Neural semantic compressor backed by an ONNX Runtime session + HF tokenizer.
///
/// Construct with [`OnnxCompressor::new`], then pass an `Arc<dyn
/// SemanticCompressor>` to [`Pipeline::with_compressor`](crate::Pipeline::with_compressor).
///
/// ```no_run
/// use std::sync::Arc;
/// use ltk_core::{Pipeline, OnnxCompressor, RegexNormalizer, JaccardClusterer};
/// # use ltk_core::Formatter;
/// # struct F; impl Formatter for F { fn format(&self, _: &[ltk_core::LogCluster]) -> String { String::new() } }
///
/// let comp = Arc::new(OnnxCompressor::new("model.onnx", "tokenizer.json"));
/// let (out, stats) = Pipeline::new(RegexNormalizer::new()?, JaccardClusterer::new(), F)
///     .with_compressor(comp)
///     .with_target_rate(0.4)
///     .run(["2026-07-31 ERROR connect 10.0.0.1 refused"])?;
/// # let _ = (out, stats);
/// # Ok::<(), ltk_core::LtkError>(())
/// ```
pub struct OnnxCompressor {
    model_path: PathBuf,
    tokenizer_path: PathBuf,
    session: Mutex<Option<Session>>,
    tokenizer: OnceLock<Tokenizer>,
}

impl OnnxCompressor {
    /// Configure a compressor pointing at `model_path` (an `.onnx` file) and
    /// `tokenizer_path` (a `tokenizer.json`). The session and tokenizer load
    /// lazily on the first [`SemanticCompressor::compress`] call.
    #[must_use]
    pub fn new(model_path: impl Into<PathBuf>, tokenizer_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            tokenizer_path: tokenizer_path.into(),
            session: Mutex::new(None),
            tokenizer: OnceLock::new(),
        }
    }

    /// The configured ONNX model path.
    #[must_use]
    pub fn model_path(&self) -> &Path {
        self.model_path.as_path()
    }

    /// The configured tokenizer path.
    #[must_use]
    pub fn tokenizer_path(&self) -> &Path {
        self.tokenizer_path.as_path()
    }

    /// Lazy-load the tokenizer (once) and return a shared reference.
    fn tokenizer(&self) -> Result<&Tokenizer, LtkError> {
        if let Some(tok) = self.tokenizer.get() {
            return Ok(tok);
        }
        let tok = Tokenizer::from_file(&self.tokenizer_path)
            .map_err(|e| LtkError::Onnx(format!("tokenizer load failed: {e}")))?;
        // `get_or_init` is race-free; the first writer wins, later callers
        // reuse the stored tokenizer.
        Ok(self.tokenizer.get_or_init(|| tok))
    }

    /// Lazy-load the ONNX session (once) behind the mutex and run inference
    /// for `inputs`, returning the raw `logits` flat buffer.
    fn run_inference<'i, 'v>(
        &self,
        inputs: ort::session::SessionInputs<'i, 'v>,
    ) -> Result<Vec<f32>, LtkError> {
        let mut guard = self
            .session
            .lock()
            .map_err(|e| LtkError::Onnx(format!("session lock poisoned: {e}")))?;
        if guard.is_none() {
            let session = Session::builder()
                .map_err(|e| LtkError::Onnx(format!("session builder: {e}")))?
                .commit_from_file(&self.model_path)
                .map_err(|e| LtkError::Onnx(format!("session load failed: {e}")))?;
            *guard = Some(session);
        }
        // `guard` is `Option<Session>`; we just ensured it is `Some`.
        let session = guard
            .as_mut()
            .ok_or_else(|| LtkError::Onnx("unreachable: session not initialized".into()))?;
        let outputs =
            session.run(inputs).map_err(|e| LtkError::Onnx(format!("inference failed: {e}")))?;
        let logits_value = &outputs["logits"];
        let (_shape, logits) = logits_value
            .try_extract_tensor::<f32>()
            .map_err(|e| LtkError::Onnx(format!("logits extract failed: {e}")))?;
        Ok(logits.to_vec())
    }
}

impl std::fmt::Debug for OnnxCompressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxCompressor")
            .field("model_path", &self.model_path)
            .field("tokenizer_path", &self.tokenizer_path)
            .finish_non_exhaustive()
    }
}

impl SemanticCompressor for OnnxCompressor {
    fn compress(&self, text: &str, target_rate: f32) -> Result<String, LtkError> {
        if text.is_empty() {
            return Ok(String::new());
        }

        let tok = self.tokenizer()?;
        let encoding =
            tok.encode(text, true).map_err(|e| LtkError::Onnx(format!("encode failed: {e}")))?;

        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        let special = encoding.get_special_tokens_mask();

        // Truncate to the model's max context so long log lines don't overflow.
        let n = ids.len().min(MAX_SEQ_LEN);
        if n == 0 {
            return Ok(String::new());
        }

        let ids_i64: Vec<i64> = ids[..n].iter().map(|&v| v as i64).collect();
        let mask_i64: Vec<i64> = mask[..n].iter().map(|&v| v as i64).collect();

        let ids_tensor = Tensor::from_array(([1usize, n], ids_i64))
            .map_err(|e| LtkError::Onnx(format!("input_ids tensor: {e}")))?;
        let mask_tensor = Tensor::from_array(([1usize, n], mask_i64))
            .map_err(|e| LtkError::Onnx(format!("attention_mask tensor: {e}")))?;

        let inputs = ort::inputs! {
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor,
        };

        let logits = self.run_inference(inputs.into())?;

        // Logits are flat `[1, n, num_classes]`; infer the class count rather
        // than trust the ONNX `Shape` accessor (kept private in ort 2.0).
        if logits.len() < n {
            return Err(LtkError::Onnx(format!("logits buffer too short: {} < {n}", logits.len())));
        }
        let nc = logits.len() / n;
        if nc <= DROP_CLASS {
            return Err(LtkError::Onnx(format!(
                "model has {nc} output classes; expected at least {}",
                DROP_CLASS + 1
            )));
        }

        let drop_probs = drop_probs(&logits, n, nc);
        let keep = select_keeps(&drop_probs, &special[..n], target_rate);

        // Preserve original token order, dropping the pruned tokens.
        let surviving: Vec<u32> =
            ids[..n].iter().zip(keep.iter()).filter(|(_, k)| **k).map(|(id, _)| *id).collect();

        // `skip_special_tokens = true` so CLS/SEP don't leak into the output.
        tok.decode(&surviving, true).map_err(|e| LtkError::Onnx(format!("decode failed: {e}")))
    }
}

/// Per-token softmax drop-probability for the `DROP_CLASS` index.
///
/// `logits` is the flat `[n, nc]` buffer (row-major). Returns a `Vec<f32>` of
/// length `n` where each entry is `softmax(row)[DROP_CLASS]` — the model's
/// estimated probability that the token should be dropped.
fn drop_probs(logits: &[f32], n: usize, nc: usize) -> Vec<f32> {
    (0..n)
        .map(|t| {
            let row = &logits[t * nc..(t + 1) * nc];
            // Numerically stable softmax: subtract row max before exp.
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0_f32;
            let mut drop_exp = 0.0_f32;
            for (i, &v) in row.iter().enumerate() {
                let e = (v - max).exp();
                sum += e;
                if i == DROP_CLASS {
                    drop_exp = e;
                }
            }
            if sum > 0.0 { drop_exp / sum } else { 0.0 }
        })
        .collect()
}

/// Decide which tokens to keep.
///
/// - Special tokens (`special_mask[i] != 0`) are always kept.
/// - Among non-special tokens, the `ceil(target_rate * count)` with the *lowest* drop probability
///   are kept (highest information); the rest are pruned.
/// - `target_rate` is clamped to `[0.0, 1.0]`.
fn select_keeps(drop_probs: &[f32], special_mask: &[u32], target_rate: f32) -> Vec<bool> {
    let n = drop_probs.len();
    let mut keep = vec![false; n];

    // Always keep special tokens.
    let mut non_special: Vec<usize> = Vec::new();
    for i in 0..n {
        if special_mask.get(i).copied().unwrap_or(0) != 0 {
            keep[i] = true;
        } else {
            non_special.push(i);
        }
    }

    let total = non_special.len();
    let rate = target_rate.clamp(0.0, 1.0);
    // `ceil` so a non-zero rate never collapses to zero keeps on rounding.
    let keep_count = ((rate * total as f32).ceil() as usize).min(total);

    // Lowest drop-probability first → most informative tokens win.
    non_special.sort_by(|&a, &b| {
        drop_probs[a].partial_cmp(&drop_probs[b]).unwrap_or(std::cmp::Ordering::Equal)
    });
    for &i in non_special.iter().take(keep_count) {
        keep[i] = true;
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a flat `[n, 2]` logits buffer from per-token `(keep, drop)` pairs.
    fn logits_2class(pairs: &[(f32, f32)]) -> Vec<f32> {
        let mut out = Vec::with_capacity(pairs.len() * 2);
        for &(keep, drop) in pairs {
            out.push(keep);
            out.push(drop);
        }
        out
    }

    #[test]
    fn drop_probs_softmax_sums_to_one_per_token() {
        // logits for 3 tokens, 2 classes each.
        let pairs = [(0.0_f32, 0.0_f32), (1.0_f32, 2.0_f32), (-3.0_f32, 3.0_f32)];
        let logits = logits_2class(&pairs);
        let probs = drop_probs(&logits, pairs.len(), 2);
        assert_eq!(probs.len(), pairs.len());
        // For [0,0], softmax is uniform → drop prob 0.5.
        assert!((probs[0] - 0.5).abs() < 1e-6);
        // For [1,2], drop (class 1) should dominate: e^2/(e^1+e^2).
        let expected = (2.0_f32).exp() / ((1.0_f32).exp() + (2.0_f32).exp());
        assert!((probs[1] - expected).abs() < 1e-6);
        // Drop prob is always in [0,1].
        for &p in &probs {
            assert!((0.0..=1.0).contains(&p));
        }
    }

    #[test]
    fn select_keeps_preserves_special_tokens_at_zero_rate() {
        // 4 tokens: special at index 0 and 3, non-special at 1 and 2.
        let drop_probs = vec![0.9, 0.1, 0.8, 0.9];
        let special = vec![1, 0, 0, 1];
        let keep = select_keeps(&drop_probs, &special, 0.0);
        // Rate 0 → keep no non-special tokens; specials always kept.
        assert_eq!(keep, vec![true, false, false, true]);
    }

    #[test]
    fn select_keeps_keeps_all_non_special_at_rate_one() {
        let drop_probs = vec![0.9, 0.1, 0.8, 0.9];
        let special = vec![1, 0, 0, 1];
        let keep = select_keeps(&drop_probs, &special, 1.0);
        assert!(keep.iter().all(|&k| k));
    }

    #[test]
    fn select_keeps_prefers_lowest_drop_probability() {
        // 2 non-special tokens: drop probs 0.9 (index 1) and 0.1 (index 2).
        // rate 0.5 → keep ceil(0.5 * 2) = 1 → the lowest-drop one (index 2).
        let drop_probs = vec![0.0, 0.9, 0.1, 0.0];
        let special = vec![1, 0, 0, 1];
        let keep = select_keeps(&drop_probs, &special, 0.5);
        assert!(keep[0] && keep[3], "specials kept");
        assert!(!keep[1], "high-drop non-special pruned");
        assert!(keep[2], "low-drop non-special kept");
    }

    #[test]
    fn select_keeps_clamps_negative_and_oversized_rate() {
        let drop_probs = vec![0.5, 0.5];
        let special = vec![0, 0];
        let none = select_keeps(&drop_probs, &special, -1.0);
        assert_eq!(none, vec![false, false]);
        let all = select_keeps(&drop_probs, &special, 2.0);
        assert_eq!(all, vec![true, true]);
    }

    #[test]
    fn select_keeps_empty_input() {
        let keep = select_keeps(&[], &[], 0.5);
        assert!(keep.is_empty());
    }

    #[test]
    fn select_keeps_ceil_rounds_up_nonzero_rate() {
        // 3 non-special tokens, rate 0.34 → ceil(0.34*3)=ceil(1.02)=2 kept.
        let drop_probs = vec![0.9, 0.5, 0.1];
        let special = vec![0, 0, 0];
        let keep = select_keeps(&drop_probs, &special, 0.34);
        assert_eq!(keep.iter().filter(|&&k| k).count(), 2);
        // The two lowest-drop (0.1, 0.5) survive; 0.9 is pruned.
        assert!(keep[2] && keep[1]);
        assert!(!keep[0]);
    }

    #[test]
    #[ignore = "requires an ONNX model + tokenizer; set LTK_ONNX_MODEL and LTK_ONNX_TOKENIZER to run"]
    fn compress_round_trips_against_local_model() -> Result<(), Box<dyn std::error::Error>> {
        let model = std::env::var("LTK_ONNX_MODEL")?;
        let tokenizer = std::env::var("LTK_ONNX_TOKENIZER")?;
        let comp = OnnxCompressor::new(&model, &tokenizer);
        let out = comp.compress("the quick brown fox jumps over the lazy dog", 0.5)?;
        assert!(!out.is_empty());
        Ok(())
    }
}
