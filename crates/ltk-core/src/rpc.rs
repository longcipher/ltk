//! Stage 2 remote LLMLingua compression over HTTP.
//!
//! [`RpcCompressor`] delegates neural semantic pruning to a local LLMLingua
//! sidecar (e.g. a small Python `llmlingua` daemon) exposing a JSON endpoint:
//!
//! ```text
//! POST {endpoint}
//!   {"text": "<log line>", "target_rate": 0.4}
//! →
//!   {"compressed": "<pruned line>"}
//! ```
//!
//! ## Sync/async bridge
//!
//! [`hpx::Client`] is async-only, but [`SemanticCompressor::compress`] is a
//! synchronous trait method (the [`Pipeline`](crate::Pipeline) is sync). Each
//! compressor therefore owns a dedicated single-threaded tokio runtime and
//! drives the request future to completion with [`Runtime::block_on`].
//!
//! Calling `compress` from *within* an existing tokio runtime would panic
//! (`block_on` cannot re-enter a runtime), so [`compress`](RpcCompressor::compress)
//! detects that case and returns a clean [`LtkError::Rpc`] instead of panicking.
//! Embed `ltk-core` in an async app by spawning the pipeline on a blocking thread
//! (`tokio::task::spawn_blocking`).

use serde::{Deserialize, Serialize};
use tokio::runtime::{Builder, Handle, Runtime};

use crate::{LtkError, SemanticCompressor};

/// JSON request body sent to the LLMLingua daemon.
#[derive(Debug, Serialize)]
struct CompressRequest<'a> {
    text: &'a str,
    target_rate: f32,
}

/// JSON response body returned by the LLMLingua daemon.
#[derive(Debug, Deserialize)]
struct CompressResponse {
    compressed: String,
}

/// Stage 2 compressor delegating to a remote LLMLingua HTTP daemon.
///
/// Construct with [`RpcCompressor::new`], then pass an `Arc<dyn
/// SemanticCompressor>` to [`Pipeline::with_compressor`](crate::Pipeline::with_compressor).
///
/// ```no_run
/// use std::sync::Arc;
/// use ltk_core::{Pipeline, OutputFormat, RpcCompressor, RegexNormalizer, JaccardClusterer};
/// # use ltk_core::Formatter;
/// # struct F; impl Formatter for F { fn format(&self, _: &[ltk_core::LogCluster]) -> String { String::new() } }
///
/// let comp = Arc::new(RpcCompressor::new("http://127.0.0.1:8080/compress")?);
/// let (out, stats) = Pipeline::new(RegexNormalizer::new()?, JaccardClusterer::new(), F)
///     .with_compressor(comp)
///     .with_target_rate(0.4)
///     .run(["2026-07-31 ERROR connect 10.0.0.1 refused"])?;
/// # let _ = (out, stats);
/// # Ok::<(), ltk_core::LtkError>(())
/// ```
pub struct RpcCompressor {
    endpoint: String,
    client: hpx::Client,
    runtime: Runtime,
}

impl RpcCompressor {
    /// Create a compressor targeting `endpoint` (e.g.
    /// `http://127.0.0.1:8080/compress`).
    ///
    /// # Errors
    /// Returns [`LtkError::Rpc`] if the internal tokio runtime fails to
    /// initialize (only on OS resource exhaustion).
    pub fn new(endpoint: impl Into<String>) -> Result<Self, LtkError> {
        let runtime = Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| LtkError::Rpc(format!("tokio runtime init failed: {e}")))?;
        Ok(Self { endpoint: endpoint.into(), client: hpx::Client::new(), runtime })
    }

    /// The daemon endpoint URL.
    #[must_use]
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Build the JSON request body for a compress call. Exposed for testing
    /// and for callers that drive the HTTP round-trip themselves.
    #[must_use]
    pub fn build_request(&self, text: &str, target_rate: f32) -> serde_json::Value {
        serde_json::json!({ "text": text, "target_rate": target_rate })
    }
}

impl std::fmt::Debug for RpcCompressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RpcCompressor").field("endpoint", &self.endpoint).finish_non_exhaustive()
    }
}

impl SemanticCompressor for RpcCompressor {
    fn compress(&self, text: &str, target_rate: f32) -> Result<String, LtkError> {
        // `block_on` panics if re-entered inside an existing tokio runtime;
        // surface that as a clean error rather than panicking.
        if Handle::try_current().is_ok() {
            return Err(LtkError::Rpc(
                "compress() must not be called from within a tokio runtime; \
                 run the pipeline via spawn_blocking instead"
                    .into(),
            ));
        }

        let body = CompressRequest { text, target_rate };
        let endpoint = self.endpoint.as_str();
        let client = &self.client;

        self.runtime.block_on(async move {
            let resp = client
                .post(endpoint)
                .json(&body)
                .send()
                .await
                .map_err(|e| LtkError::Rpc(format!("http send failed: {e}")))?;
            let resp = resp
                .error_for_status()
                .map_err(|e| LtkError::Rpc(format!("daemon returned non-2xx: {e}")))?;
            let parsed: CompressResponse = resp
                .json()
                .await
                .map_err(|e| LtkError::Rpc(format!("response decode failed: {e}")))?;
            Ok(parsed.compressed)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

    #[test]
    fn build_request_carries_text_and_rate() -> Result<(), Box<dyn Error>> {
        let comp = RpcCompressor::new("http://127.0.0.1:8080/compress")?;
        let req = comp.build_request("hello world", 0.4);
        assert_eq!(req["text"], "hello world");
        assert!((req["target_rate"].as_f64().unwrap_or(0.0) - 0.4).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn response_decodes_compressed_field() -> Result<(), Box<dyn Error>> {
        let raw = r#"{"compressed":"pruned line"}"#;
        let parsed: CompressResponse = serde_json::from_str(raw)?;
        assert_eq!(parsed.compressed, "pruned line");
        Ok(())
    }

    #[test]
    fn response_rejects_missing_field() {
        let raw = r#"{"text":"pruned line"}"#;
        let res: Result<CompressResponse, _> = serde_json::from_str(raw);
        assert!(res.is_err());
    }

    #[test]
    fn compress_refuses_runtime_reentry() -> Result<(), Box<dyn Error>> {
        // Inside a tokio runtime, compress must return a clean error instead
        // of panicking on block_on reentry.
        let rt = Builder::new_current_thread().enable_all().build()?;
        let comp = RpcCompressor::new("http://127.0.0.1:8080/compress")?;
        let res: Result<String, LtkError> = rt.block_on(async { comp.compress("x", 0.5) });
        assert!(res.is_err());
        assert!(matches!(res, Err(LtkError::Rpc(_))));
        Ok(())
    }

    #[test]
    #[ignore = "requires a live LLMLingua daemon; set LTK_RPC_ENDPOINT to run"]
    fn compress_round_trips_against_live_daemon() -> Result<(), Box<dyn Error>> {
        let endpoint = std::env::var("LTK_RPC_ENDPOINT")?;
        let comp = RpcCompressor::new(&endpoint)?;
        let out = comp.compress("the quick brown fox jumps over the lazy dog", 0.5)?;
        assert!(!out.is_empty());
        Ok(())
    }
}
