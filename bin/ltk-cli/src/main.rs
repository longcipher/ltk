//! `ltk` CLI — Less Token cascade log compression.

#![expect(clippy::print_stdout, reason = "CLI output goes to stdout")]
#![expect(clippy::print_stderr, reason = "Stats goes to stderr")]

use std::{
    io::{self, Read},
    path::PathBuf,
};

use clap::{Parser, Subcommand, ValueEnum};
use eyre::Result;

/// Less Token — cascade log compression for LLM/agent contexts.
///
/// Reads log lines from stdin or files, normalizes and clusters them
/// via rule-based similarity, optionally applies neural pruning (Stage 2),
/// and emits token-efficient output.
#[derive(Debug, Parser)]
#[command(name = "ltk", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Count tokens in raw vs compressed files using HuggingFace BPE.
    ///
    /// Trains a BPE tokenizer on the raw files, then counts tokens in both
    /// raw and compressed files and reports savings.
    #[cfg(feature = "token-counter")]
    TokenCounter {
        /// Raw input files (used for training the tokenizer).
        #[arg(required = true)]
        raw: Vec<PathBuf>,

        /// Separator between raw and compressed file lists.
        /// Must be "--" followed by compressed files.
        /// If omitted, only raw files are tokenized.
        #[arg(long)]
        compressed: Vec<PathBuf>,
    },
}

/// Top-level CLI arguments when no subcommand is given (pipeline mode).
#[derive(Debug, Parser)]
struct PipelineArgs {
    /// Output format.
    #[arg(short, long, value_enum, default_value_t = Format::Compact)]
    format: Format,

    /// Stage 2 target keep-rate (0.0–1.0). Only effective with --rpc-endpoint
    /// or --onnx-model. Keeps ~this fraction of tokens after neural pruning.
    #[arg(short = 'r', long, default_value_t = 0.5)]
    target_rate: f32,

    /// Maximum number of live clusters (0 = unbounded). When the window fills,
    /// the least-recently-used cluster is evicted and its count is merged into
    /// the nearest surviving cluster.
    #[arg(short, long, default_value_t = 0)]
    window: usize,

    /// Jaccard similarity threshold for matching (0.0–1.0).
    #[arg(long, default_value_t = 0.7)]
    threshold: f64,

    /// Print token-savings statistics to stderr after processing.
    #[arg(long)]
    stats: bool,

    /// Path to an ONNX token-classification model for Stage 2 neural pruning.
    /// Requires the llmlingua-onnx feature.
    #[arg(long)]
    onnx_model: Option<PathBuf>,

    /// Path to a HuggingFace tokenizer.json for Stage 2 ONNX inference.
    /// Required when --onnx-model is set.
    #[arg(long)]
    onnx_tokenizer: Option<PathBuf>,

    /// URL of a LLMLingua HTTP daemon for Stage 2 remote neural pruning.
    /// Requires the llmlingua-rpc feature.
    #[arg(long)]
    rpc_endpoint: Option<String>,

    /// Input log files. Reads from stdin if none are provided.
    files: Vec<PathBuf>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Format {
    /// `[xN] <template>` — one line per cluster (default).
    Compact,
    /// Tab-separated: `count\ttemplate`.
    Tsv,
    /// TOON (tagged object-optimized notation) for LLM-friendly structured logs.
    Toon,
    /// Minimal JSON array of `{count, template}` objects.
    JsonMinimal,
}

impl From<Format> for ltk_core::OutputFormat {
    fn from(f: Format) -> Self {
        match f {
            Format::Compact => Self::Compact,
            Format::Tsv => Self::Tsv,
            Format::Toon => Self::Toon,
            Format::JsonMinimal => Self::JsonMinimal,
        }
    }
}

fn formatter_for(format: ltk_core::OutputFormat) -> Box<dyn ltk_core::Formatter> {
    match format {
        ltk_core::OutputFormat::Compact => Box::new(ltk_core::CompactFormatter::new()),
        ltk_core::OutputFormat::Tsv => Box::new(ltk_core::TsvFormatter::new()),
        ltk_core::OutputFormat::Toon => Box::new(ltk_core::ToonFormatter::new()),
        ltk_core::OutputFormat::JsonMinimal => Box::new(ltk_core::JsonMinimalFormatter::new()),
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    if let Some(cmd) = cli.command {
        match cmd {
            #[cfg(feature = "token-counter")]
            Commands::TokenCounter { raw, compressed } => {
                return run_token_counter(&raw, &compressed)
            }
        }
    }

    run_pipeline(PipelineArgs::parse_from(std::env::args_os()))
}

fn run_pipeline(args: PipelineArgs) -> Result<()> {
    // Collect input lines from stdin or files.
    let lines: Vec<String> = if args.files.is_empty() {
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf)?;
        buf.lines().map(str::to_owned).collect()
    } else {
        let mut all = Vec::new();
        for path in &args.files {
            let content = std::fs::read_to_string(path)?;
            all.extend(content.lines().map(str::to_owned));
        }
        all
    };

    // Build pipeline components.
    let normalizer = ltk_core::RegexNormalizer::new()?;
    let clusterer = ltk_core::JaccardClusterer::with_params(args.threshold, args.window);
    let format: ltk_core::OutputFormat = args.format.into();
    let formatter = formatter_for(format);

    let pipeline = ltk_core::Pipeline::new(normalizer, clusterer, formatter)
        .with_target_rate(args.target_rate);

    // Stage 2: attach neural compressor if requested (re-bind to avoid unused mut).
    #[cfg(feature = "llmlingua-onnx")]
    let pipeline = if let (Some(model), Some(tokenizer)) = (&args.onnx_model, &args.onnx_tokenizer)
    {
        let comp = std::sync::Arc::new(ltk_core::OnnxCompressor::new(model, tokenizer));
        pipeline.with_compressor(comp)
    } else {
        pipeline
    };

    #[cfg(feature = "llmlingua-rpc")]
    let pipeline = if let Some(endpoint) = &args.rpc_endpoint {
        let comp = std::sync::Arc::new(ltk_core::RpcCompressor::new(endpoint)?);
        pipeline.with_compressor(comp)
    } else {
        pipeline
    };

    let (output, stats) = pipeline.run(&lines)?;

    // Emit output.
    println!("{output}");

    // Emit stats to stderr if requested.
    if args.stats {
        eprintln!("--- ltk stats ---");
        eprintln!("raw lines:       {}", stats.raw_lines);
        eprintln!("raw bytes:       {}", stats.raw_bytes);
        eprintln!("clusters:        {}", stats.cluster_count);
        eprintln!("raw tokens:      {}", stats.raw_tokens);
        eprintln!("stage1 tokens:   {}", stats.stage1_tokens);
        let pct1 = (1.0 - stats.stage1_retention()) * 100.0;
        eprintln!("stage1 savings:  {pct1:.1}%");
        if let Some(s2) = stats.stage2_tokens {
            eprintln!("stage2 tokens:   {s2}");
            let pct2 = stats.stage2_retention().map_or(0.0, |r| (1.0 - r) * 100.0);
            eprintln!("stage2 savings:  {pct2:.1}%");
        }
        if stats.lost_lines > 0 {
            eprintln!("lost lines:      {} (merged into nearest cluster)", stats.lost_lines);
        }
    }

    Ok(())
}

#[cfg(feature = "token-counter")]
fn run_token_counter(raw: &[PathBuf], compressed: &[PathBuf]) -> Result<()> {
    use tokenizers::{
        AddedToken, TokenizerImpl,
        decoders::DecoderWrapper,
        models::bpe::{BPE, BpeTrainer},
        normalizers::NormalizerWrapper,
        pre_tokenizers::{PreTokenizerWrapper, whitespace::Whitespace},
        processors::PostProcessorWrapper,
    };

    type BpeTokenizer = TokenizerImpl<
        BPE,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >;

    fn build_tokenizer(files: &[PathBuf]) -> BpeTokenizer {
        let mut tokenizer = TokenizerImpl::new(BPE::default());
        tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Whitespace(Whitespace)));

        let mut trainer = BpeTrainer::builder()
            .special_tokens(vec![
                AddedToken::from("[UNK]", false),
                AddedToken::from("[CLS]", false),
                AddedToken::from("[SEP]", false),
                AddedToken::from("[PAD]", false),
                AddedToken::from("[MASK]", false),
            ])
            .vocab_size(10000)
            .min_frequency(2)
            .build();

        let paths: Vec<String> =
            files.iter().filter_map(|p| p.to_str().map(String::from)).collect();
        if let Err(e) = tokenizer.train_from_files(&mut trainer, paths) {
            eprintln!("tokenizer training failed: {e}");
            std::process::exit(1);
        }
        tokenizer
    }

    fn count_tokens(tokenizer: &BpeTokenizer, text: &str) -> usize {
        match tokenizer.encode(text, false) {
            Ok(encoding) => encoding.get_tokens().len(),
            Err(_) => 0,
        }
    }

    fn fmt_num(n: usize) -> String {
        let s = n.to_string();
        let mut out = String::new();
        for (i, c) in s.chars().rev().enumerate() {
            if i > 0 && i % 3 == 0 {
                out.push(',');
            }
            out.push(c);
        }
        out.chars().rev().collect()
    }

    eprintln!("Training BPE tokenizer on {} files...", raw.len());
    let tokenizer = build_tokenizer(raw);
    let vocab_size = tokenizer.get_vocab_size(true);
    eprintln!("Vocabulary size: {vocab_size}");

    println!("=== Token Count Results (HuggingFace tokenizers BPE) ===\n");

    for (i, path) in raw.iter().enumerate() {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Failed to read {}: {e}", path.display());
                continue;
            }
        };
        let size = text.len();
        let tokens = count_tokens(&tokenizer, &text);
        let lines = text.lines().count();
        let name = path
            .file_name()
            .map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned());
        println!("  {name}:");
        println!("    Lines:  {:>8}", fmt_num(lines));
        println!("    Size:   {:>8} bytes ({:.1} KB)", fmt_num(size), size as f64 / 1024.0);
        println!("    Tokens: {:>8}", fmt_num(tokens));

        if i < compressed.len() {
            let comp_text = match std::fs::read_to_string(&compressed[i]) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Failed to read {}: {e}", compressed[i].display());
                    continue;
                }
            };
            let comp_size = comp_text.len();
            let comp_tokens = count_tokens(&tokenizer, &comp_text);
            let comp_lines = comp_text.lines().count();
            let comp_name = compressed[i].file_name().map_or_else(
                || compressed[i].display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            );
            println!("  {comp_name} (compressed):");
            println!("    Lines:  {:>8}", fmt_num(comp_lines));
            println!(
                "    Size:   {:>8} bytes ({:.1} KB)",
                fmt_num(comp_size),
                comp_size as f64 / 1024.0
            );
            println!("    Tokens: {:>8}", fmt_num(comp_tokens));

            let size_pct = (1.0 - comp_size as f64 / size as f64) * 100.0;
            let token_pct = (1.0 - comp_tokens as f64 / tokens as f64) * 100.0;
            let line_pct = (1.0 - comp_lines as f64 / lines as f64) * 100.0;
            println!("    Size savings:   {size_pct:.1}%");
            println!("    Token savings:  {token_pct:.1}%");
            println!("    Line savings:   {line_pct:.1}%");
        }
        println!();
    }

    Ok(())
}
