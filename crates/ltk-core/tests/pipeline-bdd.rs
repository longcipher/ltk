//! BDD scenarios for the ltk log compression pipeline.

use std::path::PathBuf;

use cucumber::{World, given, then, when};
use ltk_core::{
    Clusterer, CompactFormatter, Formatter, JaccardClusterer, JsonMinimalFormatter, LogCluster,
    Normalizer, OutputFormat, RegexNormalizer, Stats, ToonFormatter, TsvFormatter,
};

#[derive(Debug, Default, World)]
struct PipelineWorld {
    output: Option<String>,
    stats: Option<Stats>,
    clusters: Vec<LogCluster>,
    format: OutputFormat,
}

#[given("a pipeline with default settings")]
async fn default_pipeline(world: &mut PipelineWorld) {
    world.format = OutputFormat::Compact;
}

#[given(expr = "a pipeline with {word} format")]
async fn pipeline_with_format(world: &mut PipelineWorld, format: String) {
    world.format = match format.as_str() {
        "compact" => OutputFormat::Compact,
        "tsv" => OutputFormat::Tsv,
        "toon" => OutputFormat::Toon,
        "json-minimal" => OutputFormat::JsonMinimal,
        _ => OutputFormat::Compact,
    };
}

fn run_pipeline(world: &mut PipelineWorld, lines: Vec<String>) {
    let normalizer = RegexNormalizer::new();
    assert!(normalizer.is_ok(), "normalizer init failed");
    if let Ok(normalizer) = normalizer {
        let mut clusterer = JaccardClusterer::new();
        let mut raw_lines: u64 = 0;
        let mut raw_bytes: u64 = 0;
        let mut raw_tokens: u64 = 0;

        for line in &lines {
            raw_lines += 1;
            raw_bytes += u64::try_from(line.len()).unwrap_or(u64::MAX);
            raw_tokens += u64::try_from(ltk_core::estimate_tokens(line)).unwrap_or(u64::MAX);
            let normalized = normalizer.normalize(line);
            clusterer.ingest(normalized);
        }

        let clusters = clusterer.finish();
        let cluster_count = clusters.len();

        let formatter: Box<dyn Formatter> = match world.format {
            OutputFormat::Compact => Box::new(CompactFormatter::new()),
            OutputFormat::Tsv => Box::new(TsvFormatter::new()),
            OutputFormat::Toon => Box::new(ToonFormatter::new()),
            OutputFormat::JsonMinimal => Box::new(JsonMinimalFormatter::new()),
        };

        let stage1_output = formatter.format(&clusters);
        let stage1_tokens =
            u64::try_from(ltk_core::estimate_tokens(&stage1_output)).unwrap_or(u64::MAX);

        let stats = Stats {
            raw_lines,
            raw_bytes,
            cluster_count,
            raw_tokens,
            stage1_tokens,
            stage2_tokens: None,
            lost_lines: 0,
        };

        world.output = Some(stage1_output);
        world.stats = Some(stats);
        world.clusters = clusters;
    }
}

#[when(expr = "the pipeline processes {int} similar error lines")]
async fn process_similar_error_lines(world: &mut PipelineWorld, count: usize) {
    let lines: Vec<String> = (0..count)
        .map(|i| format!("2026-07-31T10:00:0{i}Z [ERROR] connect to 192.168.1.{i}:5432 refused"))
        .collect();
    run_pipeline(world, lines);
}

#[when(expr = "the pipeline processes {int} dissimilar lines")]
async fn process_dissimilar_lines(world: &mut PipelineWorld, count: usize) {
    let lines: Vec<String> = match count {
        2 => vec![
            "connection refused to 192.168.1.10:5432".to_owned(),
            "disk usage at 95 percent on /var/log".to_owned(),
        ],
        _ => vec!["single line".to_owned(); count],
    };
    run_pipeline(world, lines);
}

#[when(expr = "the pipeline processes {int} identical lines")]
async fn process_identical_lines(world: &mut PipelineWorld, count: usize) {
    let lines: Vec<String> = vec!["error at host-alpha".to_owned(); count];
    run_pipeline(world, lines);
}

#[when(expr = "the pipeline processes {int} ansi-styled line")]
async fn process_ansi_lines(world: &mut PipelineWorld, count: usize) {
    let lines: Vec<String> = vec!["\x1b[31mconnection error\x1b[0m".to_owned(); count];
    run_pipeline(world, lines);
}

#[when(expr = "the pipeline processes {int} lines")]
async fn process_n_lines(world: &mut PipelineWorld, count: usize) {
    if count == 0 {
        run_pipeline(world, vec![]);
    } else {
        let lines: Vec<String> = vec!["some log line".to_owned(); count];
        run_pipeline(world, lines);
    }
}

#[then(expr = "the output should contain {int} cluster")]
#[then(expr = "the output should contain {int} clusters")]
async fn output_cluster_count(world: &mut PipelineWorld, count: usize) {
    assert!(world.stats.is_some(), "no pipeline output; did the 'When' step run?");
    if let Some(stats) = world.stats.as_ref() {
        assert_eq!(
            stats.cluster_count, count,
            "expected {count} clusters, got {}",
            stats.cluster_count
        );
    }
}

#[then(expr = "the cluster occurrence count should be {int}")]
async fn cluster_occurrence_count(world: &mut PipelineWorld, count: u64) {
    assert!(!world.clusters.is_empty(), "no clusters produced");
    if let Some(cluster) = world.clusters.first() {
        assert_eq!(cluster.count, count, "expected cluster count {count}, got {}", cluster.count);
    }
}

#[then(expr = "the output should contain {string}")]
async fn output_contains(world: &mut PipelineWorld, needle: String) {
    assert!(world.output.is_some(), "no pipeline output");
    if let Some(output) = world.output.as_ref() {
        assert!(output.contains(&needle), "output does not contain {needle:?}; got: {output:?}");
    }
}

#[then(expr = "the output should start with {string}")]
async fn output_starts_with(world: &mut PipelineWorld, prefix: String) {
    assert!(world.output.is_some(), "no pipeline output");
    let prefix = prefix.replace("\\t", "\t");
    if let Some(output) = world.output.as_ref() {
        assert!(
            output.starts_with(&prefix),
            "output does not start with {prefix:?}; got: {output:?}"
        );
    }
}

#[then("the output should not contain escape bytes")]
async fn output_no_escapes(world: &mut PipelineWorld) {
    assert!(world.output.is_some(), "no pipeline output");
    if let Some(output) = world.output.as_ref() {
        assert!(!output.contains('\x1b'), "output contains escape bytes; got: {output:?}");
    }
}

#[then("the output should be empty")]
async fn output_empty(world: &mut PipelineWorld) {
    assert!(world.output.is_some(), "no pipeline output");
    if let Some(output) = world.output.as_ref() {
        assert!(output.trim().is_empty(), "expected empty output, got: {output:?}");
    }
}

#[tokio::main]
async fn main() {
    let feature_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../features");
    PipelineWorld::run(feature_path.as_path()).await;
}
