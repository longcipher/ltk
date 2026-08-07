# ltk — Less Token

Cascade log compression pipeline (rule-based clustering + optional LLMLingua neural pruning) for LLM/agent contexts. Saves 50%–95% of tokens when compressing log streams.

## Pipeline

```text
raw line → Normalizer → Clusterer → (optional) SemanticCompressor → Formatter → stdout
```

- **Stage 0 — Normalizer**: strips ANSI, masks volatile sub-tokens (IP, UUID, timestamp, hex, path, number, port) into stable placeholders.
- **Stage 1 — Clusterer**: Jaccard-similarity online clustering with move-to-front LRU and frozen exemplars. Groups near-identical lines into parameterized templates with occurrence counts.
- **Stage 2 — SemanticCompressor** *(optional, feature-gated)*: LLMLingua-style neural token pruning via ONNX Runtime or remote HTTP daemon.
- **Stage 3 — Formatter**: renders clusters into `compact`, `tsv`, `toon`, or `json-minimal` format.

## Compression Results

Tested on real Discord and Telegram chat export files, tokenized with HuggingFace BPE (10k vocabulary, `tokenizers` crate).

### Discord Archive (~3k messages, multi-channel)

| Metric | Raw | Compressed | Savings |
|--------|-----|------------|---------|
| Lines | 6,934 | 3,267 | 52.9% |
| Size | 432 KB | 210 KB | 51.4% |
| Tokens (BPE) | 162,848 | 74,938 | 54.0% |

### Telegram Export (~100k messages, support bot heavy)

| Metric | Raw | Compressed | Savings |
|--------|-----|------------|---------|
| Lines | 101,356 | 17,654 | 82.6% |
| Size | 3,078 KB | 1,074 KB | 65.1% |
| Tokens (BPE) | 871,494 | 341,622 | 60.8% |

Telegram achieves higher compression because its 100k messages are highly repetitive (support bot greetings, FAQ answers, repeated Q&A patterns). The clusterer collapses these into templates with high occurrence counts like `[x21457]` and `[x8354]`.

Discord has more diverse conversation content across multiple channels, so fewer lines collapse into the same cluster.

**Data integrity**: all `[xN]` counts sum to the exact original line count — zero information loss.

## Quick Start

```bash
# Build
cargo build -p ltk-cli

# Compress a log file (compact format, stats to stderr)
./target/debug/ltk --stats input.log > compressed.txt

# Compress with TSV output
./target/debug/ltk --format tsv input.log > output.tsv

# Compress with custom window and threshold
./target/debug/ltk --window 5000 --threshold 0.8 --stats input.log > output.txt

# Pipe from stdin
cat *.log | ./target/debug/ltk --stats > compressed.txt

# Token counter (requires --features token-counter)
cargo build -p ltk-cli --features token-counter
./target/debug/ltk token-counter raw1.md raw2.md --compressed compressed1.txt --compressed compressed2.txt
```

## CLI Options

```
Usage: ltk [OPTIONS] [FILES]...
       ltk token-counter <RAW>... --compressed <COMPRESSED>...

Options:
  -f, --format <FORMAT>          Output format: compact, tsv, toon, json-minimal [default: compact]
  -r, --target-rate <RATE>       Stage 2 keep-rate (0.0–1.0) [default: 0.5]
  -w, --window <WINDOW>          Max live clusters, 0 = unbounded [default: 0]
      --threshold <THRESHOLD>    Jaccard similarity threshold (0.0–1.0) [default: 0.7]
      --stats                    Print token-savings statistics to stderr
      --onnx-model <PATH>        Stage 2 ONNX model path (requires llmlingua-onnx)
      --onnx-tokenizer <PATH>    Stage 2 tokenizer.json path
      --rpc-endpoint <URL>       Stage 2 LLMLingua daemon URL (requires llmlingua-rpc)
```

## Feature Flags

| Feature | Default | Adds |
|---------|---------|------|
| `default` | yes | Stage 0/1/3 — zero-model, sub-5ms rule-based mode |
| `tiktoken` | no | Accurate OpenAI token counting for savings stats |
| `llmlingua-onnx` | no | Stage 2 local neural pruning via ONNX Runtime |
| `llmlingua-rpc` | no | Stage 2 remote neural pruning via HTTP daemon |
| `token-counter` | no | BPE token counter subcommand (`ltk token-counter`) |

## Project Structure

```text
bin/
  ltk-cli/          ltk CLI binary (clap), with optional token-counter subcommand
crates/
  ltk-core/         core library: normalizer, clusterer, formatter, pipeline, onnx, rpc
features/
  checkout.feature   BDD checkout scenario (template demo)
  pipeline.feature   BDD ltk pipeline scenarios
```

## Testing

```bash
just test       # unit tests + property tests
just test-all   # everything
just lint       # clippy + typos + cargo shear
just format     # rustfmt + cargo sort
just ci         # lint + test-all + build
```

## License

Apache-2.0
