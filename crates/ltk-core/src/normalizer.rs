//! Stage 0: regex-based normalizer.
//!
//! [`RegexNormalizer`] strips ANSI escape sequences and masks volatile log
//! sub-tokens (IP, UUID, timestamp, hex, path, number, port) into stable
//! placeholders, then tokenizes the masked text for Jaccard similarity.
//!
//! ## Zero-copy
//!
//! When the input contains no ANSI escapes *and* no maskable sub-tokens, the
//! returned [`NormalizedLine::masked`] borrows the raw input directly
//! (`Cow::Borrowed`). Any masking or ANSI stripping promotes the result to
//! `Cow::Owned`.

use std::borrow::Cow;

use compact_str::CompactString;
use regex::Captures;
use strip_ansi_escapes::strip_str;

use crate::{NormalizedLine, Normalizer, Placeholder, Token};

/// Combined masking regex: one pass with named groups, ordered most-specific
/// first (timestamp before number, IP:port before bare IP, etc.).
///
/// `(?-u)` restricts `\w` / `\b` to ASCII for speed on log text.
const MASK_PATTERN: &str = concat!(
    r"(?-u)",
    // ISO-8601 date or full timestamp (date required, time optional).
    r"(?P<ts>\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?)",
    // RFC 4122 UUID.
    r"|(?P<uuid>\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b)",
    // IPv4 with trailing :port (matched before bare IPv4 so the port is tagged <PORT>).
    r"|(?P<ipv4port>\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}\b)",
    // Bare IPv4.
    r"|(?P<ipv4>\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b)",
    // Colon-prefixed port (`:8080`); colon is preserved in the replacement.
    r"|(?P<port>:\d{2,5}\b)",
    // `0x`-prefixed hex literal.
    r"|(?P<hex0x>\b0x[0-9a-fA-F]+\b)",
    // Long bare hex run (16+ chars: hash fragments, addresses without `0x`).
    r"|(?P<hexlong>\b[0-9a-fA-F]{16,}\b)",
    // Unix-style filesystem path.
    r"|(?P<path>(?:/[A-Za-z0-9_.+-]+)+/?)",
    // Decimal / float number (last so IP/timestamp/port win first).
    r"|(?P<num>\b\d+(?:\.\d+)?\b)",
);

/// Tokenizer regex: a placeholder tag, a word run, or a single non-word
/// non-space punctuation char (structure-preserving, à la `logslop`).
///
/// Unicode mode (the default) is required so `[^\w\s]` only matches valid
/// UTF-8; the masker alone uses `(?-u)` for ASCII speed since it has no
/// negated classes.
const TOKEN_PATTERN: &str = r"<(?:IP|UUID|TS|HEX|NUM|PORT|PATH)>|\w+|[^\w\s]";

/// A [`Normalizer`] backed by compiled regexes for ANSI stripping, volatile
/// sub-token masking, and structure-preserving tokenization.
///
/// Construct once with [`RegexNormalizer::new`] and reuse across lines; the
/// regexes are compiled a single time.
#[derive(Debug)]
pub struct RegexNormalizer {
    mask_re: regex::Regex,
    token_re: regex::Regex,
}

impl RegexNormalizer {
    /// Compile the normalizer's regexes.
    ///
    /// # Errors
    /// Returns [`crate::LtkError::Regex`] if a built-in pattern fails to
    /// compile (only possible on a bug in the constant patterns).
    pub fn new() -> Result<Self, crate::LtkError> {
        let mask_re =
            regex::Regex::new(MASK_PATTERN).map_err(|e| crate::LtkError::Regex(e.to_string()))?;
        let token_re =
            regex::Regex::new(TOKEN_PATTERN).map_err(|e| crate::LtkError::Regex(e.to_string()))?;
        Ok(Self { mask_re, token_re })
    }

    /// Tokenize `masked` into placeholder tags, word runs, and single
    /// punctuation characters.
    fn tokenize(&self, masked: &str) -> Vec<Token> {
        let mut out = Vec::new();
        for m in self.token_re.find_iter(masked) {
            let s = m.as_str();
            let tok = match s {
                "<IP>" => Token::Mask(Placeholder::Ip),
                "<UUID>" => Token::Mask(Placeholder::Uuid),
                "<TS>" => Token::Mask(Placeholder::Timestamp),
                "<HEX>" => Token::Mask(Placeholder::Hex),
                "<NUM>" => Token::Mask(Placeholder::Number),
                "<PORT>" => Token::Mask(Placeholder::Port),
                "<PATH>" => Token::Mask(Placeholder::Path),
                _ => Token::Lit(CompactString::new(s)),
            };
            out.push(tok);
        }
        out
    }
}

impl Normalizer for RegexNormalizer {
    fn normalize<'a>(&self, raw: &'a str) -> NormalizedLine<'a> {
        // 1. Strip ANSI escapes only when an ESC byte is present, so plain lines stay zero-copy.
        //    `strip_str` returns an owned `String`.
        let ansi: Cow<'a, str> =
            if raw.contains('\x1b') { Cow::Owned(strip_str(raw)) } else { Cow::Borrowed(raw) };
        let ansi_changed = matches!(ansi, Cow::Owned(_));

        // 2. Mask volatile sub-tokens in a single regex pass. The replacer returns
        //    `Cow::Borrowed(&'static str)` placeholders, so no per-match allocation. `replace_all`
        //    yields `Cow::Borrowed` iff nothing matched.
        let masked: Cow<'_, str> = {
            let s: &str = match &ansi {
                Cow::Borrowed(b) => b,
                Cow::Owned(o) => o.as_str(),
            };
            self.mask_re.replace_all(s, |caps: &Captures<'_>| -> Cow<'static, str> {
                if caps.name("ts").is_some() {
                    return Cow::Borrowed(Placeholder::Timestamp.as_str());
                }
                if caps.name("uuid").is_some() {
                    return Cow::Borrowed(Placeholder::Uuid.as_str());
                }
                if caps.name("ipv4port").is_some() {
                    return Cow::Borrowed("<IP>:<PORT>");
                }
                if caps.name("ipv4").is_some() {
                    return Cow::Borrowed(Placeholder::Ip.as_str());
                }
                if caps.name("port").is_some() {
                    return Cow::Borrowed(":<PORT>");
                }
                if caps.name("hex0x").is_some() || caps.name("hexlong").is_some() {
                    return Cow::Borrowed(Placeholder::Hex.as_str());
                }
                if caps.name("path").is_some() {
                    return Cow::Borrowed(Placeholder::Path.as_str());
                }
                if caps.name("num").is_some() {
                    return Cow::Borrowed(Placeholder::Number.as_str());
                }
                Cow::Borrowed("")
            })
        };
        let mask_changed = matches!(masked, Cow::Owned(_));

        // 3. Finalize: stay zero-copy only when neither step changed anything.
        let final_masked: Cow<'a, str> = if !ansi_changed && !mask_changed {
            Cow::Borrowed(raw)
        } else {
            Cow::Owned(masked.into_owned())
        };

        // 4. Tokenize (tokens own their content via CompactString, so they don't borrow from
        //    `final_masked`).
        let tokens = self.tokenize(&final_masked);
        NormalizedLine::new(raw, final_masked, tokens)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

    fn n() -> Result<RegexNormalizer, Box<dyn Error>> {
        Ok(RegexNormalizer::new()?)
    }

    fn masked(norm: &RegexNormalizer, raw: &str) -> String {
        norm.normalize(raw).masked.into_owned()
    }

    #[test]
    fn zero_copy_when_nothing_maskable() -> Result<(), Box<dyn Error>> {
        let norm = n()?;
        let line = norm.normalize("plain text with nothing volatile");
        assert!(matches!(line.masked, Cow::Borrowed(_)));
        Ok(())
    }

    #[test]
    fn ansi_stripped() -> Result<(), Box<dyn Error>> {
        let norm = n()?;
        assert_eq!(masked(&norm, "\x1b[31mred text\x1b[0m"), "red text");
        Ok(())
    }

    #[test]
    fn ipv4_masked() -> Result<(), Box<dyn Error>> {
        let norm = n()?;
        assert_eq!(masked(&norm, "connect 10.0.0.1 ok"), "connect <IP> ok");
        Ok(())
    }

    #[test]
    fn ipv4_port_masked_together() -> Result<(), Box<dyn Error>> {
        let norm = n()?;
        assert_eq!(masked(&norm, "dial 192.168.1.10:5432 refused"), "dial <IP>:<PORT> refused");
        Ok(())
    }

    #[test]
    fn bare_port_after_colon_masked() -> Result<(), Box<dyn Error>> {
        let norm = n()?;
        assert_eq!(masked(&norm, "listen :8080"), "listen :<PORT>");
        Ok(())
    }

    #[test]
    fn uuid_masked() -> Result<(), Box<dyn Error>> {
        let norm = n()?;
        assert_eq!(
            masked(&norm, "req 550e8400-e29b-41d4-a716-446655440000 done"),
            "req <UUID> done"
        );
        Ok(())
    }

    #[test]
    fn iso_timestamp_masked() -> Result<(), Box<dyn Error>> {
        let norm = n()?;
        assert_eq!(masked(&norm, "2026-07-31T10:00:01.001Z start"), "<TS> start");
        assert_eq!(masked(&norm, "2026-07-31 10:00:01 start"), "<TS> start");
        assert_eq!(masked(&norm, "2026-07-31 start"), "<TS> start");
        Ok(())
    }

    #[test]
    fn hex_masked() -> Result<(), Box<dyn Error>> {
        let norm = n()?;
        assert_eq!(masked(&norm, "addr 0xdeadbeef"), "addr <HEX>");
        assert_eq!(masked(&norm, "hash a1b2c3d4e5f60718293a4b5c6d7e8f9a"), "hash <HEX>");
        Ok(())
    }

    #[test]
    fn path_masked() -> Result<(), Box<dyn Error>> {
        let norm = n()?;
        assert_eq!(masked(&norm, "open /var/log/syslog"), "open <PATH>");
        Ok(())
    }

    #[test]
    fn number_masked() -> Result<(), Box<dyn Error>> {
        let norm = n()?;
        assert_eq!(masked(&norm, "retry 3 of 14 pct 99.5"), "retry <NUM> of <NUM> pct <NUM>");
        Ok(())
    }

    #[test]
    fn example_line_normalizes_to_template() -> Result<(), Box<dyn Error>> {
        let norm = n()?;
        assert_eq!(
            masked(
                &norm,
                "2026-07-31T10:00:01.001Z [ERROR] [auth] Failed to connect to DB at 192.168.1.10:5432: Connection refused"
            ),
            "<TS> [ERROR] [auth] Failed to connect to DB at <IP>:<PORT>: Connection refused"
        );
        Ok(())
    }

    #[test]
    fn tokenization_emits_placeholders_and_punct() -> Result<(), Box<dyn Error>> {
        let norm = n()?;
        let line = norm.normalize("2026-07-31 [ERROR] retry 3");
        let toks: Vec<&str> = line.tokens.iter().map(Token::render).collect();
        assert_eq!(toks, vec!["<TS>", "[", "ERROR", "]", "retry", "<NUM>"]);
        Ok(())
    }

    #[test]
    fn two_lines_differing_only_in_ip_mask_identically() -> Result<(), Box<dyn Error>> {
        let norm = n()?;
        let a = masked(&norm, "Failed to connect to 192.168.1.10:5432 refused");
        let b = masked(&norm, "Failed to connect to 192.168.1.11:5432 refused");
        assert_eq!(a, b);
        Ok(())
    }
}
