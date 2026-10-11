//! Standard base64 encoding (RFC 4648, padded).
//!
//! The `emit` op of [`crate::helix_apply`] hands a whole WebAssembly module to
//! the browser inside JSON, and this workspace carries no base64 crate, so the
//! encoder ships here: one table, one loop, no dependency.
//!
//! Standard alphabet rather than URL-safe: the payload travels inside JSON
//! strings, where `+` and `/` need no escaping.

/// Alphabet of the standard encoding.
const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encodes `data` with `=` padding to a multiple of four characters.
#[must_use]
pub fn encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        // The final chunk is zero-padded; its tail characters become `=`.
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = u32::from_be_bytes([0, b[0], b[1], b[2]]);
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Inverse of [`encode`], test-only: the shipped surface never decodes, but
/// the apply tests need to check the emitted module's header.
#[cfg(test)]
pub(crate) fn decode(text: &str) -> Vec<u8> {
    let mut out = Vec::new();
    let mut acc = 0u32;
    let mut bits = 0u32;
    for c in text.bytes() {
        let value = match c {
            b'A'..=b'Z' => u32::from(c - b'A'),
            b'a'..=b'z' => u32::from(c - b'a') + 26,
            b'0'..=b'9' => u32::from(c - b'0') + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            other => panic!("not a standard base64 character: {other:#x}"),
        };
        acc = (acc << 6) | value;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The RFC 4648 §10 vectors: every padding length plus a long alphabet run.
    #[test]
    fn encodes_the_rfc4648_vectors() {
        for (input, want) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
            (
                "abcdefghijklmnopqrstuvwxyz",
                "YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXo=",
            ),
            ("0123456789", "MDEyMzQ1Njc4OQ=="),
        ] {
            assert_eq!(encode(input.as_bytes()), want, "vector for {input:?}");
        }
    }

    /// Binary input must round-trip; the tests decode so the encoder is
    /// checked against its inverse rather than against itself.
    #[test]
    fn encodes_binary_data_without_loss() {
        let data: Vec<u8> = (0..=255u8).collect();
        let text = encode(&data);
        assert_eq!(text.len(), 4 * data.len().div_ceil(3));
        assert_eq!(decode(&text), data);
        // Every alignment of a long payload keeps its tail padding.
        for len in 0..16 {
            let data: Vec<u8> = (0..len).map(|i| i as u8 * 7 + 1).collect();
            assert_eq!(decode(&encode(&data)), data, "round-trip at len {len}");
        }
    }
}
