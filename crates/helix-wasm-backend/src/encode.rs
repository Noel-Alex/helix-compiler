//! Byte-level primitives for the WebAssembly binary format.
//!
//! The emitter hand-writes the module instead of depending on a wasm builder:
//! the shipped asset is the whole surface the browser sees, and the format is
//! small enough that explicit encoding is easier to audit than a transitive
//! dependency. Everything here is total — no path panics.

/// wasm `i32` value type.
pub(crate) const I32: u8 = 0x7F;
/// wasm `i64` value type.
pub(crate) const I64: u8 = 0x7E;
/// wasm `f32` value type.
pub(crate) const F32: u8 = 0x7D;
/// wasm `f64` value type.
pub(crate) const F64: u8 = 0x7C;

/// Section id of the type section.
pub(crate) const SECTION_TYPE: u8 = 1;
/// Section id of the import section.
pub(crate) const SECTION_IMPORT: u8 = 2;
/// Section id of the function section.
pub(crate) const SECTION_FUNCTION: u8 = 3;
/// Section id of the memory section.
pub(crate) const SECTION_MEMORY: u8 = 5;
/// Section id of the global section.
pub(crate) const SECTION_GLOBAL: u8 = 6;
/// Section id of the export section.
pub(crate) const SECTION_EXPORT: u8 = 7;
/// Section id of the code section.
pub(crate) const SECTION_CODE: u8 = 10;
/// Section id of the (diagnostic-only) name section.
pub(crate) const SECTION_NAME: u8 = 0;

/// Unsigned LEB128.
pub(crate) fn push_u32(out: &mut Vec<u8>, mut value: u32) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Signed LEB128 (the encoding of `i32.const`).
pub(crate) fn push_i32(out: &mut Vec<u8>, value: i32) {
    push_signed(out, i64::from(value));
}

/// Signed LEB128 (the encoding of `i64.const`).
pub(crate) fn push_i64(out: &mut Vec<u8>, value: i64) {
    push_signed(out, value);
}

fn push_signed(out: &mut Vec<u8>, mut value: i64) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        let done = (value == 0 && byte & 0x40 == 0) || (value == -1 && byte & 0x40 != 0);
        out.push(if done { byte } else { byte | 0x80 });
        if done {
            return;
        }
    }
}

/// A length-prefixed UTF-8 name.
pub(crate) fn push_name(out: &mut Vec<u8>, name: &str) {
    push_u32(out, name.len() as u32);
    out.extend_from_slice(name.as_bytes());
}

/// Frames one section: id, byte length, payload.
pub(crate) fn push_section(id: u8, body: Vec<u8>) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() + 8);
    out.push(id);
    push_u32(&mut out, body.len() as u32);
    out.extend_from_slice(&body);
    out
}

/// Frames a vector of entries with its element count.
pub(crate) fn vector_body(entries: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    push_u32(&mut out, entries.len() as u32);
    for entry in entries {
        out.extend_from_slice(entry);
    }
    out
}

/// Frames the code section: a vector of *size-prefixed* bodies, which is the
/// one section whose entries carry their own length.
pub(crate) fn code_body(bodies: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    push_u32(&mut out, bodies.len() as u32);
    for body in bodies {
        push_u32(&mut out, body.len() as u32);
        out.extend_from_slice(body);
    }
    out
}

/// Accumulates one function body: declared locals first, then code, then `end`.
///
/// Every opcode builder here takes and returns nothing, because the emitted
/// instruction sequences are fixed by the ABI — the helpers exist so the
/// sequences stay readable in one place.
#[derive(Debug)]
pub(crate) struct Body {
    params: u32,
    groups: Vec<(u32, u8)>,
    code: Vec<u8>,
}

impl Body {
    /// `params` is the function's wasm parameter count; declared locals are
    /// indexed above it, exactly as the binary format requires.
    pub(crate) fn new(params: u32) -> Self {
        Self {
            params,
            groups: Vec::new(),
            code: Vec::new(),
        }
    }

    /// Declares one local of `ty`, merging with the previous group when the
    /// type repeats. Returns the local's absolute index.
    pub(crate) fn declare_local(&mut self, ty: u8) -> u32 {
        let index = self.params + self.groups.iter().map(|(count, _)| count).sum::<u32>();
        match self.groups.last_mut() {
            Some((count, group_ty)) if *group_ty == ty => *count += 1,
            _ => self.groups.push((1, ty)),
        }
        index
    }

    pub(crate) fn code(&mut self) -> &mut Vec<u8> {
        &mut self.code
    }

    /// Serializes the entry: local declarations, code, then `end`.
    pub(crate) fn finish(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.code.len() + self.groups.len() * 3 + 1);
        push_u32(&mut out, self.groups.len() as u32);
        for (count, ty) in &self.groups {
            push_u32(&mut out, *count);
            out.push(*ty);
        }
        out.extend_from_slice(&self.code);
        out.push(0x0B); // end
        out
    }

    // -- instruction builders -------------------------------------------------

    pub(crate) fn raw(&mut self, bytes: &[u8]) {
        self.code.extend_from_slice(bytes);
    }

    pub(crate) fn local_get(&mut self, index: u32) {
        self.code.push(0x20);
        push_u32(&mut self.code, index);
    }

    pub(crate) fn local_set(&mut self, index: u32) {
        self.code.push(0x21);
        push_u32(&mut self.code, index);
    }

    pub(crate) fn local_tee(&mut self, index: u32) {
        self.code.push(0x22);
        push_u32(&mut self.code, index);
    }

    pub(crate) fn global_get(&mut self, index: u32) {
        self.code.push(0x23);
        push_u32(&mut self.code, index);
    }

    pub(crate) fn global_set(&mut self, index: u32) {
        self.code.push(0x24);
        push_u32(&mut self.code, index);
    }

    pub(crate) fn i32_const(&mut self, value: i32) {
        self.code.push(0x41);
        push_i32(&mut self.code, value);
    }

    pub(crate) fn i64_const(&mut self, value: i64) {
        self.code.push(0x42);
        push_i64(&mut self.code, value);
    }

    pub(crate) fn f32_const(&mut self, value: f32) {
        self.code.push(0x43);
        self.code.extend_from_slice(&value.to_bits().to_le_bytes());
    }

    pub(crate) fn f64_const(&mut self, value: f64) {
        self.code.push(0x44);
        self.code.extend_from_slice(&value.to_bits().to_le_bytes());
    }

    pub(crate) fn call(&mut self, index: u32) {
        self.code.push(0x10);
        push_u32(&mut self.code, index);
    }

    pub(crate) fn if_block(&mut self) {
        self.raw(&[0x04, 0x40]); // if, no result
    }

    pub(crate) fn loop_block(&mut self) {
        self.raw(&[0x03, 0x40]); // loop, no result
    }

    pub(crate) fn block(&mut self) {
        self.raw(&[0x02, 0x40]); // block, no result
    }

    pub(crate) fn end(&mut self) {
        self.code.push(0x0B);
    }

    pub(crate) fn branch(&mut self, depth: u32) {
        self.code.push(0x0C);
        push_u32(&mut self.code, depth);
    }

    pub(crate) fn branch_if(&mut self, depth: u32) {
        self.code.push(0x0D);
        push_u32(&mut self.code, depth);
    }

    pub(crate) fn wrap_i64(&mut self) {
        self.code.push(0xA7); // i32.wrap_i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsigned_leb128_matches_known_vectors() {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, 0);
        push_u32(&mut bytes, 624_485);
        push_u32(&mut bytes, u32::MAX);
        assert_eq!(
            bytes,
            vec![0x00, 0xE5, 0x8E, 0x26, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F]
        );
    }

    #[test]
    fn signed_leb128_matches_known_vectors() {
        let mut bytes = Vec::new();
        push_i32(&mut bytes, 0);
        push_i32(&mut bytes, -1);
        push_i32(&mut bytes, 63);
        push_i32(&mut bytes, -64);
        push_i32(&mut bytes, i32::MIN);
        push_i32(&mut bytes, i32::MAX);
        push_i64(&mut bytes, i64::MIN);
        assert_eq!(
            bytes,
            vec![
                0x00, 0x7F, 0x3F, 0x40, 0x80, 0x80, 0x80, 0x80, 0x78, 0xFF, 0xFF, 0xFF, 0xFF, 0x07,
                // i64::MIN: nine zero groups, then the sign bit.
                0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7F
            ]
        );
    }

    #[test]
    fn locals_merge_into_runs_above_the_parameters() {
        let mut body = Body::new(1);
        assert_eq!(body.declare_local(I32), 1);
        assert_eq!(body.declare_local(I32), 2);
        assert_eq!(body.declare_local(I64), 3);
        // group count, {2 × i32}, {1 × i64}, then `end` (no code of its own).
        assert_eq!(body.finish(), vec![0x02, 0x02, 0x7F, 0x01, 0x7E, 0x0B]);
    }

    #[test]
    fn section_prefixes_its_length() {
        assert_eq!(
            push_section(3, vec![0x01, 0x02]),
            vec![0x03, 0x02, 0x01, 0x02]
        );
    }

    #[test]
    fn vector_body_counts_its_entries() {
        assert_eq!(
            vector_body(&[vec![0x01], vec![0x02, 0x03]]),
            vec![0x02, 0x01, 0x02, 0x03]
        );
    }

    #[test]
    fn names_are_length_prefixed() {
        let mut bytes = Vec::new();
        push_name(&mut bytes, "main");
        assert_eq!(bytes, vec![4, b'm', b'a', b'i', b'n']);
    }

    #[test]
    fn body_serializes_declarations_then_code() {
        let mut body = Body::new(0);
        body.declare_local(I32);
        body.code().push(0x00);
        assert_eq!(body.finish(), vec![0x01, 0x01, 0x7F, 0x00, 0x0B]);
    }
}
