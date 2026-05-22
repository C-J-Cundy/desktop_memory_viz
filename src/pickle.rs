//! Streaming pickle parser specialized for PyTorch memory snapshots.
//!
//! Supports the subset of pickle protocol 4 opcodes observed in
//! torch.cuda.memory._dump_snapshot output. Recognizes event and annotation
//! dicts at SETITEMS time, extracts their fields into a `Snapshot`, and
//! discards the dict immediately (stack slot + memo slot) so the heap is
//! freed instead of accumulating ~31M dicts in memory.

use anyhow::{bail, Context, Result};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::rc::Rc;

// ── Opcode constants (pickle protocol 4 subset) ─────────────────────────

const PROTO: u8 = 0x80;
const FRAME: u8 = 0x95;
const STOP: u8 = b'.';

const EMPTY_DICT: u8 = b'}';
const EMPTY_LIST: u8 = b']';
const EMPTY_TUPLE: u8 = b')';
const MARK: u8 = b'(';

const SETITEM: u8 = b's';
const SETITEMS: u8 = b'u';
const APPEND: u8 = b'a';
const APPENDS: u8 = b'e';

const TUPLE: u8 = b't';
const TUPLE1: u8 = 0x85;
const TUPLE2: u8 = 0x86;
const TUPLE3: u8 = 0x87;

const MEMOIZE: u8 = 0x94;
const BINPUT: u8 = b'q';
const LONG_BINPUT: u8 = b'r';
const BINGET: u8 = b'h';
const LONG_BINGET: u8 = b'j';

const SHORT_BINUNICODE: u8 = 0x8c;
const BINUNICODE: u8 = b'X';
const BINUNICODE8: u8 = 0x8d;

const SHORT_BINBYTES: u8 = b'C';
const BINBYTES: u8 = b'B';
const BINBYTES8: u8 = 0x8e;

const BININT: u8 = b'J';
const BININT1: u8 = b'K';
const BININT2: u8 = b'M';
const LONG1: u8 = 0x8a;
const LONG4: u8 = 0x8b;

const BINFLOAT: u8 = b'G';

const NONE_OP: u8 = b'N';
const NEWTRUE: u8 = 0x88;
const NEWFALSE: u8 = 0x89;

// ── Value ───────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct DictInner {
    pub pairs: Vec<(Value, Value)>,
    /// Memo position assigned to this dict by MEMOIZE, if any.
    /// Used by the event/annotation specialization to release the memo
    /// slot when the dict has been consumed.
    pub memo_id: Option<u32>,
}

#[derive(Clone, Debug)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Rc<str>),
    Bytes(Rc<[u8]>),
    Tuple(Rc<[Value]>),
    List(Rc<RefCell<Vec<Value>>>),
    Dict(Rc<RefCell<DictInner>>),
    /// Stack-only sentinel for MARK opcode.
    Mark,
    /// Placeholder left behind after an event/annotation dict has been
    /// extracted and dropped. If BINGET ever resolves to this, the
    /// "discard events" optimization is unsafe for this snapshot and
    /// parsing bails with a clear error.
    Consumed,
}

// ── Snapshot output ─────────────────────────────────────────────────────

#[derive(Debug)]
pub struct Annotation {
    pub stage: String,
    pub name: String,
    pub time_us: i64,
}

#[derive(Debug, Default)]
pub struct Snapshot {
    /// (action_code, addr, size, time_us, frame_idx)
    pub events: Vec<(u8, u64, u64, i64, u32)>,
    pub frame_strings: Vec<String>,
    pub annotations: Vec<Annotation>,
    /// (address, size) for blocks live at snapshot time whose address was
    /// never seen as an alloc in `events`. These represent allocations that
    /// existed before the recording window started and persisted through it.
    pub live_blocks: Vec<(u64, u64)>,
}

fn action_code(s: &str) -> Option<u8> {
    match s {
        "alloc" => Some(0),
        "free_requested" => Some(1),
        "free_completed" => Some(2),
        "segment_alloc" => Some(3),
        "segment_free" => Some(4),
        "segment_map" => Some(5),
        "segment_unmap" => Some(6),
        _ => None,
    }
}

// ── Parser ──────────────────────────────────────────────────────────────

pub struct Parser<R: Read> {
    rdr: R,
    stack: Vec<Value>,
    memo: Vec<Value>,
    out: Snapshot,
    /// Maps frame-list Rc identity (by pointer) to interned frame_strings index,
    /// so events sharing the same frames-list Rc skip recomputing frame_summary.
    frame_cache: HashMap<*const RefCell<Vec<Value>>, u32>,
    /// Frame-summary string → interned index in out.frame_strings.
    frame_intern: HashMap<String, u32>,
    /// Addresses that appeared as `alloc` events. Used at STOP to identify
    /// segment blocks whose addr was never alloc'd during the window (i.e.
    /// allocations that pre-dated the recording).
    alloced_addrs: HashSet<u64>,
}

impl<R: Read> Parser<R> {
    pub fn new(rdr: R) -> Self {
        Self {
            rdr,
            stack: Vec::with_capacity(64),
            memo: Vec::with_capacity(1 << 20),
            out: Snapshot::default(),
            frame_cache: HashMap::new(),
            frame_intern: HashMap::new(),
            alloced_addrs: HashSet::new(),
        }
    }

    // ── Reader helpers ───────────────────────────────────────────────

    fn read_u8(&mut self) -> Result<u8> {
        let mut b = [0u8; 1];
        self.rdr.read_exact(&mut b).context("read_u8")?;
        Ok(b[0])
    }
    fn read_u16_le(&mut self) -> Result<u16> {
        let mut b = [0u8; 2];
        self.rdr.read_exact(&mut b).context("read_u16_le")?;
        Ok(u16::from_le_bytes(b))
    }
    fn read_i32_le(&mut self) -> Result<i32> {
        let mut b = [0u8; 4];
        self.rdr.read_exact(&mut b).context("read_i32_le")?;
        Ok(i32::from_le_bytes(b))
    }
    fn read_u32_le(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.rdr.read_exact(&mut b).context("read_u32_le")?;
        Ok(u32::from_le_bytes(b))
    }
    fn read_u64_le(&mut self) -> Result<u64> {
        let mut b = [0u8; 8];
        self.rdr.read_exact(&mut b).context("read_u64_le")?;
        Ok(u64::from_le_bytes(b))
    }
    fn read_n(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; n];
        self.rdr.read_exact(&mut buf).context("read_n")?;
        Ok(buf)
    }
    fn read_str(&mut self, n: usize) -> Result<Rc<str>> {
        let buf = self.read_n(n)?;
        let s = std::str::from_utf8(&buf).context("invalid utf-8 in pickle string")?;
        Ok(Rc::from(s))
    }

    fn pop(&mut self) -> Result<Value> {
        self.stack.pop().context("pickle stack underflow")
    }
    fn find_mark(&self) -> Result<usize> {
        for (i, v) in self.stack.iter().enumerate().rev() {
            if matches!(v, Value::Mark) {
                return Ok(i);
            }
        }
        bail!("no MARK on stack")
    }
    fn drain_to_mark(&mut self) -> Result<Vec<Value>> {
        let idx = self.find_mark()?;
        let items: Vec<Value> = self.stack.drain(idx + 1..).collect();
        self.stack.pop(); // remove MARK
        Ok(items)
    }

    // ── Memo ─────────────────────────────────────────────────────────

    fn memo_put(&mut self, idx: usize) -> Result<()> {
        // Tag the dict with its memo id so SETITEMS can clear the slot
        // after extracting an event/annotation dict.
        if let Some(Value::Dict(d)) = self.stack.last() {
            let mut dref = d.borrow_mut();
            if dref.memo_id.is_none() {
                dref.memo_id = Some(idx as u32);
            }
        }
        let v = self
            .stack
            .last()
            .context("memo put on empty stack")?
            .clone();
        if idx >= self.memo.len() {
            self.memo.resize(idx + 1, Value::None);
        }
        self.memo[idx] = v;
        Ok(())
    }

    fn memo_get(&mut self, idx: usize) -> Result<()> {
        let v = self
            .memo
            .get(idx)
            .with_context(|| format!("memo get out of range: {}", idx))?
            .clone();
        if matches!(v, Value::Consumed) {
            bail!(
                "memo slot {} resolves to a consumed event/annotation dict; \
                 this snapshot uses BINGET on event-shell dicts and the \
                 'discard events on SETITEMS' optimization is unsafe for it",
                idx
            );
        }
        self.stack.push(v);
        Ok(())
    }

    // ── Event / annotation specialization ─────────────────────────────

    /// If `pairs` looks like an event or annotation dict, extract the fields
    /// into self.out and return true. Otherwise return false (caller leaves
    /// the dict on the stack as a normal Value::Dict).
    fn try_emit_specialized(&mut self, pairs: &[(Value, Value)]) -> bool {
        // First scan: find action and stage values.
        let mut action: Option<Rc<str>> = None;
        let mut stage: Option<Rc<str>> = None;
        for (k, v) in pairs {
            let Value::Str(k) = k else { continue };
            match &**k {
                "action" => {
                    if let Value::Str(s) = v {
                        action = Some(s.clone());
                    }
                }
                "stage" => {
                    if let Value::Str(s) = v {
                        stage = Some(s.clone());
                    }
                }
                _ => {}
            }
        }

        if let Some(act) = action {
            // Event dict.
            let Some(code) = action_code(&act) else {
                // Unknown action: match Python (skip event) but still consume
                // the dict so we can free it.
                return true;
            };
            let mut addr: u64 = 0;
            let mut size: u64 = 0;
            let mut time_us: i64 = 0;
            let mut frames_ref: Option<Rc<RefCell<Vec<Value>>>> = None;
            for (k, v) in pairs {
                let Value::Str(k) = k else { continue };
                match &**k {
                    "addr" => {
                        if let Value::Int(i) = v {
                            addr = *i as u64;
                        }
                    }
                    "size" => {
                        if let Value::Int(i) = v {
                            size = *i as u64;
                        }
                    }
                    "time_us" => {
                        if let Value::Int(i) = v {
                            time_us = *i;
                        }
                    }
                    "frames" => {
                        if let Value::List(l) = v {
                            frames_ref = Some(l.clone());
                        }
                    }
                    _ => {}
                }
            }
            let frame_idx = self.intern_frames(frames_ref);
            self.out.events.push((code, addr, size, time_us, frame_idx));
            // Track addrs that appeared as `alloc` (code 0) for later
            // segments-baseline matching.
            if code == 0 {
                self.alloced_addrs.insert(addr);
            }
            return true;
        }

        if let Some(stage) = stage {
            // Annotation dict: require a "time_us" key to disambiguate.
            let mut name = String::new();
            let mut time_us: i64 = 0;
            let mut has_time = false;
            for (k, v) in pairs {
                let Value::Str(k) = k else { continue };
                match &**k {
                    "name" => {
                        if let Value::Str(s) = v {
                            name = s.to_string();
                        }
                    }
                    "time_us" => {
                        if let Value::Int(i) = v {
                            time_us = *i;
                            has_time = true;
                        }
                    }
                    _ => {}
                }
            }
            if has_time {
                self.out.annotations.push(Annotation {
                    stage: stage.to_string(),
                    name,
                    time_us,
                });
                return true;
            }
        }
        false
    }

    /// Intern a frame_summary string, using the Rc-identity cache to skip the
    /// O(frames) walk when this exact frames Rc has been seen before.
    fn intern_frames(&mut self, frames: Option<Rc<RefCell<Vec<Value>>>>) -> u32 {
        let Some(frames_rc) = frames else {
            return self.intern_str(String::new());
        };
        let key = Rc::as_ptr(&frames_rc);
        if let Some(&idx) = self.frame_cache.get(&key) {
            return idx;
        }
        let summary = frame_summary(&frames_rc.borrow());
        let idx = self.intern_str(summary);
        self.frame_cache.insert(key, idx);
        idx
    }

    fn intern_str(&mut self, s: String) -> u32 {
        if let Some(&i) = self.frame_intern.get(&s) {
            return i;
        }
        let i = self.out.frame_strings.len() as u32;
        self.frame_intern.insert(s.clone(), i);
        self.out.frame_strings.push(s);
        i
    }

    /// Walk the top-level dict's `segments[*].blocks` and record blocks whose
    /// address never appeared as an `alloc` event during the trace. These are
    /// allocations that pre-dated the recording window and persisted through
    /// it; they're not visible in the event stream alone.
    fn collect_live_blocks(&mut self, root: &Value) {
        let Value::Dict(top) = root else { return };
        let top = top.borrow();
        let mut segments_val: Option<Value> = None;
        for (k, v) in &top.pairs {
            if let Value::Str(k) = k {
                if &**k == "segments" {
                    segments_val = Some(v.clone());
                    break;
                }
            }
        }
        let Some(Value::List(segments)) = segments_val else {
            return;
        };
        for seg in segments.borrow().iter() {
            let Value::Dict(seg) = seg else { continue };
            let seg = seg.borrow();
            // Find blocks
            let mut blocks_val: Option<Value> = None;
            for (k, v) in &seg.pairs {
                if let Value::Str(k) = k {
                    if &**k == "blocks" {
                        blocks_val = Some(v.clone());
                        break;
                    }
                }
            }
            let Some(Value::List(blocks)) = blocks_val else {
                continue;
            };
            for blk in blocks.borrow().iter() {
                let Value::Dict(blk) = blk else { continue };
                let blk = blk.borrow();
                let mut addr: u64 = 0;
                let mut size: u64 = 0;
                let mut state_active = false;
                for (k, v) in &blk.pairs {
                    let Value::Str(k) = k else { continue };
                    match &**k {
                        "address" => {
                            if let Value::Int(i) = v {
                                addr = *i as u64;
                            }
                        }
                        "size" => {
                            if let Value::Int(i) = v {
                                size = *i as u64;
                            }
                        }
                        "state" => {
                            if let Value::Str(s) = v {
                                if &**s == "active_allocated" {
                                    state_active = true;
                                }
                            }
                        }
                        _ => {}
                    }
                }
                if state_active && !self.alloced_addrs.contains(&addr) {
                    self.out.live_blocks.push((addr, size));
                }
            }
        }
    }

    // ── Main parse loop ──────────────────────────────────────────────

    pub fn parse(mut self) -> Result<Snapshot> {
        loop {
            let op = self.read_u8()?;
            match op {
                PROTO => {
                    let _v = self.read_u8()?;
                }
                FRAME => {
                    let _len = self.read_u64_le()?;
                }
                STOP => {
                    let root = self.pop()?;
                    self.collect_live_blocks(&root);
                    return Ok(self.out);
                }

                EMPTY_DICT => self.stack.push(Value::Dict(Rc::new(RefCell::new(
                    DictInner {
                        pairs: Vec::new(),
                        memo_id: None,
                    },
                )))),
                EMPTY_LIST => self
                    .stack
                    .push(Value::List(Rc::new(RefCell::new(Vec::new())))),
                EMPTY_TUPLE => self.stack.push(Value::Tuple(Rc::new([]))),
                MARK => self.stack.push(Value::Mark),

                SETITEM => {
                    let v = self.pop()?;
                    let k = self.pop()?;
                    match self.stack.last() {
                        Some(Value::Dict(d)) => d.borrow_mut().pairs.push((k, v)),
                        _ => bail!("SETITEM without dict on stack"),
                    }
                }
                SETITEMS => self.handle_setitems()?,
                APPEND => {
                    let v = self.pop()?;
                    match self.stack.last() {
                        Some(Value::List(l)) => l.borrow_mut().push(v),
                        _ => bail!("APPEND without list on stack"),
                    }
                }
                APPENDS => {
                    let items = self.drain_to_mark()?;
                    let l = match self.stack.last() {
                        Some(Value::List(l)) => l.clone(),
                        _ => bail!("APPENDS without list on stack"),
                    };
                    let mut lref = l.borrow_mut();
                    lref.reserve(items.len());
                    lref.extend(items);
                }

                TUPLE => {
                    let items = self.drain_to_mark()?;
                    self.stack.push(Value::Tuple(items.into()));
                }
                TUPLE1 => {
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(Rc::from([a])));
                }
                TUPLE2 => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(Rc::from([a, b])));
                }
                TUPLE3 => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(Rc::from([a, b, c])));
                }

                MEMOIZE => {
                    let idx = self.memo.len();
                    self.memo_put(idx)?;
                }
                BINPUT => {
                    let id = self.read_u8()? as usize;
                    self.memo_put(id)?;
                }
                LONG_BINPUT => {
                    let id = self.read_u32_le()? as usize;
                    self.memo_put(id)?;
                }
                BINGET => {
                    let id = self.read_u8()? as usize;
                    self.memo_get(id)?;
                }
                LONG_BINGET => {
                    let id = self.read_u32_le()? as usize;
                    self.memo_get(id)?;
                }

                SHORT_BINUNICODE => {
                    let n = self.read_u8()? as usize;
                    let s = self.read_str(n)?;
                    self.stack.push(Value::Str(s));
                }
                BINUNICODE => {
                    let n = self.read_u32_le()? as usize;
                    let s = self.read_str(n)?;
                    self.stack.push(Value::Str(s));
                }
                BINUNICODE8 => {
                    let n = self.read_u64_le()? as usize;
                    let s = self.read_str(n)?;
                    self.stack.push(Value::Str(s));
                }

                SHORT_BINBYTES => {
                    let n = self.read_u8()? as usize;
                    let b = self.read_n(n)?;
                    self.stack.push(Value::Bytes(b.into()));
                }
                BINBYTES => {
                    let n = self.read_u32_le()? as usize;
                    let b = self.read_n(n)?;
                    self.stack.push(Value::Bytes(b.into()));
                }
                BINBYTES8 => {
                    let n = self.read_u64_le()? as usize;
                    let b = self.read_n(n)?;
                    self.stack.push(Value::Bytes(b.into()));
                }

                BININT => {
                    let v = self.read_i32_le()?;
                    self.stack.push(Value::Int(v as i64));
                }
                BININT1 => {
                    let v = self.read_u8()? as i64;
                    self.stack.push(Value::Int(v));
                }
                BININT2 => {
                    let v = self.read_u16_le()? as i64;
                    self.stack.push(Value::Int(v));
                }
                LONG1 => {
                    let n = self.read_u8()? as usize;
                    let bytes = self.read_n(n)?;
                    self.stack.push(Value::Int(parse_long_le(&bytes)?));
                }
                LONG4 => {
                    let n = self.read_u32_le()? as usize;
                    let bytes = self.read_n(n)?;
                    self.stack.push(Value::Int(parse_long_le(&bytes)?));
                }

                BINFLOAT => {
                    let mut b = [0u8; 8];
                    self.rdr.read_exact(&mut b).context("BINFLOAT")?;
                    self.stack.push(Value::Float(f64::from_be_bytes(b)));
                }

                NONE_OP => self.stack.push(Value::None),
                NEWTRUE => self.stack.push(Value::Bool(true)),
                NEWFALSE => self.stack.push(Value::Bool(false)),

                _ => bail!("unsupported pickle opcode: 0x{:02x}", op),
            }
        }
    }

    fn handle_setitems(&mut self) -> Result<()> {
        let items = self.drain_to_mark()?;
        let n_pairs = items.len() / 2;
        let d = match self.stack.last() {
            Some(Value::Dict(d)) => d.clone(),
            _ => bail!("SETITEMS without dict on stack"),
        };
        {
            let mut dref = d.borrow_mut();
            dref.pairs.reserve(n_pairs);
            let mut it = items.into_iter();
            while let Some(k) = it.next() {
                let v = it.next().context("SETITEMS odd item count")?;
                dref.pairs.push((k, v));
            }
        }

        // Try to recognize this dict as an event or annotation, extract its
        // fields, and drop the dict to free its heap.
        let (specialized, memo_id) = {
            let dref = d.borrow();
            // Borrow the pairs immutably to classify, then release before touching memo.
            let memo_id = dref.memo_id;
            // We need to call self.try_emit_specialized, which mutates self.
            // To avoid simultaneous borrow, drop dref first and re-borrow inside.
            drop(dref);
            let dref = d.borrow();
            // Build a temporary view; try_emit_specialized takes &[..].
            // (Calling through self because it touches self.out / self.frame_cache.)
            // Safe because self.stack still holds an Rc to the dict.
            let specialized = self.try_emit_specialized(&dref.pairs);
            (specialized, memo_id)
        };

        if specialized {
            // Replace stack top with Consumed; drop local Rc; clear memo slot.
            drop(d);
            self.stack.pop();
            self.stack.push(Value::Consumed);
            if let Some(mid) = memo_id {
                if (mid as usize) < self.memo.len() {
                    self.memo[mid as usize] = Value::Consumed;
                }
            }
        }
        Ok(())
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Parse a little-endian, two's-complement variable-length integer.
fn parse_long_le(bytes: &[u8]) -> Result<i64> {
    if bytes.is_empty() {
        return Ok(0);
    }
    if bytes.len() > 8 {
        bail!(
            "pickle LONG too large for i64 ({} bytes); add BigInt support",
            bytes.len()
        );
    }
    let mut v: i64 = 0;
    for (i, &b) in bytes.iter().enumerate() {
        v |= (b as i64) << (8 * i);
    }
    let msb = *bytes.last().unwrap();
    if msb & 0x80 != 0 {
        let used_bits = 8 * bytes.len();
        if used_bits < 64 {
            v |= -1i64 << used_bits;
        }
    }
    Ok(v)
}

fn as_str(v: &Value) -> Option<&str> {
    if let Value::Str(s) = v {
        Some(s)
    } else {
        None
    }
}
fn as_int(v: &Value) -> Option<i64> {
    if let Value::Int(i) = v {
        Some(*i)
    } else {
        None
    }
}
fn frame_dict_lookup<'a>(d: &'a [(Value, Value)], key: &str) -> Option<&'a Value> {
    for (k, v) in d {
        if let Value::Str(s) = k {
            if &**s == key {
                return Some(v);
            }
        }
    }
    None
}

/// Build the frame-summary string identical to extract_snapshot.py's `frame_summary()`.
fn frame_summary(frames: &[Value]) -> String {
    if frames.is_empty() {
        return String::new();
    }
    let mut parts: Vec<String> = Vec::with_capacity(frames.len());
    for f in frames {
        let Value::Dict(d) = f else { continue };
        let d = d.borrow();
        let filename = frame_dict_lookup(&d.pairs, "filename")
            .and_then(as_str)
            .unwrap_or("");
        let name = frame_dict_lookup(&d.pairs, "name")
            .and_then(as_str)
            .unwrap_or("");
        let line = frame_dict_lookup(&d.pairs, "line")
            .and_then(as_int)
            .unwrap_or(0);
        let short = filename.rsplit('/').next().unwrap_or(filename);
        parts.push(format!("{}:{} ({})", short, line, name));
    }
    parts.join(" <- ")
}

/// Convenience: read pickle from `rdr` and produce a Snapshot.
pub fn parse_snapshot<R: Read>(rdr: R) -> Result<Snapshot> {
    Parser::new(rdr).parse()
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn tiny_pickle() -> Vec<u8> {
        let mut b = vec![PROTO, 4];
        b.push(EMPTY_DICT);
        b.push(MARK);
        b.push(SHORT_BINUNICODE);
        b.push(1);
        b.extend_from_slice(b"a");
        b.push(BININT1);
        b.push(1);
        b.push(SHORT_BINUNICODE);
        b.push(1);
        b.extend_from_slice(b"b");
        b.push(EMPTY_LIST);
        b.push(MARK);
        b.push(BININT1);
        b.push(10);
        b.push(BININT1);
        b.push(20);
        b.push(APPENDS);
        b.push(SETITEMS);
        b.push(STOP);
        b
    }

    #[test]
    fn parses_tiny_dict_without_specialization() {
        // No "action" or "stage" key, so the dict is not extracted.
        let bytes = tiny_pickle();
        let snap = Parser::new(Cursor::new(bytes)).parse().unwrap();
        // No events / annotations were emitted because the dict isn't an event/annotation.
        assert!(snap.events.is_empty());
        assert!(snap.annotations.is_empty());
    }

    #[test]
    fn parse_long_le_basic() {
        assert_eq!(parse_long_le(&[]).unwrap(), 0);
        assert_eq!(parse_long_le(&[0x01]).unwrap(), 1);
        assert_eq!(parse_long_le(&[0xff]).unwrap(), -1);
        assert_eq!(parse_long_le(&[0x80]).unwrap(), -128);
        assert_eq!(parse_long_le(&[0x00, 0x01]).unwrap(), 256);
    }
}
