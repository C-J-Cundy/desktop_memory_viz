//! Standalone CLI that parses a PyTorch memory snapshot .pickle (or .pickle.gz)
//! and writes the same JSON format produced by extract_snapshot.py.
//!
//! Usage:
//!     extract_pickle <input.pickle> <output.json>

use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

// Share the pickle module with the main binary by path.
#[path = "../pickle.rs"]
mod pickle;

#[cfg(feature = "heap-profile")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn open_input(path: &PathBuf) -> Result<BufReader<File>> {
    let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    Ok(BufReader::with_capacity(1 << 20, f))
}

fn main() -> Result<()> {
    #[cfg(feature = "heap-profile")]
    let _profiler = dhat::Profiler::new_heap();

    let mut args = std::env::args_os().skip(1);
    let input = PathBuf::from(args.next().context("missing <input>")?);
    let output = PathBuf::from(args.next().context("missing <output>")?);
    if args.next().is_some() {
        anyhow::bail!("extra arguments; usage: extract_pickle <input> <output>");
    }

    let t0 = Instant::now();
    eprintln!("Parsing {}...", input.display());
    let rdr = open_input(&input)?;
    let snap = pickle::parse_snapshot(rdr)?;
    eprintln!(
        "  parsed in {:.1}s: {} events, {} unique frames, {} annotations",
        t0.elapsed().as_secs_f64(),
        snap.events.len(),
        snap.frame_strings.len(),
        snap.annotations.len()
    );

    eprintln!(
        "  {} segments-baseline blocks (live before recording, persisted)",
        snap.live_blocks.len()
    );

    let t1 = Instant::now();
    eprintln!("Writing {}...", output.display());
    let f = File::create(&output).with_context(|| format!("creating {}", output.display()))?;
    let mut w = BufWriter::with_capacity(1 << 20, f);
    write_json(&mut w, &snap)?;
    w.flush()?;
    eprintln!("  written in {:.1}s", t1.elapsed().as_secs_f64());
    eprintln!("Total: {:.1}s", t0.elapsed().as_secs_f64());
    Ok(())
}

/// Hand-rolled JSON writer that matches extract_snapshot.py's `json.dump(result, f)`
/// byte-for-byte: default separators ", " and ": ", no trailing newline,
/// ASCII-safe escaping for non-ASCII codepoints.
fn write_json<W: Write>(w: &mut W, snap: &pickle::Snapshot) -> Result<()> {
    w.write_all(b"{\"events\": [")?;
    for (i, ev) in snap.events.iter().enumerate() {
        if i > 0 {
            w.write_all(b", ")?;
        }
        write!(w, "[{}, {}, {}, {}, {}]", ev.0, ev.1, ev.2, ev.3, ev.4)?;
    }
    w.write_all(b"], \"frame_strings\": [")?;
    for (i, s) in snap.frame_strings.iter().enumerate() {
        if i > 0 {
            w.write_all(b", ")?;
        }
        write_json_str(w, s)?;
    }
    w.write_all(b"], \"annotations\": [")?;
    for (i, a) in snap.annotations.iter().enumerate() {
        if i > 0 {
            w.write_all(b", ")?;
        }
        w.write_all(b"{\"stage\": ")?;
        write_json_str(w, &a.stage)?;
        w.write_all(b", \"name\": ")?;
        write_json_str(w, &a.name)?;
        write!(w, ", \"time_us\": {}}}", a.time_us)?;
    }
    w.write_all(b"], \"live_blocks\": [")?;
    for (i, &(addr, size)) in snap.live_blocks.iter().enumerate() {
        if i > 0 {
            w.write_all(b", ")?;
        }
        write!(w, "[{}, {}]", addr, size)?;
    }
    w.write_all(b"]}")?;
    Ok(())
}

/// Match Python's json.dumps default (ensure_ascii=True): escape control chars
/// and non-ASCII codepoints with \uXXXX.
fn write_json_str<W: Write>(w: &mut W, s: &str) -> Result<()> {
    w.write_all(b"\"")?;
    for ch in s.chars() {
        match ch {
            '"' => w.write_all(b"\\\"")?,
            '\\' => w.write_all(b"\\\\")?,
            '\n' => w.write_all(b"\\n")?,
            '\r' => w.write_all(b"\\r")?,
            '\t' => w.write_all(b"\\t")?,
            '\x08' => w.write_all(b"\\b")?,
            '\x0c' => w.write_all(b"\\f")?,
            c if (c as u32) < 0x20 => write!(w, "\\u{:04x}", c as u32)?,
            c if (c as u32) < 0x7f => {
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                w.write_all(s.as_bytes())?;
            }
            c => {
                // Non-ASCII: escape as \uXXXX (surrogate pair if > BMP).
                let cp = c as u32;
                if cp <= 0xFFFF {
                    write!(w, "\\u{:04x}", cp)?;
                } else {
                    let cp = cp - 0x10000;
                    let hi = 0xD800 + (cp >> 10);
                    let lo = 0xDC00 + (cp & 0x3FF);
                    write!(w, "\\u{:04x}\\u{:04x}", hi, lo)?;
                }
            }
        }
    }
    w.write_all(b"\"")?;
    Ok(())
}
