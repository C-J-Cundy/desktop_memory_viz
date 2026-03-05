use anyhow::{Context, Result};
use clap::Parser;
use eframe::egui;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

// ── CLI ─────────────────────────────────────────────────────────────

/// Native GPU-accelerated CUDA memory snapshot visualizer.
///
/// Port of the PyTorch memory visualizer's "Active Memory Timeline" view.
/// Renders allocations as stacked polygons using egui/eframe.
/// Accepts .pickle files (auto-converts via Python) or pre-extracted .json files.
#[derive(Parser)]
#[command(name = "desktop-memory-viz")]
struct Cli {
    /// Path to the PyTorch memory snapshot (.pickle or pre-extracted .json)
    input: PathBuf,

    /// Only show annotations matching this pattern (e.g., "##" for user annotations)
    #[arg(long)]
    annotation_filter: Option<String>,

    /// Show all annotations including PyTorch internals (dynamo_timed, etc.)
    #[arg(long, default_value = "false")]
    all_annotations: bool,

    /// Maximum number of allocations to render individually (rest are summarized)
    #[arg(long, default_value = "1000000000")]
    max_entries: usize,

    /// HuggingFace model ID (e.g., "google/gemma-2-2b"). Fetches config.json to
    /// get hidden_size and intermediate_size for tensor shape display.
    #[arg(long)]
    model: Option<String>,

    /// Override vocab size (e.g., when using added tokens)
    #[arg(long)]
    vocab_size: Option<usize>,

    /// Show 4-bit and int8 factorizations in tooltip (off by default)
    #[arg(long)]
    quantized: bool,

    /// GPU model for showing memory capacity line (e.g., "H100", "H100-80", "A100-80", "A100-40")
    #[arg(long)]
    gpu: Option<String>,
}

// ── Model config (fetched from HuggingFace) ─────────────────────────

#[derive(Clone)]
struct ModelConfig {
    model_id: String,
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
}

/// Fetch config.json from HuggingFace and extract hidden/intermediate sizes.
/// Reads HF token from ~/.cache/huggingface/token or HF_TOKEN env var for gated models.
fn fetch_model_config(model_id: &str) -> Result<ModelConfig> {
    let url = format!(
        "https://huggingface.co/{}/resolve/main/config.json",
        model_id
    );
    eprintln!("Fetching model config from {}...", url);

    // Try to read HF token for gated models
    let token = std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join(".cache/huggingface/token"))
        .and_then(|p| std::fs::read_to_string(p).ok())
        .or_else(|| std::env::var("HF_TOKEN").ok())
        .map(|t| t.trim().to_string());

    let mut req = ureq::get(&url);
    if let Some(ref tok) = token {
        req = req.header("Authorization", &format!("Bearer {}", tok));
    }

    let body: String = req
        .call()
        .with_context(|| format!("failed to fetch config from {}", url))?
        .body_mut()
        .read_to_string()
        .context("failed to read response body")?;

    let config: serde_json::Value =
        serde_json::from_str(&body).context("failed to parse config.json")?;

    // Handle both flat configs and nested configs (e.g., multimodal models
    // with text_config.hidden_size)
    let text_cfg = if config.get("text_config").is_some() {
        &config["text_config"]
    } else {
        &config
    };

    let hidden_size = text_cfg["hidden_size"]
        .as_u64()
        .context("config.json missing hidden_size (checked text_config too)")? as usize;
    let intermediate_size = text_cfg["intermediate_size"]
        .as_u64()
        .context("config.json missing intermediate_size")? as usize;
    let mut vocab_size = config["vocab_size"]
        .as_u64()
        .or_else(|| text_cfg["vocab_size"].as_u64())
        .unwrap_or(0) as usize;

    // Fallback: fetch tokenizer.json and count vocab entries
    if vocab_size == 0 {
        let tok_url = format!(
            "https://huggingface.co/{}/resolve/main/tokenizer.json",
            model_id
        );
        eprintln!("  vocab_size not in config.json, trying tokenizer.json...");
        let mut tok_req = ureq::get(&tok_url);
        if let Some(ref tok) = token {
            tok_req = tok_req.header("Authorization", &format!("Bearer {}", tok));
        }
        if let Ok(mut resp) = tok_req.call() {
            if let Ok(tok_body) = resp.body_mut().read_to_string() {
                if let Ok(tok_json) = serde_json::from_str::<serde_json::Value>(&tok_body) {
                    if let Some(vocab) = tok_json.get("model").and_then(|m| m.get("vocab")) {
                        if let Some(obj) = vocab.as_object() {
                            vocab_size = obj.len();
                        } else if let Some(arr) = vocab.as_array() {
                            vocab_size = arr.len();
                        }
                    }
                }
            }
        }
    }

    // Gemma 3 models have vocab_size=262145 but this isn't in config.json and
    // tokenizer.json reports 262144. Hardcode the correct value.
    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if model_type.starts_with("gemma3") && vocab_size != 262145 {
        eprintln!(
            "  Warning: Gemma 3 detected (model_type={}), overriding vocab_size {} -> 262145",
            model_type, vocab_size
        );
        vocab_size = 262145;
    }

    eprintln!(
        "  Model: {} | hidden_size={} | intermediate_size={} | vocab_size={}",
        model_id, hidden_size, intermediate_size, vocab_size
    );

    Ok(ModelConfig {
        model_id: model_id.to_string(),
        hidden_size,
        intermediate_size,
        vocab_size,
    })
}

// ── JSON input structures (from extract_snapshot.py) ────────────────

#[derive(Deserialize)]
struct SnapshotJson {
    /// Events: [action_code, addr, size, time_us, frame_idx]
    events: Vec<(u8, u64, u64, i64, u32)>,
    frame_strings: Vec<String>,
    annotations: Vec<JsonAnnotation>,
}

#[derive(Deserialize)]
struct JsonAnnotation {
    stage: String,
    name: String,
    time_us: i64,
}

// ── Allocation rectangle (from pairing alloc/free events) ──────────

struct AllocRect {
    size: u64,
    start_us: i64,
    end_us: i64,
    frame_idx: u32,
}

// ── Paired annotation ───────────────────────────────────────────────

struct PairedAnnotation {
    name: String,
    start_us: f64,
    end_us: Option<f64>,
}

// ── Pair alloc/free events into rectangles ──────────────────────────

fn pair_alloc_free(events: &[(u8, u64, u64, i64, u32)], last_time: i64) -> Vec<AllocRect> {
    let mut live: HashMap<u64, (u64, i64, u32)> = HashMap::new();
    let mut rects: Vec<AllocRect> = Vec::new();

    for &(action, addr, size, time_us, frame_idx) in events {
        match action {
            0 => {
                // alloc
                live.insert(addr, (size, time_us, frame_idx));
            }
            1 | 2 => {
                // free_requested or free_completed
                if let Some((alloc_size, start_us, alloc_frame)) = live.remove(&addr) {
                    if start_us < time_us {
                        rects.push(AllocRect {
                            size: alloc_size,
                            start_us,
                            end_us: time_us,
                            frame_idx: alloc_frame,
                        });
                    }
                }
            }
            _ => {
                // segment events — skip
            }
        }
    }

    // Still-live allocations: end at recording end
    for (_addr, (size, start_us, frame_idx)) in live.drain() {
        rects.push(AllocRect {
            size,
            start_us,
            end_us: last_time,
            frame_idx,
        });
    }

    rects
}

// ── Pair annotations ────────────────────────────────────────────────

fn pair_annotations(
    raw: &[JsonAnnotation],
    filter: &Option<String>,
    all_annotations: bool,
) -> Vec<PairedAnnotation> {
    let mut starts: HashMap<String, i64> = HashMap::new();
    let mut paired = Vec::new();

    for ann in raw {
        // Apply explicit filter if provided
        if let Some(ref f) = filter {
            if !ann.name.contains(f.as_str()) {
                continue;
            }
        }
        // Auto-filter PyTorch internals unless --all-annotations
        if !all_annotations && filter.is_none() {
            if ann.name.contains("(dynamo_timed)")
                || ann.name.contains("pad_mm_benchmark")
                || ann.name.contains("CompiledFxGraph")
                || (ann.name.contains('#') && !ann.name.contains("##"))
            {
                continue;
            }
        }
        match ann.stage.as_str() {
            "START" => {
                starts.insert(ann.name.clone(), ann.time_us);
            }
            "END" => {
                let start = starts.remove(&ann.name);
                paired.push(PairedAnnotation {
                    name: ann.name.clone(),
                    start_us: start.map(|s| s as f64).unwrap_or(ann.time_us as f64),
                    end_us: Some(ann.time_us as f64),
                });
            }
            _ => {}
        }
    }
    for (name, time_us) in starts {
        paired.push(PairedAnnotation {
            name,
            start_us: time_us as f64,
            end_us: None,
        });
    }
    paired.sort_by(|a, b| {
        a.start_us
            .partial_cmp(&b.start_us)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    paired
}

// ── Pickle-to-JSON conversion ───────────────────────────────────────

fn convert_pickle_to_json(pickle_path: &PathBuf) -> Result<PathBuf> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()));

    let script_candidates = [
        exe_dir.as_ref().map(|d| d.join("extract_snapshot.py")),
        exe_dir
            .as_ref()
            .map(|d| d.join("../../extract_snapshot.py")),
        exe_dir
            .as_ref()
            .map(|d| d.join("../../../extract_snapshot.py")),
        Some(PathBuf::from("extract_snapshot.py")),
    ];

    let script_path = script_candidates
        .into_iter()
        .flatten()
        .find(|p| p.exists())
        .context(
            "Could not find extract_snapshot.py. \
             Place it next to the desktop-memory-viz binary, or run from the project directory.\n\
             Alternatively, pre-convert the pickle: python extract_snapshot.py input.pickle output.json",
        )?;

    let json_path = pickle_path.with_extension("extracted.json");

    // Skip extraction if JSON already exists and is newer than the pickle
    if json_path.exists() {
        let pickle_modified = fs::metadata(pickle_path).ok().and_then(|m| m.modified().ok());
        let json_modified = fs::metadata(&json_path).ok().and_then(|m| m.modified().ok());
        if let (Some(pm), Some(jm)) = (pickle_modified, json_modified) {
            if jm >= pm {
                eprintln!("Using cached JSON: {}", json_path.display());
                return Ok(json_path);
            }
        }
    }

    eprintln!(
        "Converting pickle to JSON via Python ({})... (this may take a minute for large snapshots)",
        script_path.display()
    );
    let output = Command::new("python3")
        .arg(&script_path)
        .arg(pickle_path)
        .arg(&json_path)
        .output()
        .context("failed to run python3. Is Python installed?")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Python extraction failed:\n{}", stderr);
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    eprint!("{}", stderr);

    Ok(json_path)
}

// ── Polygon data (ported from PyTorch MemoryViz.js) ────────────────

/// One tracked allocation's polygon data.
/// The bottom edge follows (times_us[i], offsets[i]).
/// The top edge is offset + size.
struct AllocPolygon {
    rect_idx: u32,
    size: u64,
    /// Parallel arrays: at times_us[i], the y_offset is offsets[i].
    times_us: Vec<i64>,
    offsets: Vec<f64>,
    color: egui::Color32,
}

/// Summarized band for untracked (small) allocations.
/// Sits on top of the tracked allocations.
struct SummarizedBand {
    /// Parallel arrays: at times_us[i], the band offset is offsets[i]
    /// and height is sizes[i].
    times_us: Vec<i64>,
    offsets: Vec<f64>,
    sizes: Vec<f64>,
}

/// Precomputed layout data (direct port of PyTorch's process_alloc_data).
struct PolygonLayout {
    /// Individual tracked allocation polygons.
    polygons: Vec<AllocPolygon>,
    /// Summarized band for untracked allocations.
    summarized: SummarizedBand,
    /// Maps rect_idx -> polygon index (u32::MAX if not tracked individually).
    rect_to_poly: Vec<u32>,
    /// Original allocation rectangles (for hover info).
    rects: Vec<AllocRect>,
    /// Frame strings for hover info.
    frame_strings: Vec<String>,
    /// Paired annotations.
    annotations: Vec<PairedAnnotation>,
    /// Time range in microseconds.
    time_min_us: i64,
    time_max_us: i64,
    /// Peak total memory (tracked + summarized).
    peak_bytes: u64,
    /// Stats.
    total_events: usize,
    total_rects: usize,
    max_entries: usize,
}

/// Build the polygon layout (direct port of PyTorch's process_alloc_data).
///
/// 1. Select top max_entries allocations by size for individual tracking
/// 2. Process events in time order, maintaining an ordered stack
/// 3. On alloc of tracked: push to stack, offset = total_tracked_mem
/// 4. On free of tracked: remove, shift all above down by freed size
/// 5. On alloc/free of untracked: adjust summarized band
fn build_polygon_layout(
    rects: Vec<AllocRect>,
    frame_strings: Vec<String>,
    annotations: Vec<PairedAnnotation>,
    time_min_us: i64,
    time_max_us: i64,
    total_events: usize,
    max_entries: usize,
) -> PolygonLayout {
    let total_rects = rects.len();

    // 1. Select top max_entries allocations by size
    let mut by_size: Vec<(u64, u32)> = rects
        .iter()
        .enumerate()
        .map(|(i, r)| (r.size, i as u32))
        .collect();
    by_size.sort_unstable_by(|a, b| b.0.cmp(&a.0)); // largest first

    let tracked: HashSet<u32> = by_size
        .iter()
        .take(max_entries)
        .map(|&(_, idx)| idx)
        .collect();

    let tracked_count = tracked.len();
    let summarized_count = total_rects - tracked_count;
    eprintln!(
        "  {} tracked individually, {} summarized",
        tracked_count, summarized_count
    );

    // 2. Build sorted event list
    let n = rects.len();
    let mut sorted_events: Vec<(i64, u8, u32)> = Vec::with_capacity(n * 2);
    for (i, r) in rects.iter().enumerate() {
        sorted_events.push((r.start_us, 1, i as u32)); // alloc
        sorted_events.push((r.end_us, 0, i as u32)); // free
    }
    // Sort by time; at same time, allocs before frees (type 1 > type 0, so reverse)
    sorted_events.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));

    // 3. Process events in time order
    let mut current: Vec<u32> = Vec::new(); // ordered stack of tracked rect_idxs
    let mut polygon_map: HashMap<u32, usize> = HashMap::new(); // rect_idx -> index in polygons vec
    let mut polygons: Vec<AllocPolygon> = Vec::with_capacity(tracked_count);
    let mut rect_to_poly: Vec<u32> = vec![u32::MAX; n];

    let mut total_mem: f64 = 0.0; // sum of tracked live allocations
    let mut total_summarized_mem: f64 = 0.0;
    let mut peak: f64 = 0.0;

    let mut summarized = SummarizedBand {
        times_us: Vec::new(),
        offsets: Vec::new(),
        sizes: Vec::new(),
    };

    // Record summarized band state
    let record_summarized =
        |s: &mut SummarizedBand, time: i64, total_mem: f64, total_summarized: f64| {
            s.times_us.push(time);
            s.offsets.push(total_mem);
            s.sizes.push(total_summarized);
        };

    for &(time_us, event_type, rect_idx) in &sorted_events {
        let size = rects[rect_idx as usize].size as f64;

        if !tracked.contains(&rect_idx) {
            // Untracked: adjust summarized band
            match event_type {
                1 => {
                    total_summarized_mem += size;
                    record_summarized(
                        &mut summarized,
                        time_us,
                        total_mem,
                        total_summarized_mem,
                    );
                }
                0 => {
                    total_summarized_mem = (total_summarized_mem - size).max(0.0);
                    record_summarized(
                        &mut summarized,
                        time_us,
                        total_mem,
                        total_summarized_mem,
                    );
                }
                _ => {}
            }
            let total = total_mem + total_summarized_mem;
            if total > peak {
                peak = total;
            }
            continue;
        }

        match event_type {
            1 => {
                // Alloc: push onto stack
                current.push(rect_idx);
                let poly_idx = polygons.len();
                polygons.push(AllocPolygon {
                    rect_idx,
                    size: rects[rect_idx as usize].size,
                    times_us: vec![time_us],
                    offsets: vec![total_mem],
                    color: alloc_color(rect_idx),
                });
                polygon_map.insert(rect_idx, poly_idx);
                rect_to_poly[rect_idx as usize] = poly_idx as u32;
                total_mem += size;
                record_summarized(
                    &mut summarized,
                    time_us,
                    total_mem,
                    total_summarized_mem,
                );
            }
            0 => {
                // Free: find in current, remove, shift above down
                if let Some(stack_idx) = current.iter().rposition(|&x| x == rect_idx) {
                    // Record final position for freed allocation
                    if let Some(&poly_idx) = polygon_map.get(&rect_idx) {
                        let last_offset = *polygons[poly_idx].offsets.last().unwrap_or(&0.0);
                        polygons[poly_idx].times_us.push(time_us);
                        polygons[poly_idx].offsets.push(last_offset);
                    }

                    current.remove(stack_idx);

                    // Shift all elements above down
                    for j in stack_idx..current.len() {
                        let above_rect_idx = current[j];
                        if let Some(&poly_idx) = polygon_map.get(&above_rect_idx) {
                            let last_offset =
                                *polygons[poly_idx].offsets.last().unwrap_or(&0.0);
                            // Record old position
                            polygons[poly_idx].times_us.push(time_us);
                            polygons[poly_idx].offsets.push(last_offset);
                            // Record new position (shifted down)
                            polygons[poly_idx].times_us.push(time_us);
                            polygons[poly_idx].offsets.push(last_offset - size);
                        }
                    }

                    total_mem -= size;
                    record_summarized(
                        &mut summarized,
                        time_us,
                        total_mem,
                        total_summarized_mem,
                    );
                }
            }
            _ => {}
        }

        let total = total_mem + total_summarized_mem;
        if total > peak {
            peak = total;
        }
    }

    // Finalize: extend all live tracked allocations to end time
    for &rect_idx in &current {
        if let Some(&poly_idx) = polygon_map.get(&rect_idx) {
            let last_offset = *polygons[poly_idx].offsets.last().unwrap_or(&0.0);
            polygons[poly_idx].times_us.push(time_max_us);
            polygons[poly_idx].offsets.push(last_offset);
        }
    }
    // Final summarized state
    record_summarized(
        &mut summarized,
        time_max_us,
        total_mem,
        total_summarized_mem,
    );

    eprintln!("  Peak memory: {:.2} GB", peak / 1e9);
    eprintln!("  {} tracked polygons", polygons.len());
    let total_history: usize = polygons.iter().map(|p| p.times_us.len()).sum();
    eprintln!("  {} total offset history entries", total_history);
    eprintln!("  {} summarized band entries", summarized.times_us.len());

    PolygonLayout {
        polygons,
        summarized,
        rect_to_poly,
        rects,
        frame_strings,
        annotations,
        time_min_us,
        time_max_us,
        peak_bytes: peak as u64,
        total_events,
        total_rects,
        max_entries,
    }
}

// ── Color generation ────────────────────────────────────────────────

/// Muted, professional palette based on FAR AI brand colors.
/// 20 distinct colors visible against dark navy background.
const PALETTE: [egui::Color32; 20] = [
    egui::Color32::from_rgb(247, 157, 92),  // Orange
    egui::Color32::from_rgb(126, 188, 230), // Futures Blue
    egui::Color32::from_rgb(137, 128, 245), // Labs Purple
    egui::Color32::from_rgb(90, 138, 143),  // Teal
    egui::Color32::from_rgb(212, 137, 106), // Muted Coral
    egui::Color32::from_rgb(139, 170, 122), // Sage Green
    egui::Color32::from_rgb(201, 168, 86),  // Warm Gold
    egui::Color32::from_rgb(176, 122, 165), // Dusty Mauve
    egui::Color32::from_rgb(91, 158, 166),  // Ocean Teal
    egui::Color32::from_rgb(212, 165, 116), // Warm Sand
    egui::Color32::from_rgb(122, 143, 176), // Steel Blue
    egui::Color32::from_rgb(158, 187, 138), // Moss Green
    egui::Color32::from_rgb(196, 142, 122), // Terracotta
    egui::Color32::from_rgb(107, 142, 160), // Slate Blue
    egui::Color32::from_rgb(138, 126, 176), // Soft Indigo
    egui::Color32::from_rgb(136, 196, 168), // Seafoam
    egui::Color32::from_rgb(201, 138, 150), // Dusty Rose
    egui::Color32::from_rgb(160, 168, 112), // Olive
    egui::Color32::from_rgb(232, 184, 154), // Peach
    egui::Color32::from_rgb(120, 144, 168), // Dusk Blue
];

/// Pick a color from the palette. Uses a prime multiplier to ensure
/// adjacent allocation indices map to distant palette entries.
fn alloc_color(idx: u32) -> egui::Color32 {
    PALETTE[((idx as usize) * 7 + 5) % PALETTE.len()]
}

// ── Rasterized view cache ────────────────────────────────────────────

/// Rasterized chart data: a pixel buffer + hover map.
/// Replaces the old VisibleRect strip list — instead of drawing 500K+ egui
/// rectangles per frame, we rasterize into a pixel buffer and display as a
/// single texture. Per-frame cost drops from O(rects) to O(1).
struct ViewCache {
    /// RGBA pixel buffer (width * height), row-major, top-to-bottom.
    pixels: Vec<egui::Color32>,
    /// Allocation rect_idx at each pixel (u32::MAX = background/summarized).
    hover_map: Vec<u32>,
    view_x_min_us: f64,
    view_x_max_us: f64,
    view_y_min_bytes: f64,
    view_y_max_bytes: f64,
    width_px: usize,
    height_px: usize,
}

/// Rasterize the polygon layout into a pixel buffer for the given viewport.
///
/// Paints each tracked polygon's visible segments directly into the pixel buffer,
/// then paints the summarized band on top. Also fills a hover_map for O(1) hover
/// detection. Total pixel writes bounded by width*height regardless of polygon count.
fn build_view_cache(
    layout: &PolygonLayout,
    view_x_min_us: f64,
    view_x_max_us: f64,
    view_y_min_bytes: f64,
    view_y_max_bytes: f64,
    width_px: usize,
    height_px: usize,
) -> ViewCache {
    let bg = egui::Color32::from_rgb(20, 20, 38);
    let mut pixels = vec![bg; width_px * height_px];
    let mut hover_map = vec![u32::MAX; width_px * height_px];

    let x_range = view_x_max_us - view_x_min_us;
    let y_range = view_y_max_bytes - view_y_min_bytes;
    if x_range <= 0.0 || y_range <= 0.0 {
        return ViewCache {
            pixels,
            hover_map,
            view_x_min_us,
            view_x_max_us,
            view_y_min_bytes,
            view_y_max_bytes,
            width_px,
            height_px,
        };
    }

    let x_scale = width_px as f64 / x_range;
    let y_scale = height_px as f64 / y_range;

    // Helper: paint a data-space rectangle into the pixel buffer
    let paint_rect =
        |pixels: &mut [egui::Color32],
         hover_map: &mut [u32],
         t_start: f64,
         t_end: f64,
         y_bottom: f64,
         y_top: f64,
         color: egui::Color32,
         rect_idx: u32| {
            let px_left = ((t_start - view_x_min_us) * x_scale) as usize;
            let px_right =
                (((t_end - view_x_min_us) * x_scale).ceil() as usize).min(width_px);
            // Y is inverted: top of screen = max bytes
            let py_top =
                (((view_y_max_bytes - y_top) * y_scale) as usize).min(height_px);
            let py_bottom =
                (((view_y_max_bytes - y_bottom) * y_scale).ceil() as usize).min(height_px);

            if px_right <= px_left || py_bottom <= py_top {
                return;
            }

            for py in py_top..py_bottom {
                let row_start = py * width_px;
                pixels[row_start + px_left..row_start + px_right].fill(color);
                hover_map[row_start + px_left..row_start + px_right].fill(rect_idx);
            }
        };

    // Paint each tracked polygon
    for poly in &layout.polygons {
        let size = poly.size as f64;
        let times = &poly.times_us;
        let offsets = &poly.offsets;
        if times.len() < 2 {
            continue;
        }

        // Binary search for first entry with time >= view_x_min
        let start_idx = times.partition_point(|&t| (t as f64) < view_x_min_us);
        let start_idx = if start_idx > 0 { start_idx - 1 } else { 0 };

        for i in start_idx..times.len().saturating_sub(1) {
            let t0 = times[i] as f64;
            let t1 = times[i + 1] as f64;

            if t1 <= view_x_min_us {
                continue;
            }
            if t0 >= view_x_max_us {
                break;
            }

            let offset = offsets[i];
            let y_bottom = offset;
            let y_top = offset + size;

            if y_top < view_y_min_bytes || y_bottom > view_y_max_bytes {
                continue;
            }

            let clip_start = t0.max(view_x_min_us);
            let clip_end = t1.min(view_x_max_us);

            paint_rect(
                &mut pixels,
                &mut hover_map,
                clip_start,
                clip_end,
                y_bottom,
                y_top,
                poly.color,
                poly.rect_idx,
            );
        }
    }

    // Paint summarized band (grey, on top of tracked allocations)
    let stimes = &layout.summarized.times_us;
    let soffsets = &layout.summarized.offsets;
    let ssizes = &layout.summarized.sizes;
    let summarized_color = egui::Color32::from_rgb(100, 100, 100);

    if stimes.len() >= 2 {
        let start_idx = stimes.partition_point(|&t| (t as f64) < view_x_min_us);
        let start_idx = if start_idx > 0 { start_idx - 1 } else { 0 };

        for i in start_idx..stimes.len().saturating_sub(1) {
            let t0 = stimes[i] as f64;
            let t1 = stimes[i + 1] as f64;

            if t1 <= view_x_min_us {
                continue;
            }
            if t0 >= view_x_max_us {
                break;
            }

            let size = ssizes[i];
            if size <= 0.0 {
                continue;
            }

            let offset = soffsets[i];
            let y_bottom = offset;
            let y_top = offset + size;

            if y_top < view_y_min_bytes || y_bottom > view_y_max_bytes {
                continue;
            }

            let clip_start = t0.max(view_x_min_us);
            let clip_end = t1.min(view_x_max_us);

            paint_rect(
                &mut pixels,
                &mut hover_map,
                clip_start,
                clip_end,
                y_bottom,
                y_top,
                summarized_color,
                u32::MAX,
            );
        }
    }

    ViewCache {
        pixels,
        hover_map,
        view_x_min_us,
        view_x_max_us,
        view_y_min_bytes,
        view_y_max_bytes,
        width_px,
        height_px,
    }
}

// ── App state ───────────────────────────────────────────────────────

struct MemoryVizApp {
    layout: PolygonLayout,

    // Viewport in data coordinates
    view_x_min_us: f64,
    view_x_max_us: f64,
    view_y_min_bytes: f64,
    view_y_max_bytes: f64,

    // Rasterized view cache
    cache: Option<ViewCache>,

    // Chart texture (uploaded from pixel buffer)
    chart_texture: Option<egui::TextureHandle>,

    // Hover info (current frame — cleared when not hovering an allocation)
    hover_info: Option<HoverInfo>,

    // Persistent hover info for the bottom bar (kept until a new allocation is hovered)
    last_hover_info: Option<HoverInfo>,

    // Pinned allocation rect_idx for highlight drawing
    pinned_rect_idx: Option<u32>,

    // Cmd+drag region selection
    drag_select: Option<DragSelect>,

    // Show annotations toggle
    show_annotations: bool,

    // Tooltip dismissed via right-click (resets when hovering a different allocation)
    tooltip_dismissed: bool,
    dismissed_rect_idx: u32,

    // Model config for tensor shape display
    model_config: Option<ModelConfig>,

    // Show quantized dtype factorizations (4-bit, int8)
    show_quantized: bool,

    // Only show exact shape matches (no near-miss annotations)
    exact_shapes_only: bool,

    // GPU memory capacity in bytes (for drawing capacity line)
    gpu_capacity_bytes: Option<f64>,
    gpu_label: Option<String>,

    // FPS counter
    last_frame_time: std::time::Instant,
    fps_smooth: f64,
}

#[derive(Clone)]
struct HoverInfo {
    size_bytes: u64,
    start_us: i64,
    end_us: i64,
    frame_str: String,
    /// Total memory allocated below this tensor (its y offset)
    total_allocated_bytes: u64,
    /// Total memory at the time this tensor was freed
    total_at_dealloc_bytes: u64,
}

/// State for cmd+drag region selection
struct DragSelect {
    start_us: f64,
    start_bytes: f64,
    start_screen_x: f32,
    start_screen_y: f32,
}

impl MemoryVizApp {
    fn new(
        layout: PolygonLayout,
        model_config: Option<ModelConfig>,
        show_quantized: bool,
        gpu: Option<String>,
    ) -> Self {
        let view_x_min_us = layout.time_min_us as f64;
        let view_x_max_us = layout.time_max_us as f64;
        let view_y_min_bytes = 0.0;
        let view_y_max_bytes = layout.peak_bytes as f64 * 1.05;

        let (gpu_capacity_bytes, gpu_label) = match gpu.as_deref() {
            Some(g) => {
                let capacity: Option<f64> = match g.to_uppercase().as_str() {
                    "H100" | "H100-80" => Some(80.0 * 1e9),
                    "A100" | "A100-80" => Some(80.0 * 1e9),
                    "A100-40" => Some(40.0 * 1e9),
                    "H200" => Some(141.0 * 1e9),
                    "A10G" => Some(24.0 * 1e9),
                    "L40S" => Some(48.0 * 1e9),
                    "V100" | "V100-32" => Some(32.0 * 1e9),
                    "V100-16" => Some(16.0 * 1e9),
                    "4090" | "RTX4090" => Some(24.0 * 1e9),
                    _ => {
                        eprintln!("Warning: unknown GPU '{}', no capacity line will be shown", g);
                        None
                    }
                };
                (capacity, capacity.map(|_| g.to_string()))
            }
            None => (None, None),
        };

        MemoryVizApp {
            layout,
            view_x_min_us,
            view_x_max_us,
            view_y_min_bytes,
            view_y_max_bytes,
            cache: None,
            chart_texture: None,
            hover_info: None,
            last_hover_info: None,
            pinned_rect_idx: None,
            drag_select: None,
            show_annotations: true,
            tooltip_dismissed: false,
            dismissed_rect_idx: u32::MAX,
            model_config,
            show_quantized,
            exact_shapes_only: false,
            gpu_capacity_bytes,
            gpu_label,
            last_frame_time: std::time::Instant::now(),
            fps_smooth: 0.0,
        }
    }

    fn invalidate_cache(&mut self) {
        self.cache = None;
    }

    fn ensure_cache(&mut self, ctx: &egui::Context, width_px: usize, height_px: usize) {
        let needs_rebuild = match &self.cache {
            None => true,
            Some(c) => {
                c.view_x_min_us != self.view_x_min_us
                    || c.view_x_max_us != self.view_x_max_us
                    || c.view_y_min_bytes != self.view_y_min_bytes
                    || c.view_y_max_bytes != self.view_y_max_bytes
                    || c.width_px != width_px
                    || c.height_px != height_px
            }
        };

        if needs_rebuild {
            let cache = build_view_cache(
                &self.layout,
                self.view_x_min_us,
                self.view_x_max_us,
                self.view_y_min_bytes,
                self.view_y_max_bytes,
                width_px,
                height_px,
            );

            // Upload pixel buffer as texture
            let image = egui::ColorImage {
                size: [cache.width_px, cache.height_px],
                pixels: cache.pixels.clone(),
            };
            match &mut self.chart_texture {
                Some(handle) => handle.set(image, egui::TextureOptions::NEAREST),
                None => {
                    self.chart_texture =
                        Some(ctx.load_texture("chart", image, egui::TextureOptions::NEAREST));
                }
            }

            self.cache = Some(cache);
        }
    }

    /// Format exact byte count with comma separators (e.g., "2,621,440")
    fn format_exact_bytes(bytes: u64) -> String {
        let s = bytes.to_string();
        let mut result = String::with_capacity(s.len() + s.len() / 3);
        for (i, c) in s.chars().enumerate() {
            if i > 0 && (s.len() - i) % 3 == 0 {
                result.push(',');
            }
            result.push(c);
        }
        result
    }

    /// Try to express a byte count as a tensor shape using model config dimensions.
    /// Tries each dtype (4-bit, int8, bf16, fp32) and factors by hidden_size (h)
    /// and intermediate_size (i). Returns all clean factorizations.
    fn format_tensor_shape(&self, bytes: u64) -> Option<String> {
        let cfg = self.model_config.as_ref()?;
        let h = cfg.hidden_size as u64;
        let i = cfg.intermediate_size as u64;
        let v = cfg.vocab_size as u64;
        if h == 0 {
            return None;
        }

        // (numerator, denominator, label) such that elements = bytes * num / den
        let all_dtypes: &[(u64, u64, &str)] = &[
            (2, 1, "4-bit"), // 2 elements per byte
            (1, 1, "int8"),  // 1 element per byte
            (1, 2, "bf16"),  // 0.5 elements per byte
            (1, 4, "fp32"),  // 0.25 elements per byte
        ];
        let non_quantized: &[(u64, u64, &str)] = &[
            (1, 2, "bf16"),
            (1, 4, "fp32"),
        ];
        let dtypes = if self.show_quantized { all_dtypes } else { non_quantized };

        let mut results: Vec<String> = Vec::new();

        for &(num, den, dtype_label) in dtypes {
            // elements = bytes * num / den, must be exact
            let numer = bytes * num;
            if numer % den != 0 {
                continue;
            }
            let elements = numer / den;
            if elements == 0 {
                continue;
            }

            // Try factorizations, most specific first
            if let Some(desc) = Self::try_factor(elements, h, i, v) {
                results.push(format!("{}: {}", dtype_label, desc));
            }
        }

        // If no exact matches, try near-miss: check if bytes is within a small
        // tolerance of a clean factorization (e.g. allocator overhead / alignment).
        if results.is_empty() && !self.exact_shapes_only {
            let max_slop: u64 = 512; // bytes
            for &(num, den, dtype_label) in dtypes {
                for delta in 1..=max_slop {
                    if bytes <= delta {
                        continue;
                    }
                    let candidate = bytes - delta;
                    let numer = candidate * num;
                    if numer % den != 0 {
                        continue;
                    }
                    let elements = numer / den;
                    if elements == 0 {
                        continue;
                    }
                    if let Some(desc) = Self::try_factor(elements, h, i, v) {
                        results.push(format!("≈{}: {} (+{}B)", dtype_label, desc, delta));
                        break; // one near-miss per dtype is enough
                    }
                }
            }
        }

        if results.is_empty() {
            None
        } else {
            Some(results.join("  |  "))
        }
    }

    /// Try to factor `elements` as products of hidden_size (h), intermediate_size (i),
    /// vocab_size (v), (v+1) for padded vocab, and (2v+1) for tied embed/unembed.
    fn try_factor(elements: u64, h: u64, i: u64, v: u64) -> Option<String> {
        let vp = if v > 0 { v + 1 } else { 0 }; // padded vocab
        let v2 = if v > 0 { 2 * v + 1 } else { 0 }; // tied embed+unembed

        // v x h (embedding / lm_head)
        if v > 0 && h > 0 && elements % (v * h) == 0 {
            let n = elements / (v * h);
            if n == 1 {
                return Some("v x h".to_string());
            }
            return Some(format!("{} x v x h", n));
        }

        // (v+1) x h (padded embedding)
        if vp > 0 && h > 0 && elements % (vp * h) == 0 {
            let n = elements / (vp * h);
            if n == 1 {
                return Some("(v+1) x h".to_string());
            }
            return Some(format!("{} x (v+1) x h", n));
        }

        // (2v+1) x h (tied embed+unembed)
        if v2 > 0 && h > 0 && elements % (v2 * h) == 0 {
            let n = elements / (v2 * h);
            if n == 1 {
                return Some("(2v+1) x h".to_string());
            }
            return Some(format!("{} x (2v+1) x h", n));
        }

        // h x i or i x h
        if h > 0 && i > 0 && elements % (h * i) == 0 {
            let n = elements / (h * i);
            if n == 1 {
                return Some("h x i".to_string());
            }
            return Some(format!("{} x h x i", n));
        }

        // h x h
        if h > 0 && elements % (h * h) == 0 {
            let n = elements / (h * h);
            if n == 1 {
                return Some("h x h".to_string());
            }
            return Some(format!("{} x h x h", n));
        }

        // v x i
        if v > 0 && i > 0 && elements % (v * i) == 0 {
            let n = elements / (v * i);
            if n == 1 {
                return Some("v x i".to_string());
            }
            return Some(format!("{} x v x i", n));
        }

        // (v+1) x i
        if vp > 0 && i > 0 && elements % (vp * i) == 0 {
            let n = elements / (vp * i);
            if n == 1 {
                return Some("(v+1) x i".to_string());
            }
            return Some(format!("{} x (v+1) x i", n));
        }

        // (2v+1) x i
        if v2 > 0 && i > 0 && elements % (v2 * i) == 0 {
            let n = elements / (v2 * i);
            if n == 1 {
                return Some("(2v+1) x i".to_string());
            }
            return Some(format!("{} x (2v+1) x i", n));
        }

        // i x i (less common but possible)
        if i > 0 && i != h && elements % (i * i) == 0 {
            let n = elements / (i * i);
            if n == 1 {
                return Some("i x i".to_string());
            }
            return Some(format!("{} x i x i", n));
        }

        // N x v
        if v > 0 && v != h && v != i && elements % v == 0 {
            let n = elements / v;
            if n == 1 {
                return Some("[v]".to_string());
            }
            return Some(format!("{} x v", n));
        }

        // N x (v+1)
        if vp > 0 && vp != h && vp != i && elements % vp == 0 {
            let n = elements / vp;
            if n == 1 {
                return Some("[(v+1)]".to_string());
            }
            return Some(format!("{} x (v+1)", n));
        }

        // N x (2v+1)
        if v2 > 0 && v2 != h && v2 != i && elements % v2 == 0 {
            let n = elements / v2;
            if n == 1 {
                return Some("[(2v+1)]".to_string());
            }
            return Some(format!("{} x (2v+1)", n));
        }

        // N x i
        if i > 0 && i != h && elements % i == 0 {
            let n = elements / i;
            if n == 1 {
                return Some("[i]".to_string());
            }
            return Some(format!("{} x i", n));
        }

        // N x h
        if h > 0 && elements % h == 0 {
            let n = elements / h;
            if n == 1 {
                return Some("[h]".to_string());
            }
            return Some(format!("{} x h", n));
        }

        None
    }

    fn format_bytes(bytes: f64) -> String {
        let abs = bytes.abs();
        if abs >= 1e9 {
            format!("{:.2} GB", bytes / 1e9)
        } else if abs >= 1e6 {
            format!("{:.1} MB", bytes / 1e6)
        } else if abs >= 1e3 {
            format!("{:.0} KB", bytes / 1e3)
        } else {
            format!("{:.0} B", bytes)
        }
    }

    fn format_time_us(us: f64) -> String {
        let sec = us / 1e6;
        if sec >= 60.0 {
            format!("{:.1}m", sec / 60.0)
        } else if sec >= 1.0 {
            format!("{:.2}s", sec)
        } else {
            format!("{:.1}ms", sec * 1000.0)
        }
    }

    /// Format a time value for axis ticks, choosing unit and precision based on
    /// the visible tick spacing so that adjacent ticks show distinct values.
    fn format_axis_time_us(us: f64, tick_spacing_us: f64) -> String {
        if tick_spacing_us < 1.0 {
            // Sub-microsecond spacing: show fractional microseconds
            let sec = us / 1e6;
            let minutes = (sec / 60.0).floor();
            let remainder_s = sec - minutes * 60.0;
            if minutes >= 1.0 {
                format!("{}m{:.6}s", minutes as i64, remainder_s)
            } else {
                format!("{:.6}s", sec)
            }
        } else if tick_spacing_us < 1e3 {
            // Microsecond scale
            let sec = us / 1e6;
            let minutes = (sec / 60.0).floor();
            let remainder_s = sec - minutes * 60.0;
            if minutes >= 1.0 {
                format!("{}m{:.4}s", minutes as i64, remainder_s)
            } else {
                format!("{:.4}s", sec)
            }
        } else if tick_spacing_us < 1e4 {
            // Tens of microseconds
            let sec = us / 1e6;
            let minutes = (sec / 60.0).floor();
            let remainder_s = sec - minutes * 60.0;
            if minutes >= 1.0 {
                format!("{}m{:.3}s", minutes as i64, remainder_s)
            } else {
                format!("{:.3}s", sec)
            }
        } else if tick_spacing_us < 1e5 {
            // Hundreds of microseconds
            let sec = us / 1e6;
            let minutes = (sec / 60.0).floor();
            let remainder_s = sec - minutes * 60.0;
            if minutes >= 1.0 {
                format!("{}m{:.2}s", minutes as i64, remainder_s)
            } else {
                format!("{:.2}s", sec)
            }
        } else if tick_spacing_us < 1e6 {
            // Millisecond scale
            let sec = us / 1e6;
            let minutes = (sec / 60.0).floor();
            let remainder_s = sec - minutes * 60.0;
            if minutes >= 1.0 {
                format!("{}m{:.1}s", minutes as i64, remainder_s)
            } else {
                format!("{:.1}s", sec)
            }
        } else {
            // Seconds or minutes scale — use default formatter
            Self::format_time_us(us)
        }
    }

    fn format_duration_us(us: f64) -> String {
        if us >= 1e6 * 60.0 {
            format!("{:.1}m", us / 1e6 / 60.0)
        } else if us >= 1e6 {
            format!("{:.2}s", us / 1e6)
        } else if us >= 1e3 {
            format!("{:.1}ms", us / 1e3)
        } else {
            format!("{:.0}us", us)
        }
    }

    /// Find the most recent annotation at or before the given time (microseconds).
    /// Binary searches the sorted annotations list.
    fn find_annotation_at(&self, time_us: f64) -> Option<&PairedAnnotation> {
        let anns = &self.layout.annotations;
        if anns.is_empty() {
            return None;
        }
        let idx = anns.partition_point(|a| a.start_us <= time_us);
        if idx == 0 {
            return None;
        }
        let ann = &anns[idx - 1];
        // Don't show if we've gone past the end of a paired annotation
        if let Some(end_us) = ann.end_us {
            if time_us > end_us {
                return None;
            }
        }
        Some(ann)
    }
}

impl eframe::App for MemoryVizApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // FPS tracking
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last_frame_time).as_secs_f64();
        self.last_frame_time = now;
        if dt > 0.0 {
            let fps = 1.0 / dt;
            // Exponential moving average (smoothing factor 0.05)
            self.fps_smooth = if self.fps_smooth == 0.0 {
                fps
            } else {
                self.fps_smooth * 0.95 + fps * 0.05
            };
        }

        // Dark background
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = egui::Color32::from_rgb(26, 26, 46);
        ctx.set_visuals(visuals);

        // Top panel: header and controls
        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.colored_label(
                    egui::Color32::from_rgb(233, 69, 96),
                    egui::RichText::new("CUDA Memory Timeline").strong().size(16.0),
                );
                ui.separator();
                let mut header = format!(
                    "{} events | {} allocs ({} tracked) | Peak: {} | Duration: {}",
                    format_count(self.layout.total_events),
                    format_count(self.layout.total_rects),
                    format_count(self.layout.max_entries.min(self.layout.total_rects)),
                    Self::format_bytes(self.layout.peak_bytes as f64),
                    Self::format_time_us(
                        (self.layout.time_max_us - self.layout.time_min_us) as f64
                    ),
                );
                if let Some(cfg) = &self.model_config {
                    header.push_str(&format!(
                        " | {} (hidden_size H={}, intermediate_size I={})",
                        cfg.model_id, cfg.hidden_size, cfg.intermediate_size
                    ));
                }
                ui.label(header);
            });
            ui.horizontal(|ui| {
                if ui.button("Reset Zoom").clicked() {
                    self.view_x_min_us = self.layout.time_min_us as f64;
                    self.view_x_max_us = self.layout.time_max_us as f64;
                    self.view_y_min_bytes = 0.0;
                    self.view_y_max_bytes = self.layout.peak_bytes as f64 * 1.05;
                    self.invalidate_cache();
                }
                if ui.button("Fit Y").clicked() {
                    self.view_y_min_bytes = 0.0;
                    self.view_y_max_bytes = self.layout.peak_bytes as f64 * 1.05;
                    self.invalidate_cache();
                }
                ui.checkbox(&mut self.show_annotations, "Annotations");
                if self.model_config.is_some() {
                    ui.checkbox(&mut self.exact_shapes_only, "Exact shapes only");
                }
                ui.separator();
                if let Some(cache) = &self.cache {
                    ui.label(format!(
                        "{}x{} texture | {:.0} fps",
                        cache.width_px, cache.height_px, self.fps_smooth
                    ));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!(
                        "View: {} - {} | {} - {}",
                        Self::format_axis_time_us(self.view_x_min_us - self.layout.time_min_us as f64, (self.view_x_max_us - self.view_x_min_us) / 10.0),
                        Self::format_axis_time_us(self.view_x_max_us - self.layout.time_min_us as f64, (self.view_x_max_us - self.view_x_min_us) / 10.0),
                        Self::format_bytes(self.view_y_min_bytes),
                        Self::format_bytes(self.view_y_max_bytes),
                    ));
                });
            });
        });

        // Ctrl+C to copy stack trace (from pinned info, or current hover)
        let copy_source = self.last_hover_info.as_ref().or(self.hover_info.as_ref());
        if let Some(info) = copy_source {
            if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::C)) {
                ctx.copy_text(info.frame_str.clone());
            }
        }

        // Bottom panel: shows pinned allocation if set, otherwise current hover
        let is_pinned = self.last_hover_info.is_some();
        let bottom_info = if is_pinned {
            self.last_hover_info.clone()
        } else {
            self.hover_info.clone()
        };
        let shape_str = bottom_info
            .as_ref()
            .and_then(|info| self.format_tensor_shape(info.size_bytes));
        let time_min_us = self.layout.time_min_us as f64;

        egui::TopBottomPanel::bottom("hover_info")
            .exact_height(100.0)
            .show(ctx, |ui| {
                if let Some(info) = &bottom_info {
                    ui.horizontal_wrapped(|ui| {
                        ui.colored_label(
                            egui::Color32::from_rgb(233, 69, 96),
                            egui::RichText::new(format!(
                                "{} ({} bytes)",
                                Self::format_bytes(info.size_bytes as f64),
                                Self::format_exact_bytes(info.size_bytes),
                            ))
                            .strong(),
                        );
                        ui.label(format!(
                            "| {} - {} | Duration: {}",
                            Self::format_axis_time_us(info.start_us as f64 - time_min_us, (self.view_x_max_us - self.view_x_min_us) / 10.0),
                            Self::format_axis_time_us(info.end_us as f64 - time_min_us, (self.view_x_max_us - self.view_x_min_us) / 10.0),
                            Self::format_duration_us((info.end_us - info.start_us) as f64),
                        ));
                        if let Some(shape) = &shape_str {
                            ui.colored_label(
                                egui::Color32::from_rgb(158, 187, 138),
                                format!("| {}", shape),
                            );
                        }
                        ui.label(format!(
                            "| Before alloc: {} | After alloc: {} | At dealloc: {}",
                            Self::format_bytes(info.total_allocated_bytes as f64),
                            Self::format_bytes((info.total_allocated_bytes + info.size_bytes) as f64),
                            Self::format_bytes(info.total_at_dealloc_bytes as f64),
                        ));
                        if let Some(ann) = self.find_annotation_at(info.start_us as f64) {
                            let ann_label = ann.name.replace("##", "").trim().to_string();
                            ui.colored_label(
                                egui::Color32::from_rgb(230, 190, 100),
                                format!("| Annotation: {}", ann_label),
                            );
                        }
                        if is_pinned {
                            let copy_hint = if cfg!(target_os = "macos") { "Cmd+C" } else { "Ctrl+C" };
                            ui.label(format!("| [pinned] Click empty to unpin | {} to copy", copy_hint));
                        } else {
                            let copy_hint = if cfg!(target_os = "macos") { "Cmd+C" } else { "Ctrl+C" };
                            ui.label(format!("| Click to pin | {} to copy", copy_hint));
                        }
                    });
                    if !info.frame_str.is_empty() {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            for frame in info.frame_str.split(" <- ") {
                                ui.label(
                                    egui::RichText::new(frame)
                                        .small()
                                        .color(egui::Color32::from_rgb(170, 170, 170)),
                                );
                            }
                        });
                    }
                } else {
                    let copy_hint = if cfg!(target_os = "macos") { "Cmd+C" } else { "Ctrl+C" };
                    ui.label(format!("Hover over an allocation for details. Click=pin, Scroll=zoom XY, Shift+Scroll=zoom Y, Alt+Scroll=zoom X, Drag=pan, Cmd+Drag=select region, Double-click=fit Y, Right-click=dismiss tooltip, {}=copy.", copy_hint));
                }
            });

        // Central panel: the chart
        egui::CentralPanel::default().show(ctx, |ui| {
            let available = ui.available_size();
            let margin = 70.0_f32;
            let bottom_margin = 30.0_f32;
            let top_margin = 10.0_f32;
            let right_margin = 20.0_f32;

            let chart_width = (available.x - margin - right_margin).max(100.0);
            let chart_height = (available.y - bottom_margin - top_margin).max(100.0);

            let (response, painter) =
                ui.allocate_painter(available, egui::Sense::click_and_drag());
            let rect = response.rect;

            let chart_rect = egui::Rect::from_min_max(
                egui::pos2(rect.min.x + margin, rect.min.y + top_margin),
                egui::pos2(
                    rect.min.x + margin + chart_width,
                    rect.min.y + top_margin + chart_height,
                ),
            );

            // Background
            painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(26, 26, 46));
            painter.rect_filled(chart_rect, 0.0, egui::Color32::from_rgb(20, 20, 38));

            // Coordinate transforms
            let vx_min = self.view_x_min_us;
            let vx_max = self.view_x_max_us;
            let vy_min = self.view_y_min_bytes;
            let vy_max = self.view_y_max_bytes;
            let x_range = vx_max - vx_min;
            let y_range = vy_max - vy_min;

            let us_to_screen_x = |us: f64| -> f32 {
                chart_rect.min.x + ((us - vx_min) / x_range * chart_width as f64) as f32
            };
            let bytes_to_screen_y = |bytes: f64| -> f32 {
                chart_rect.max.y - ((bytes - vy_min) / y_range * chart_height as f64) as f32
            };
            let screen_x_to_us = |sx: f32| -> f64 {
                vx_min + (sx - chart_rect.min.x) as f64 / chart_width as f64 * x_range
            };
            let screen_y_to_bytes = |sy: f32| -> f64 {
                vy_min + (chart_rect.max.y - sy) as f64 / chart_height as f64 * y_range
            };

            // Grid lines
            let grid_color = egui::Color32::from_rgb(42, 42, 78);
            for i in 0..=8 {
                let frac = i as f32 / 8.0;
                let y = chart_rect.min.y + frac * chart_height;
                painter.line_segment(
                    [
                        egui::pos2(chart_rect.min.x, y),
                        egui::pos2(chart_rect.max.x, y),
                    ],
                    egui::Stroke::new(0.5, grid_color),
                );
                let val = vy_max - frac as f64 * y_range;
                painter.text(
                    egui::pos2(chart_rect.min.x - 6.0, y),
                    egui::Align2::RIGHT_CENTER,
                    Self::format_bytes(val),
                    egui::FontId::monospace(10.0),
                    egui::Color32::from_rgb(136, 136, 136),
                );
            }
            let tick_spacing_us = x_range / 10.0;
            for i in 0..=10 {
                let frac = i as f32 / 10.0;
                let x = chart_rect.min.x + frac * chart_width;
                painter.line_segment(
                    [
                        egui::pos2(x, chart_rect.min.y),
                        egui::pos2(x, chart_rect.max.y),
                    ],
                    egui::Stroke::new(0.5, grid_color),
                );
                let val = vx_min + frac as f64 * x_range;
                let relative = val - self.layout.time_min_us as f64;
                painter.text(
                    egui::pos2(x, chart_rect.max.y + 6.0),
                    egui::Align2::CENTER_TOP,
                    Self::format_axis_time_us(relative, tick_spacing_us),
                    egui::FontId::monospace(10.0),
                    egui::Color32::from_rgb(136, 136, 136),
                );
            }

            // Ensure view cache + texture
            let w_px = chart_width.max(1.0) as usize;
            let h_px = chart_height.max(1.0) as usize;
            self.ensure_cache(ctx, w_px, h_px);

            // Draw chart as a single textured quad
            if let Some(texture) = &self.chart_texture {
                painter.image(
                    texture.id(),
                    chart_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            }

            // Draw annotations with label de-overlapping
            if self.show_annotations {
                let ann_color = egui::Color32::from_rgba_premultiplied(255, 255, 255, 76);
                let ann_text_color = egui::Color32::from_rgba_premultiplied(255, 255, 255, 128);
                let font = egui::FontId::proportional(9.0);
                let row_spacing = 12.0_f32;
                let num_rows = 4_usize;
                let label_padding = 6.0_f32; // horizontal gap between labels

                // First pass: draw vertical lines and collect visible labels
                struct LabelInfo {
                    x: f32,
                    width: f32,
                    label: String,
                    color: egui::Color32,
                }
                let mut labels: Vec<LabelInfo> = Vec::new();

                for ann in &self.layout.annotations {
                    let ann_x = us_to_screen_x(ann.start_us);
                    if ann_x >= chart_rect.min.x && ann_x <= chart_rect.max.x {
                        painter.line_segment(
                            [
                                egui::pos2(ann_x, chart_rect.min.y),
                                egui::pos2(ann_x, chart_rect.max.y),
                            ],
                            egui::Stroke::new(1.0, ann_color),
                        );

                        let label = ann.name.replace("##", "").trim().to_string();
                        let galley = painter.layout_no_wrap(label.clone(), font.clone(), ann_text_color);
                        let width = galley.size().x;
                        labels.push(LabelInfo { x: ann_x, width, label, color: ann_text_color });
                    }

                    if let Some(end_us) = ann.end_us {
                        let end_x = us_to_screen_x(end_us);
                        if end_x >= chart_rect.min.x && end_x <= chart_rect.max.x {
                            painter.line_segment(
                                [
                                    egui::pos2(end_x, chart_rect.min.y),
                                    egui::pos2(end_x, chart_rect.max.y),
                                ],
                                egui::Stroke::new(
                                    1.0,
                                    egui::Color32::from_rgba_premultiplied(255, 255, 255, 40),
                                ),
                            );

                            let end_label = format!(
                                "{} [end]",
                                ann.name.replace("##", "").trim()
                            );
                            let end_text_color = egui::Color32::from_rgba_premultiplied(255, 255, 255, 76);
                            let galley = painter.layout_no_wrap(end_label.clone(), font.clone(), end_text_color);
                            let width = galley.size().x;
                            labels.push(LabelInfo { x: end_x, width, label: end_label, color: end_text_color });
                        }
                    }
                }

                // Sort labels left-to-right (end markers may interleave with starts)
                labels.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());

                // Second pass: assign labels to rows using greedy stagger
                // row_ends[r] = rightmost x extent of labels assigned to row r
                let mut row_ends: Vec<f32> = vec![f32::NEG_INFINITY; num_rows];

                for info in &labels {
                    let label_left = info.x + 3.0;
                    let label_right = label_left + info.width + label_padding;

                    // Find first row where this label doesn't overlap
                    let mut assigned_row = 0;
                    for r in 0..num_rows {
                        if label_left >= row_ends[r] {
                            assigned_row = r;
                            break;
                        }
                        // If no row fits, use the one with the smallest extent
                        if r == num_rows - 1 {
                            assigned_row = row_ends
                                .iter()
                                .enumerate()
                                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                                .map(|(i, _)| i)
                                .unwrap_or(0);
                        }
                    }

                    row_ends[assigned_row] = label_right;
                    let y = chart_rect.min.y + 12.0 + (assigned_row as f32) * row_spacing;

                    painter.text(
                        egui::pos2(info.x + 3.0, y),
                        egui::Align2::LEFT_TOP,
                        &info.label,
                        font.clone(),
                        info.color,
                    );
                }
            }

            // Chart border
            painter.rect_stroke(
                chart_rect,
                0.0,
                egui::Stroke::new(1.0, egui::Color32::from_rgb(15, 52, 96)),
                egui::StrokeKind::Outside,
            );

            // GPU memory capacity line
            if let Some(capacity) = self.gpu_capacity_bytes {
                let cap_y = bytes_to_screen_y(capacity);
                if cap_y >= chart_rect.min.y && cap_y <= chart_rect.max.y {
                    let cap_color = egui::Color32::from_rgba_premultiplied(200, 50, 50, 140);
                    painter.line_segment(
                        [
                            egui::pos2(chart_rect.min.x, cap_y),
                            egui::pos2(chart_rect.max.x, cap_y),
                        ],
                        egui::Stroke::new(1.5, cap_color),
                    );
                    if let Some(ref label) = self.gpu_label {
                        painter.text(
                            egui::pos2(chart_rect.max.x - 4.0, cap_y - 2.0),
                            egui::Align2::RIGHT_BOTTOM,
                            format!("{} max: {}", label, Self::format_bytes(capacity)),
                            egui::FontId::proportional(10.0),
                            cap_color,
                        );
                    }
                }
            }

            // Helper: draw polygon outline for a given rect_idx
            let draw_poly_outline =
                |painter: &egui::Painter, layout: &PolygonLayout, ri: usize, color: egui::Color32, width: f32| {
                    let poly_idx = layout.rect_to_poly[ri];
                    if poly_idx == u32::MAX {
                        return;
                    }
                    let poly = &layout.polygons[poly_idx as usize];
                    let size = layout.rects[ri].size as f64;
                    let n = poly.times_us.len();
                    if n < 2 {
                        return;
                    }
                    let mut points: Vec<egui::Pos2> = Vec::with_capacity(n * 4);
                    for i in 0..n - 1 {
                        let x0 = us_to_screen_x(poly.times_us[i] as f64);
                        let x1 = us_to_screen_x(poly.times_us[i + 1] as f64);
                        let y = bytes_to_screen_y(poly.offsets[i]);
                        points.push(egui::pos2(x0, y));
                        points.push(egui::pos2(x1, y));
                    }
                    for i in (0..n - 1).rev() {
                        let x0 = us_to_screen_x(poly.times_us[i + 1] as f64);
                        let x1 = us_to_screen_x(poly.times_us[i] as f64);
                        let y = bytes_to_screen_y(poly.offsets[i] + size);
                        points.push(egui::pos2(x0, y));
                        points.push(egui::pos2(x1, y));
                    }
                    if let Some(&first) = points.first() {
                        points.push(first);
                    }
                    let clipped = painter.with_clip_rect(chart_rect);
                    clipped.add(egui::Shape::line(points, egui::Stroke::new(width, color)));
                };

            // Draw pinned allocation highlight (dark red)
            if let Some(pinned_idx) = self.pinned_rect_idx {
                draw_poly_outline(
                    &painter,
                    &self.layout,
                    pinned_idx as usize,
                    egui::Color32::from_rgb(200, 50, 50),
                    2.5,
                );
            }

            // ── Interactions ────────────────────────────────────

            // Hover: O(1) lookup via hover_map
            self.hover_info = None;
            if let Some(pos) = response.hover_pos() {
                if chart_rect.contains(pos) {
                    let hover_us = screen_x_to_us(pos.x);
                    let hover_bytes = screen_y_to_bytes(pos.y);

                    // Convert screen position to pixel coordinates in the cache
                    let px = ((pos.x - chart_rect.min.x) / chart_width * w_px as f32) as usize;
                    let py = ((pos.y - chart_rect.min.y) / chart_height * h_px as f32) as usize;

                    if let Some(cache) = &self.cache {
                        if px < w_px && py < h_px {
                            let rect_idx = cache.hover_map[py * w_px + px];

                            // Reset dismiss when hovering a different allocation
                            if self.tooltip_dismissed && rect_idx != self.dismissed_rect_idx {
                                self.tooltip_dismissed = false;
                            }

                            // Right-click to dismiss tooltip
                            if response.secondary_clicked() {
                                self.tooltip_dismissed = true;
                                self.dismissed_rect_idx = rect_idx;
                            }

                            // Click on empty space unpins the bottom bar
                            // Use drag_stopped with small delta as click, since
                            // click_and_drag sense eats clicks as drags
                            let is_click = response.clicked()
                                || (response.drag_stopped()
                                    && response.drag_delta().length() < 3.0);
                            if rect_idx == u32::MAX && is_click {
                                self.last_hover_info = None;
                                self.pinned_rect_idx = None;
                            }

                            if rect_idx != u32::MAX && self.drag_select.is_none() {
                                let ri = rect_idx as usize;
                                let r = &self.layout.rects[ri];

                                // Set hover_info only if tooltip not dismissed
                                if !self.tooltip_dismissed {
                                    let frame_str = self.layout.frame_strings
                                        [r.frame_idx as usize]
                                        .clone();

                                    let (y_offset, total_at_dealloc) = {
                                        let poly_idx = self.layout.rect_to_poly[ri];
                                        if poly_idx != u32::MAX {
                                            let poly =
                                                &self.layout.polygons[poly_idx as usize];
                                            let time_idx = poly
                                                .times_us
                                                .partition_point(|&t| (t as f64) <= hover_us);
                                            let seg_idx = if time_idx > 0 {
                                                time_idx - 1
                                            } else {
                                                0
                                            };
                                            let offset = poly.offsets[seg_idx] as u64;
                                            let last_offset = *poly.offsets.last().unwrap_or(&0.0);
                                            let dealloc_total = (last_offset + r.size as f64) as u64;
                                            (offset, dealloc_total)
                                        } else {
                                            (0, 0)
                                        }
                                    };

                                    let info = HoverInfo {
                                        size_bytes: r.size,
                                        start_us: r.start_us,
                                        end_us: r.end_us,
                                        frame_str,
                                        total_allocated_bytes: y_offset,
                                        total_at_dealloc_bytes: total_at_dealloc,
                                    };
                                    // Pin to bottom bar only on click
                                    // Use is_click (includes drag_stopped fallback)
                                    if is_click {
                                        self.last_hover_info = Some(info.clone());
                                        self.pinned_rect_idx = Some(rect_idx);
                                    }
                                    self.hover_info = Some(info);
                                }

                                // Highlight: always draw hover polygon outline (white)
                                draw_poly_outline(
                                    &painter,
                                    &self.layout,
                                    ri,
                                    egui::Color32::from_rgb(255, 255, 255),
                                    2.0,
                                );
                            }
                        }
                    }

                    // Tooltip (suppressed by right-click dismiss)
                    if let Some(info) = &self.hover_info {
                        egui::show_tooltip_at_pointer(ctx, response.layer_id, egui::Id::new("alloc_tooltip"), |ui| {
                            ui.colored_label(
                                egui::Color32::from_rgb(126, 188, 230),
                                egui::RichText::new(format!(
                                    "Allocation: {} ({} bytes)",
                                    Self::format_bytes(info.size_bytes as f64),
                                    Self::format_exact_bytes(info.size_bytes),
                                ))
                                .strong(),
                            );
                            ui.label(format!(
                                "Duration: {}",
                                Self::format_duration_us((info.end_us - info.start_us) as f64)
                            ));
                            let tick_sp = (self.view_x_max_us - self.view_x_min_us) / 10.0;
                            ui.label(format!(
                                "Time: {} - {}",
                                Self::format_axis_time_us(
                                    info.start_us as f64 - self.layout.time_min_us as f64,
                                    tick_sp,
                                ),
                                Self::format_axis_time_us(
                                    info.end_us as f64 - self.layout.time_min_us as f64,
                                    tick_sp,
                                ),
                            ));
                            if let Some(shape) = self.format_tensor_shape(info.size_bytes) {
                                ui.colored_label(
                                    egui::Color32::from_rgb(158, 187, 138),
                                    format!("Shape: {}", shape),
                                );
                            }
                            ui.label(format!(
                                "Total before allocation: {}",
                                Self::format_bytes(info.total_allocated_bytes as f64)
                            ));
                            ui.label(format!(
                                "Total after allocation: {}",
                                Self::format_bytes((info.total_allocated_bytes + info.size_bytes) as f64)
                            ));
                            if info.total_at_dealloc_bytes > 0 {
                                ui.label(format!(
                                    "Total at deallocation: {}",
                                    Self::format_bytes(info.total_at_dealloc_bytes as f64)
                                ));
                            }
                            if let Some(ann) = self.find_annotation_at(info.start_us as f64) {
                                let ann_label = ann.name.replace("##", "").trim().to_string();
                                ui.colored_label(
                                    egui::Color32::from_rgb(230, 190, 100),
                                    format!("Annotation: {}", ann_label),
                                );
                            }
                        });
                    } else if !self.tooltip_dismissed {
                        let rel_us = hover_us - self.layout.time_min_us as f64;
                        egui::show_tooltip_at_pointer(ctx, response.layer_id, egui::Id::new("cursor_tooltip"), |ui| {
                            ui.label(format!(
                                "t = {} | mem = {}",
                                Self::format_axis_time_us(rel_us, (self.view_x_max_us - self.view_x_min_us) / 10.0),
                                Self::format_bytes(hover_bytes),
                            ));
                            if let Some(ann) = self.find_annotation_at(hover_us) {
                                let ann_label = ann.name.replace("##", "").trim().to_string();
                                ui.colored_label(
                                    egui::Color32::from_rgb(230, 190, 100),
                                    format!("Annotation: {}", ann_label),
                                );
                            }
                        });
                    }
                }
            }

            // Scroll to zoom
            let scroll = ui.input(|i| i.raw_scroll_delta);
            let shift = ui.input(|i| i.modifiers.shift);
            // On macOS, Shift+Scroll is remapped to horizontal scroll, so use scroll.x when shift is held
            let scroll_amount = if shift && scroll.y == 0.0 { scroll.x } else { scroll.y };
            if scroll_amount != 0.0 && response.hovered() {
                if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                    if chart_rect.contains(pos) {
                        let alt = ui.input(|i| i.modifiers.alt);
                        let factor = if scroll_amount > 0.0 { 0.87 } else { 1.15 };

                        // Zoom X axis (unless shift-only)
                        if !shift {
                            let pivot_us = screen_x_to_us(pos.x);
                            let new_min = pivot_us - (pivot_us - self.view_x_min_us) * factor;
                            let new_max = pivot_us + (self.view_x_max_us - pivot_us) * factor;
                            let total_range = (self.layout.time_max_us
                                - self.layout.time_min_us)
                                as f64;
                            let clamped_min = new_min.max(self.layout.time_min_us as f64);
                            let clamped_max = new_max.min(self.layout.time_max_us as f64);
                            if clamped_max - clamped_min > 1.0
                                && clamped_max - clamped_min <= total_range
                            {
                                self.view_x_min_us = clamped_min;
                                self.view_x_max_us = clamped_max;
                            }
                        }

                        // Zoom Y axis (unless alt/option-only)
                        if !alt {
                            let pivot_bytes = screen_y_to_bytes(pos.y);
                            let new_min =
                                pivot_bytes - (pivot_bytes - self.view_y_min_bytes) * factor;
                            let new_max =
                                pivot_bytes + (self.view_y_max_bytes - pivot_bytes) * factor;
                            let full_y_range = self.layout.peak_bytes as f64 * 1.05;
                            let clamped_min = new_min.max(0.0);
                            let clamped_max = new_max.min(full_y_range);
                            if clamped_max - clamped_min > 1000.0
                                && clamped_max - clamped_min <= full_y_range
                            {
                                self.view_y_min_bytes = clamped_min;
                                self.view_y_max_bytes = clamped_max;
                            }
                        }

                        self.invalidate_cache();
                    }
                }
            }

            // Drag: Cmd+drag = region select, plain drag = pan
            let cmd_held = ui.input(|i| i.modifiers.command);

            if response.drag_started() && cmd_held {
                if let Some(pos) = response.interact_pointer_pos() {
                    if chart_rect.contains(pos) {
                        self.drag_select = Some(DragSelect {
                            start_us: screen_x_to_us(pos.x),
                            start_bytes: screen_y_to_bytes(pos.y),
                            start_screen_x: pos.x,
                            start_screen_y: pos.y,
                        });
                    }
                }
            }

            if response.dragged() {
                if self.drag_select.is_some() {
                    // Cmd+drag: draw selection overlay (handled below)
                } else if !cmd_held {
                    // Plain drag: pan
                    let delta = response.drag_delta();
                    let dx_us = -delta.x as f64 / chart_width as f64 * x_range;
                    let dy_bytes = delta.y as f64 / chart_height as f64 * y_range;

                    let new_x_min = (self.view_x_min_us + dx_us)
                        .max(self.layout.time_min_us as f64)
                        .min(self.layout.time_max_us as f64 - x_range);
                    let new_x_max = new_x_min + x_range;

                    let total_range =
                        (self.layout.time_max_us - self.layout.time_min_us) as f64;
                    if new_x_max <= self.layout.time_max_us as f64 + total_range * 0.01
                    {
                        self.view_x_min_us = new_x_min;
                        self.view_x_max_us = new_x_max;
                    }

                    let new_y_min = (self.view_y_min_bytes + dy_bytes).max(0.0);
                    self.view_y_min_bytes = new_y_min;
                    self.view_y_max_bytes = new_y_min + y_range;

                    self.invalidate_cache();
                }
            }

            // Draw selection overlay during cmd+drag
            if let Some(ref sel) = self.drag_select {
                if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                    let x_start = sel.start_screen_x.max(chart_rect.min.x).min(chart_rect.max.x);
                    let x_end = pos.x.max(chart_rect.min.x).min(chart_rect.max.x);
                    let y_start = sel.start_screen_y.max(chart_rect.min.y).min(chart_rect.max.y);
                    let y_end = pos.y.max(chart_rect.min.y).min(chart_rect.max.y);
                    let (x_left, x_right) = if x_start < x_end {
                        (x_start, x_end)
                    } else {
                        (x_end, x_start)
                    };
                    let (y_top, y_bottom) = if y_start < y_end {
                        (y_start, y_end)
                    } else {
                        (y_end, y_start)
                    };
                    let sel_rect = egui::Rect::from_min_max(
                        egui::pos2(x_left, y_top),
                        egui::pos2(x_right, y_bottom),
                    );
                    painter.rect_filled(
                        sel_rect,
                        0.0,
                        egui::Color32::from_rgba_premultiplied(126, 188, 230, 40),
                    );
                    painter.rect_stroke(
                        sel_rect,
                        0.0,
                        egui::Stroke::new(
                            1.5,
                            egui::Color32::from_rgba_premultiplied(126, 188, 230, 150),
                        ),
                        egui::StrokeKind::Outside,
                    );
                }
            }

            // On drag release: if we were selecting, zoom to region
            if response.drag_stopped() {
                if let Some(sel) = self.drag_select.take() {
                    if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                        let end_us = screen_x_to_us(pos.x);
                        let end_bytes = screen_y_to_bytes(pos.y);
                        let (new_x_min, new_x_max) = if sel.start_us < end_us {
                            (sel.start_us, end_us)
                        } else {
                            (end_us, sel.start_us)
                        };
                        let (new_y_min, new_y_max) = if sel.start_bytes < end_bytes {
                            (sel.start_bytes, end_bytes)
                        } else {
                            (end_bytes, sel.start_bytes)
                        };
                        // Only zoom if selection is at least a few pixels in both axes
                        let sel_width_px =
                            ((new_x_max - new_x_min) / x_range * chart_width as f64).abs();
                        let sel_height_px =
                            ((new_y_max - new_y_min) / y_range * chart_height as f64).abs();
                        if sel_width_px > 5.0 && sel_height_px > 5.0 {
                            self.view_x_min_us =
                                new_x_min.max(self.layout.time_min_us as f64);
                            self.view_x_max_us =
                                new_x_max.min(self.layout.time_max_us as f64);
                            self.view_y_min_bytes = new_y_min.max(0.0);
                            self.view_y_max_bytes = new_y_max;
                            self.invalidate_cache();
                        }
                    }
                }
            }

            // Double-click empty space: reset to fully zoomed out
            if response.double_clicked() {
                let on_allocation = response.interact_pointer_pos().map_or(false, |pos| {
                    if !chart_rect.contains(pos) {
                        return false;
                    }
                    let px = ((pos.x - chart_rect.min.x) / chart_width * w_px as f32) as usize;
                    let py = ((pos.y - chart_rect.min.y) / chart_height * h_px as f32) as usize;
                    if let Some(cache) = &self.cache {
                        px < w_px && py < h_px && cache.hover_map[py * w_px + px] != u32::MAX
                    } else {
                        false
                    }
                });
                if !on_allocation {
                    self.view_x_min_us = self.layout.time_min_us as f64;
                    self.view_x_max_us = self.layout.time_max_us as f64;
                    self.view_y_min_bytes = 0.0;
                    self.view_y_max_bytes = self.layout.peak_bytes as f64 * 1.05;
                    self.invalidate_cache();
                }
            }
        });

        // Request repaint when interacting
        if self.cache.is_none() {
            ctx.request_repaint();
        }
    }
}

fn format_count(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}

// ── Main ────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();
    let start = Instant::now();

    // Determine if we need to convert pickle to JSON
    let json_path = match cli.input.extension().and_then(|e| e.to_str()) {
        Some("json") => cli.input.clone(),
        Some("pickle") | Some("pkl") => convert_pickle_to_json(&cli.input)?,
        _ => {
            let first_bytes = fs::read(&cli.input)
                .with_context(|| format!("failed to read {}", cli.input.display()))?;
            if first_bytes.starts_with(b"{") {
                cli.input.clone()
            } else {
                convert_pickle_to_json(&cli.input)?
            }
        }
    };

    // Read and parse JSON
    eprintln!("Reading JSON from {}...", json_path.display());
    let t_read = Instant::now();
    let json_bytes =
        fs::read(&json_path).with_context(|| format!("failed to read {}", json_path.display()))?;
    eprintln!(
        "  Read {} MB in {:.1}s",
        json_bytes.len() / 1_000_000,
        t_read.elapsed().as_secs_f64()
    );

    eprintln!("Parsing JSON...");
    let t_parse = Instant::now();
    let snapshot: SnapshotJson =
        serde_json::from_slice(&json_bytes).context("failed to parse JSON")?;
    eprintln!("  Parsed in {:.1}s", t_parse.elapsed().as_secs_f64());

    let total_events = snapshot.events.len();
    eprintln!(
        "  {} events, {} annotations, {} frame strings",
        total_events,
        snapshot.annotations.len(),
        snapshot.frame_strings.len()
    );

    // Sort events by time
    let mut events = snapshot.events;
    events.sort_by_key(|e| e.3);

    let time_min = events.first().map(|e| e.3).unwrap_or(0);
    let time_max = events.last().map(|e| e.3).unwrap_or(0);

    // Pair alloc/free events
    eprintln!("Pairing alloc/free events...");
    let t_pair = Instant::now();
    let rects = pair_alloc_free(&events, time_max);
    eprintln!(
        "  {} allocation rectangles in {:.1}s",
        rects.len(),
        t_pair.elapsed().as_secs_f64()
    );

    // Pair annotations
    let annotations = pair_annotations(
        &snapshot.annotations,
        &cli.annotation_filter,
        cli.all_annotations,
    );
    eprintln!("  {} paired annotations", annotations.len());

    // Build polygon layout (port of PyTorch's process_alloc_data)
    eprintln!(
        "Building polygon layout (max_entries={})...",
        cli.max_entries
    );
    let t_layout = Instant::now();
    let layout = build_polygon_layout(
        rects,
        snapshot.frame_strings,
        annotations,
        time_min,
        time_max,
        total_events,
        cli.max_entries,
    );
    eprintln!(
        "  Layout built in {:.1}s",
        t_layout.elapsed().as_secs_f64()
    );

    eprintln!(
        "Total data loading: {:.1}s. Launching GUI...",
        start.elapsed().as_secs_f64()
    );

    // Launch the egui window
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title(format!(
                "CUDA Memory Timeline — {}",
                cli.input.file_name().map(|f| f.to_string_lossy()).unwrap_or_default()
            )),
        ..Default::default()
    };

    // Fetch model config from HuggingFace if --model is provided
    let model_config = match &cli.model {
        Some(model_id) => match fetch_model_config(model_id) {
            Ok(mut cfg) => {
                if let Some(v) = cli.vocab_size {
                    eprintln!("  Overriding vocab_size: {} -> {}", cfg.vocab_size, v);
                    cfg.vocab_size = v;
                }
                Some(cfg)
            }
            Err(e) => {
                eprintln!("Warning: failed to fetch model config: {}", e);
                None
            }
        },
        None => None,
    };

    let app = MemoryVizApp::new(layout, model_config, cli.quantized, cli.gpu);

    eframe::run_native(
        "desktop-memory-viz",
        native_options,
        Box::new(|_cc| Ok(Box::new(app))),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {}", e))?;

    Ok(())
}
