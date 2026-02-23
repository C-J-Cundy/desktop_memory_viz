#!/usr/bin/env python3
"""Extract PyTorch memory snapshot data to a compact JSON format for the Rust visualizer.

Usage:
    python extract_snapshot.py <input.pickle> <output.json>

The output JSON has the structure:
{
    "events": [[action_code, addr, size, time_us, frame_idx], ...],
    "frame_strings": ["frame summary 0", "frame summary 1", ...],
    "annotations": [{"stage": "START"|"END", "name": "...", "time_us": 123}, ...],
}

Action codes: 0=alloc, 1=free_requested, 2=free_completed, 3=segment_alloc, 4=segment_free, 5=segment_map, 6=segment_unmap

Events include addr for pairing alloc/free events, and frame_idx indexes into
the deduplicated frame_strings array.
"""

import gzip
import json
import pickle
import sys
import time


ACTION_CODES = {
    "alloc": 0,
    "free_requested": 1,
    "free_completed": 2,
    "segment_alloc": 3,
    "segment_free": 4,
    "segment_map": 5,
    "segment_unmap": 6,
}


def frame_summary(frames, max_frames=5):
    if not frames:
        return ""
    parts = []
    for f in frames[:max_frames]:
        filename = f.get("filename", "")
        short = filename.rsplit("/", 1)[-1] if "/" in filename else filename
        parts.append(f"{short}:{f.get('line', 0)} ({f.get('name', '')})")
    return " <- ".join(parts)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.pickle> <output.json>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    t0 = time.time()
    print(f"Loading {input_path}...", file=sys.stderr)
    opener = gzip.open if input_path.endswith(".gz") else open
    with opener(input_path, "rb") as f:
        data = pickle.load(f)
    print(f"  Loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    # Extract events as compact arrays [action_code, addr, size, time_us, frame_idx]
    # Frame strings are interned (deduplicated) to save space.
    t1 = time.time()
    events = []
    frame_intern = {}  # frame_string -> index
    frame_strings = []

    for trace in data.get("device_traces", []):
        for ev in trace:
            action = ev.get("action", "")
            code = ACTION_CODES.get(action)
            if code is None:
                continue
            addr = ev.get("addr", 0)
            size = ev.get("size", 0)
            time_us = ev.get("time_us", 0)

            fs = frame_summary(ev.get("frames", []))
            fidx = frame_intern.get(fs)
            if fidx is None:
                fidx = len(frame_strings)
                frame_intern[fs] = fidx
                frame_strings.append(fs)

            events.append([code, addr, size, time_us, fidx])
    print(f"  {len(events)} events extracted in {time.time() - t1:.1f}s", file=sys.stderr)
    print(f"  {len(frame_strings)} unique frame summaries", file=sys.stderr)

    # Extract annotations
    annotations = []
    for ann in data.get("external_annotations", []):
        annotations.append({
            "stage": ann.get("stage", ""),
            "name": ann.get("name", ""),
            "time_us": ann.get("time_us", 0),
        })
    print(f"  {len(annotations)} annotations", file=sys.stderr)

    result = {
        "events": events,
        "frame_strings": frame_strings,
        "annotations": annotations,
    }

    t2 = time.time()
    print(f"Writing {output_path}...", file=sys.stderr)
    with open(output_path, "w") as f:
        json.dump(result, f)
    print(f"  Written in {time.time() - t2:.1f}s", file=sys.stderr)
    print(f"Total: {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
