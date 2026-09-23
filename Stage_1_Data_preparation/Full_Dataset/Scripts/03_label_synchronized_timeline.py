#!/usr/bin/env python3
"""
03_label_synchronized_timeline.py

Map fall annotations onto the synchronized timeline segments.
Produce labeled windows:
  - FALL windows: ±3 seconds around each annotated fall timestamp
  - NON_FALL windows: 6-second windows from remaining timeline (stride = 3s)

Each window is defined in time, not in frame count.

Output:
    Labeled_Timeline/
      windows.csv          - All window definitions
      fall_coverage.csv    - Report on which falls are covered/uncovered
      labeling_config.json - Parameters used

Usage:
    python 03_label_synchronized_timeline.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RAW_DIR = PROJECT_DIR / "Raw_data"
SYNC_DIR = PROJECT_DIR / "Synchronized"
LABEL_DIR = PROJECT_DIR / "Labeled_Timeline"

SUBJECTS = list(range(1, 11))

# Window parameters
FALL_BEFORE_S = 3.0
FALL_AFTER_S = 3.0
WINDOW_DURATION_S = FALL_BEFORE_S + FALL_AFTER_S  # 6.0 seconds
NON_FALL_STRIDE_S = 3.0

# Minimum frames required in a window to be valid
MIN_FRAMES_PER_WINDOW = 4

# Excluded fall annotations: set of (subject_id, fall_number) tuples.
# These are loaded from falls.csv but NOT treated as valid FALL windows.
# Subject 3 / Fall 2 was previously identified as an invalid fall.
EXCLUDED_FALLS = {
    (3, 2),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_falls():
    """Load falls.csv and return a dict: subject_id -> list of fall timestamps."""
    falls_path = RAW_DIR / "falls.csv"
    if not falls_path.exists():
        # Try alternatives
        for name in ["Falls.csv", "FALLS.csv"]:
            alt = RAW_DIR / name
            if alt.exists():
                falls_path = alt
                break

    if not falls_path.exists():
        raise FileNotFoundError(f"Falls file not found in {RAW_DIR}")

    df = pd.read_csv(falls_path, header=None)
    falls_dict = {}
    for col_idx in range(df.shape[1]):
        subj_id = col_idx + 1
        vals = df.iloc[:, col_idx].dropna().values.astype(np.float64)
        falls_dict[subj_id] = list(vals)
    return falls_dict


def load_segments(subj_id):
    """Load all synchronized segments for a subject.
    Returns list of (segment_id, timestamps, n_frames, filepath)."""
    subj_dir = SYNC_DIR / f"subject_{subj_id}"
    if not subj_dir.exists():
        return []

    segments = []
    seg_files = sorted(subj_dir.glob("segment_*.csv"))
    for seg_file in seg_files:
        seg_id = int(seg_file.stem.split("_")[1])
        data = np.loadtxt(seg_file, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        ts = data[:, 0]
        segments.append((seg_id, ts, len(ts), seg_file))
    return segments


def intervals_overlap(a_start, a_end, b_start, b_end):
    """Check if two intervals overlap."""
    return a_start < b_end and b_start < a_end


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LABELING SYNCHRONIZED TIMELINE")
    print("=" * 60)

    # --- Debug: confirm actual configuration ---
    print(f"\nConfiguration:")
    print(f"  FALL_BEFORE_S:    {FALL_BEFORE_S}")
    print(f"  FALL_AFTER_S:     {FALL_AFTER_S}")
    print(f"  WINDOW_DURATION:  {WINDOW_DURATION_S} s")
    print(f"  NON_FALL stride:  {NON_FALL_STRIDE_S} s")
    print(f"  EXCLUDED_FALLS:   {EXCLUDED_FALLS}")

    # Load fall annotations
    falls_dict = load_falls()
    total_annotations = sum(len(v) for v in falls_dict.values())
    print(f"\nTotal fall annotations loaded: {total_annotations}")

    all_windows = []
    all_fall_coverage = []
    sample_counter = 0

    for subj in SUBJECTS:
        print(f"\n--- Subject {subj} ---")

        segments = load_segments(subj)
        if not segments:
            print(f"  No synchronized segments found")
            # Mark all falls as uncovered
            subj_falls = falls_dict.get(subj, [])
            for fall_num, fall_ts in enumerate(subj_falls, 1):
                all_fall_coverage.append({
                    "subject_id": subj,
                    "fall_number": fall_num,
                    "fall_timestamp": fall_ts,
                    "segment_id": None,
                    "covered": False,
                    "reason": "no_synchronized_segments",
                })
            continue

        print(f"  Segments: {len(segments)}")
        for seg_id, ts, n_frames, _ in segments:
            print(f"    Segment {seg_id}: {n_frames} frames, "
                  f"[{ts[0]:.2f}, {ts[-1]:.2f}], dur={ts[-1]-ts[0]:.2f}s")

        # ----- Process FALL windows -----
        subj_falls = falls_dict.get(subj, [])
        print(f"  Annotated falls: {len(subj_falls)}")

        fall_windows = []  # (window_start, window_end, fall_num, fall_ts, seg_id)

        for fall_num, fall_ts in enumerate(subj_falls, 1):
            # Check exclusion list
            if (subj, fall_num) in EXCLUDED_FALLS:
                all_fall_coverage.append({
                    "subject_id": subj,
                    "fall_number": fall_num,
                    "fall_timestamp": fall_ts,
                    "segment_id": None,
                    "covered": False,
                    "reason": "previously_removed_invalid_fall",
                })
                print(f"  *** Fall {fall_num} (ts={fall_ts:.2f}) EXCLUDED: previously_removed_invalid_fall")
                continue

            w_start = fall_ts - FALL_BEFORE_S
            w_end = fall_ts + FALL_AFTER_S

            # Find which segment contains this window
            found = False
            for seg_id, ts, n_frames, seg_file in segments:
                seg_start = ts[0]
                seg_end = ts[-1]

                # The entire window must fit within the segment
                if w_start >= seg_start and w_end <= seg_end:
                    fall_windows.append((w_start, w_end, fall_num, fall_ts, seg_id))
                    all_fall_coverage.append({
                        "subject_id": subj,
                        "fall_number": fall_num,
                        "fall_timestamp": fall_ts,
                        "segment_id": seg_id,
                        "covered": True,
                        "reason": "",
                    })
                    found = True
                    break

            if not found:
                # Determine reason
                reason = "window_exceeds_segment_boundaries"
                # Check if the fall timestamp is in any segment at all
                in_any = False
                for seg_id, ts, n_frames, _ in segments:
                    if ts[0] <= fall_ts <= ts[-1]:
                        in_any = True
                        break
                if not in_any:
                    reason = "fall_timestamp_not_in_any_segment"

                all_fall_coverage.append({
                    "subject_id": subj,
                    "fall_number": fall_num,
                    "fall_timestamp": fall_ts,
                    "segment_id": None,
                    "covered": False,
                    "reason": reason,
                })
                print(f"  *** Fall {fall_num} (ts={fall_ts:.2f}) NOT COVERED: {reason}")

        print(f"  Covered falls: {len(fall_windows)}")

        # Record FALL windows
        for w_start, w_end, fall_num, fall_ts, seg_id in fall_windows:
            all_windows.append({
                "sample_id": sample_counter,
                "label": "FALL",
                "subject_id": subj,
                "fall_number": fall_num,
                "fall_id": f"S{subj}_F{fall_num}",
                "segment_id": seg_id,
                "center_time": fall_ts,
                "window_start": w_start,
                "window_end": w_end,
            })
            sample_counter += 1

        # ----- Process NON_FALL windows -----
        # For each segment, slide a 6s window with the configured stride.
        # Only valid FALL windows (not excluded annotations) block NON_FALL extraction.

        non_fall_count = 0
        non_fall_candidates = 0  # total candidates considered (before overlap filtering)
        for seg_id, ts, n_frames, seg_file in segments:
            seg_start = ts[0]
            seg_end = ts[-1]

            # Collect FALL intervals in this segment
            seg_fall_intervals = []
            for w_start, w_end, fall_num, fall_ts, fw_seg_id in fall_windows:
                if fw_seg_id == seg_id:
                    seg_fall_intervals.append((w_start, w_end))

            # Slide non-fall windows
            cursor = seg_start
            while cursor + WINDOW_DURATION_S <= seg_end:
                nf_start = cursor
                nf_end = cursor + WINDOW_DURATION_S
                nf_center = (nf_start + nf_end) / 2.0

                non_fall_candidates += 1

                # Check overlap with any FALL window
                overlaps_fall = False
                for f_start, f_end in seg_fall_intervals:
                    if intervals_overlap(nf_start, nf_end, f_start, f_end):
                        overlaps_fall = True
                        break

                if not overlaps_fall:
                    all_windows.append({
                        "sample_id": sample_counter,
                        "label": "NON_FALL",
                        "subject_id": subj,
                        "fall_number": "",
                        "fall_id": "",
                        "segment_id": seg_id,
                        "center_time": nf_center,
                        "window_start": nf_start,
                        "window_end": nf_end,
                    })
                    sample_counter += 1
                    non_fall_count += 1

                cursor += NON_FALL_STRIDE_S

        print(f"  NON_FALL: {non_fall_candidates} candidates (stride={NON_FALL_STRIDE_S}s) -> {non_fall_count} accepted")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    windows_df = pd.DataFrame(all_windows)
    windows_path = LABEL_DIR / "windows.csv"
    windows_df.to_csv(windows_path, index=False)
    print(f"\nWindows saved: {windows_path} ({len(windows_df)} windows)")

    coverage_df = pd.DataFrame(all_fall_coverage)
    coverage_path = LABEL_DIR / "fall_coverage.csv"
    coverage_df.to_csv(coverage_path, index=False)
    print(f"Fall coverage saved: {coverage_path}")

    config = {
        "fall_before_seconds": FALL_BEFORE_S,
        "fall_after_seconds": FALL_AFTER_S,
        "window_duration_seconds": WINDOW_DURATION_S,
        "non_fall_stride_seconds": NON_FALL_STRIDE_S,
        "min_frames_per_window": MIN_FRAMES_PER_WINDOW,
    }
    config_path = LABEL_DIR / "labeling_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("LABELING SUMMARY")
    print(f"{'='*60}")

    n_fall = len(windows_df[windows_df["label"] == "FALL"])
    n_nf = len(windows_df[windows_df["label"] == "NON_FALL"])
    print(f"  FALL windows:     {n_fall}")
    print(f"  NON_FALL windows: {n_nf}")
    print(f"  Total windows:    {len(windows_df)}")

    n_covered = len(coverage_df[coverage_df["covered"] == True])
    n_uncovered = len(coverage_df[coverage_df["covered"] == False])
    print(f"\n  Annotated falls:  {len(coverage_df)}")
    print(f"  Covered falls:    {n_covered}")
    print(f"  Uncovered falls:  {n_uncovered}")

    if n_uncovered > 0:
        print("\n  Uncovered falls detail:")
        uncovered = coverage_df[coverage_df["covered"] == False]
        for _, row in uncovered.iterrows():
            print(f"    Subject {row['subject_id']}, Fall {row['fall_number']}: {row['reason']}")

    # Per-subject breakdown
    print(f"\n  Per-subject distribution:")
    for subj in SUBJECTS:
        sf = windows_df[windows_df["subject_id"] == subj]
        n_f = len(sf[sf["label"] == "FALL"])
        n_nf_s = len(sf[sf["label"] == "NON_FALL"])
        print(f"    Subject {subj}: {n_f} FALL, {n_nf_s} NON_FALL")

    print("\nDone.")


if __name__ == "__main__":
    main()
