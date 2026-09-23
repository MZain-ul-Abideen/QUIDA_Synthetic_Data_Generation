#!/usr/bin/env python3
"""
02_synchronize_dataset.py

Synchronize IMU (accelerometer) data onto the Radar timeline for each subject.

- Radar timestamps are the reference timeline (lower frequency, already padded).
- IMU values are linearly interpolated onto the Radar timestamps.
- Radar values are preserved exactly as provided.
- Large temporal gaps split the data into continuous segments.
- Each segment is saved separately.

Output per subject:
    Synchronized/subject_{id}/segment_{k}.csv

Each synchronized row contains:
    timestamp, IMU_x, IMU_y, IMU_z, Radar_f1, ..., Radar_f1134
    = 1 + 3 + 1134 = 1138 columns

Usage:
    python 02_synchronize_dataset.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RAW_DIR = PROJECT_DIR / "Raw_data"
SYNC_DIR = PROJECT_DIR / "Synchronized"

SUBJECTS = list(range(1, 11))

# Gap threshold multiplier: gaps larger than this × median_interval
# are treated as recording discontinuities.
GAP_MULTIPLIER = 10.0

# Minimum segment duration (seconds) to keep
MIN_SEGMENT_DURATION = 1.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    SYNC_DIR.mkdir(parents=True, exist_ok=True)

    sync_report = []

    for subj in SUBJECTS:
        print(f"\n{'='*60}")
        print(f"SUBJECT {subj}")
        print(f"{'='*60}")

        imu_path = RAW_DIR / f"subject_{subj}_accelerometer.csv"
        radar_path = RAW_DIR / f"subject_{subj}_radar.csv"

        if not imu_path.exists():
            print(f"  ERROR: IMU file missing: {imu_path.name}")
            continue
        if not radar_path.exists():
            print(f"  ERROR: Radar file missing: {radar_path.name}")
            continue

        # Load data
        imu_df = pd.read_csv(imu_path, header=None)
        radar_df = pd.read_csv(radar_path, header=None)

        # Validate dimensions
        imu_n_cols = imu_df.shape[1]
        radar_n_cols = radar_df.shape[1]
        print(f"  IMU shape: {imu_df.shape} ({imu_n_cols - 1} features)")
        print(f"  Radar shape: {radar_df.shape} ({radar_n_cols - 1} features)")

        if imu_n_cols < 4:
            print(f"  ERROR: IMU has only {imu_n_cols} columns, expected >= 4")
            continue

        imu_feature_count = imu_n_cols - 1  # excluding timestamp
        radar_feature_count = radar_n_cols - 1

        # Extract arrays
        imu_ts = imu_df.iloc[:, 0].values.astype(np.float64)
        imu_features = imu_df.iloc[:, 1:4].values.astype(np.float64)  # take first 3 features only

        radar_ts = radar_df.iloc[:, 0].values.astype(np.float64)
        radar_features = radar_df.iloc[:, 1:].values.astype(np.float64)

        actual_radar_features = radar_features.shape[1]
        print(f"  Using 3 IMU features, {actual_radar_features} Radar features")
        print(f"  Expected total features: {3 + actual_radar_features}")

        # Sort by timestamp (safety)
        imu_order = np.argsort(imu_ts)
        imu_ts = imu_ts[imu_order]
        imu_features = imu_features[imu_order]

        radar_order = np.argsort(radar_ts)
        radar_ts = radar_ts[radar_order]
        radar_features = radar_features[radar_order]

        # Remove duplicate timestamps in radar
        _, unique_idx = np.unique(radar_ts, return_index=True)
        if len(unique_idx) < len(radar_ts):
            n_dup = len(radar_ts) - len(unique_idx)
            print(f"  Removing {n_dup} duplicate radar timestamps")
            radar_ts = radar_ts[unique_idx]
            radar_features = radar_features[unique_idx]

        # Remove duplicate timestamps in IMU
        _, unique_idx_imu = np.unique(imu_ts, return_index=True)
        if len(unique_idx_imu) < len(imu_ts):
            n_dup = len(imu_ts) - len(unique_idx_imu)
            print(f"  Removing {n_dup} duplicate IMU timestamps")
            imu_ts = imu_ts[unique_idx_imu]
            imu_features = imu_features[unique_idx_imu]

        # Determine gap threshold from radar intervals
        radar_diffs = np.diff(radar_ts)
        median_radar_dt = np.median(radar_diffs)
        gap_threshold = GAP_MULTIPLIER * median_radar_dt
        print(f"  Radar median interval: {median_radar_dt*1000:.2f} ms")
        print(f"  Gap threshold ({GAP_MULTIPLIER}× median): {gap_threshold:.4f} s")

        # Split radar into continuous segments
        gap_indices = np.where(radar_diffs > gap_threshold)[0]
        print(f"  Large gaps found: {len(gap_indices)}")

        # Build segment boundaries
        segment_starts = [0] + (gap_indices + 1).tolist()
        segment_ends = gap_indices.tolist() + [len(radar_ts) - 1]

        # Create output directory
        subj_dir = SYNC_DIR / f"subject_{subj}"
        subj_dir.mkdir(parents=True, exist_ok=True)

        # Determine temporal overlap between IMU and Radar
        imu_t_min, imu_t_max = imu_ts[0], imu_ts[-1]
        print(f"  IMU time range: [{imu_t_min:.2f}, {imu_t_max:.2f}]")
        print(f"  Radar time range: [{radar_ts[0]:.2f}, {radar_ts[-1]:.2f}]")

        # Build IMU interpolation functions (only within IMU range)
        imu_interp_funcs = []
        for feat_idx in range(3):
            f = interp1d(
                imu_ts, imu_features[:, feat_idx],
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            imu_interp_funcs.append(f)

        segment_count = 0
        total_synced_frames = 0

        for seg_idx, (s_start, s_end) in enumerate(zip(segment_starts, segment_ends)):
            seg_radar_ts = radar_ts[s_start: s_end + 1]
            seg_radar_features = radar_features[s_start: s_end + 1]

            seg_duration = seg_radar_ts[-1] - seg_radar_ts[0]
            if seg_duration < MIN_SEGMENT_DURATION:
                print(f"  Segment {seg_idx}: duration {seg_duration:.3f}s < {MIN_SEGMENT_DURATION}s, skipping")
                continue

            # Interpolate IMU onto radar timestamps
            imu_interped = np.column_stack([
                f(seg_radar_ts) for f in imu_interp_funcs
            ])

            # Check how many radar timestamps fall within IMU range
            in_imu_range = (seg_radar_ts >= imu_t_min) & (seg_radar_ts <= imu_t_max)
            n_in_range = int(in_imu_range.sum())
            n_total = len(seg_radar_ts)
            n_nan_imu = int(np.isnan(imu_interped).any(axis=1).sum())

            if n_nan_imu == n_total:
                print(f"  Segment {seg_idx}: no IMU coverage, skipping")
                continue

            # Only keep rows where IMU interpolation succeeded
            valid_mask = ~np.isnan(imu_interped).any(axis=1)
            seg_radar_ts_valid = seg_radar_ts[valid_mask]
            seg_radar_features_valid = seg_radar_features[valid_mask]
            imu_interped_valid = imu_interped[valid_mask]

            if len(seg_radar_ts_valid) < 2:
                print(f"  Segment {seg_idx}: fewer than 2 valid frames after IMU interpolation, skipping")
                continue

            # Re-check for gaps in the valid segment after removing NaN rows
            valid_diffs = np.diff(seg_radar_ts_valid)
            sub_gaps = np.where(valid_diffs > gap_threshold)[0]

            if len(sub_gaps) > 0:
                # Split further
                sub_starts = [0] + (sub_gaps + 1).tolist()
                sub_ends = sub_gaps.tolist() + [len(seg_radar_ts_valid) - 1]
            else:
                sub_starts = [0]
                sub_ends = [len(seg_radar_ts_valid) - 1]

            for ss, se in zip(sub_starts, sub_ends):
                sub_ts = seg_radar_ts_valid[ss: se + 1]
                sub_imu = imu_interped_valid[ss: se + 1]
                sub_radar = seg_radar_features_valid[ss: se + 1]

                sub_dur = sub_ts[-1] - sub_ts[0]
                if sub_dur < MIN_SEGMENT_DURATION or len(sub_ts) < 2:
                    continue

                # Assemble: timestamp | IMU_x | IMU_y | IMU_z | Radar_1 .. Radar_N
                synced = np.column_stack([
                    sub_ts.reshape(-1, 1),
                    sub_imu,
                    sub_radar,
                ])

                expected_cols = 1 + 3 + actual_radar_features
                assert synced.shape[1] == expected_cols, \
                    f"Column mismatch: {synced.shape[1]} != {expected_cols}"

                # Verify no NaN/Inf
                assert np.isnan(synced).sum() == 0, "NaN found in synchronized data!"
                assert np.isinf(synced).sum() == 0, "Inf found in synchronized data!"

                # Verify timestamps strictly increasing
                ts_diffs_check = np.diff(synced[:, 0])
                assert np.all(ts_diffs_check > 0), "Timestamps not strictly increasing!"

                # Save
                out_path = subj_dir / f"segment_{segment_count:03d}.csv"
                np.savetxt(out_path, synced, delimiter=",", fmt="%.10f")

                sync_report.append({
                    "subject": subj,
                    "segment": segment_count,
                    "start_time": float(sub_ts[0]),
                    "end_time": float(sub_ts[-1]),
                    "duration_s": float(sub_dur),
                    "n_frames": len(sub_ts),
                    "n_columns": synced.shape[1],
                })

                total_synced_frames += len(sub_ts)
                segment_count += 1

        print(f"\n  Subject {subj}: {segment_count} segments, {total_synced_frames} total frames")

    # -----------------------------------------------------------------------
    # Save report
    # -----------------------------------------------------------------------
    report_df = pd.DataFrame(sync_report)
    report_path = SYNC_DIR / "synchronization_report.csv"
    report_df.to_csv(report_path, index=False)
    print(f"\nSynchronization report saved to: {report_path}")

    # Also save config
    config = {
        "gap_multiplier": GAP_MULTIPLIER,
        "min_segment_duration_s": MIN_SEGMENT_DURATION,
        "imu_features_used": 3,
        "radar_features_used": actual_radar_features if 'actual_radar_features' in dir() else "unknown",
    }
    config_path = SYNC_DIR / "sync_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("SYNCHRONIZATION SUMMARY")
    print(f"{'='*60}")
    if len(report_df) > 0:
        for subj in SUBJECTS:
            subj_rows = report_df[report_df["subject"] == subj]
            if len(subj_rows) == 0:
                print(f"  Subject {subj}: NO segments")
            else:
                total_frames = subj_rows["n_frames"].sum()
                total_dur = subj_rows["duration_s"].sum()
                n_segs = len(subj_rows)
                print(f"  Subject {subj}: {n_segs} segments, {total_frames} frames, {total_dur:.1f}s total")
        print(f"\n  Total segments: {len(report_df)}")
        print(f"  Total synchronized frames: {report_df['n_frames'].sum()}")
    else:
        print("  No segments produced!")

    print("\nDone.")


if __name__ == "__main__":
    main()
