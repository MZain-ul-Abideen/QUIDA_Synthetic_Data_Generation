#!/usr/bin/env python3
"""
combine_classification_datasets.py

Build classification-ready datasets for:
  02: Real ADLs + Generated FALLS
  03: Real ADLs + Generated ADLs

Important:
- The source NPY/CSV files are never modified.
- Array row order is preserved exactly during concatenation.
- Synthetic sample-level labels are taken from the synthetic metadata when
  available. No FALL/NON_FALL ordering is guessed.
- Metadata is normalized into one consistent schema for classification,
  provenance tracking, and later train/validation/test analysis.

Usage:
    python combine_classification_datasets.py
    python combine_classification_datasets.py --base-dir PATH --out-dir PATH
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_MAP = {
    "NON_FALL": 0,
    "FALL": 1,
}

# Common schema. Fields that do not exist for a particular sample remain NaN.
METADATA_COLUMNS = [
    "sample_id",
    "label",
    "class_id",
    "activity_type",
    "source",
    "generation_type",
    "generation_class",
    "source_sample_id",
    "subject_id",
    "fall_id",
    "fall_number",
    "segment_id",
    "center_time",
    "window_start",
    "window_end",
    "original_frame_count",
    "target_frames",
    "feature_count",
    "multiplier",
]


# ---------------------------------------------------------------------------
# I/O and validation helpers
# ---------------------------------------------------------------------------

def load_npy(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing required file: {path}")

    arr = np.load(path)

    if arr.ndim != 3:
        raise ValueError(
            f"Expected a 3-D array at '{path}', got shape {arr.shape}."
        )

    return arr


def load_csv(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing required file: {path}")

    return pd.read_csv(path)


def check_no_nan_inf(arr, name):
    if not np.isfinite(arr).all():
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        raise ValueError(
            f"Validation failed for '{name}': "
            f"{n_nan} NaN and {n_inf} Inf values found."
        )


def normalize_label(value):
    """Normalize labels without silently inventing new classes."""
    if pd.isna(value):
        return np.nan

    value = str(value).strip().upper()

    aliases = {
        "NONFALL": "NON_FALL",
        "NON-FALL": "NON_FALL",
        "NON FALL": "NON_FALL",
    }

    value = aliases.get(value, value)

    if value not in CLASS_MAP:
        raise ValueError(
            f"Unsupported label '{value}'. Expected one of: "
            f"{list(CLASS_MAP)}."
        )

    return value


def add_class_columns(df):
    """Add normalized label/class/activity fields."""
    df = df.copy()

    if "label" not in df.columns:
        raise ValueError("Metadata must contain a 'label' column.")

    df["label"] = df["label"].map(normalize_label)

    if df["label"].isna().any():
        n_missing = int(df["label"].isna().sum())
        raise ValueError(
            f"Found {n_missing} metadata rows with missing labels."
        )

    df["class_id"] = df["label"].map(CLASS_MAP).astype(np.int8)

    # Only the binary activity class is supported by the supplied metadata.
    # Do not invent specific ADL names that are not present in the source.
    df["activity_type"] = df["label"]

    return df


def validate_metadata_array_alignment(X, meta, name):
    if X.shape[0] != len(meta):
        raise ValueError(
            f"{name}: array has {X.shape[0]} samples but metadata has "
            f"{len(meta)} rows."
        )

    check_no_nan_inf(X, name)


def validate_shape_compatibility(X_real, X_synth, name):
    if X_real.shape[1:] != X_synth.shape[1:]:
        raise ValueError(
            f"{name}: incompatible sample shapes: "
            f"real={X_real.shape[1:]}, synthetic={X_synth.shape[1:]}"
        )


def validate_sample_ids(meta):
    if meta["sample_id"].duplicated().any():
        duplicates = meta.loc[
            meta["sample_id"].duplicated(keep=False), "sample_id"
        ].tolist()
        raise ValueError(
            f"Duplicate sample_id values found: {duplicates[:10]}"
        )


def assign_new_sample_ids(meta, start_id):
    meta = meta.copy()
    meta["sample_id"] = np.arange(
        start_id, start_id + len(meta), dtype=np.int64
    )
    return meta


def next_sample_id(real_meta):
    numeric = pd.to_numeric(real_meta["sample_id"], errors="coerce")

    if numeric.notna().all():
        return int(numeric.max()) + 1

    return len(real_meta)


# ---------------------------------------------------------------------------
# Metadata preparation
# ---------------------------------------------------------------------------

def prepare_real_metadata(meta_real):
    """
    Preserve all useful real metadata while adding standardized fields.
    """
    meta = meta_real.copy()

    meta = add_class_columns(meta)

    meta["source"] = "REAL"

    # Real data are not generated.
    meta["generation_type"] = "REAL"
    meta["generation_class"] = meta["label"]

    # There is no synthetic source sample ID.
    meta["source_sample_id"] = pd.NA

    return meta


def prepare_config02_synthetic_metadata(
    n_synth,
    start_id,
    generation_type="TimeGAN",
    multiplier=5,
):
    """
    Config 02 contains generated FALLS only.

    Its current metadata_Synthetic_Falls_Only.csv is a one-row generation
    summary, not sample-level metadata. Therefore the only safe per-sample
    information is that every generated sample is FALL.

    We explicitly represent that fact rather than pretending the summary row
    maps one-to-one to 495 samples.
    """
    meta = pd.DataFrame(index=np.arange(n_synth))

    meta["sample_id"] = np.arange(
        start_id, start_id + n_synth, dtype=np.int64
    )
    meta["label"] = "FALL"
    meta["class_id"] = CLASS_MAP["FALL"]
    meta["activity_type"] = "FALL"
    meta["source"] = "GENERATED"
    meta["generation_type"] = generation_type
    meta["generation_class"] = "FALL"
    meta["source_sample_id"] = pd.NA

    # These real-event fields are intentionally unknown for synthetic data.
    for col in [
        "subject_id",
        "fall_id",
        "fall_number",
        "segment_id",
        "center_time",
        "window_start",
        "window_end",
        "original_frame_count",
    ]:
        meta[col] = pd.NA

    meta["target_frames"] = 32
    meta["feature_count"] = 1137
    meta["multiplier"] = multiplier

    return meta


def prepare_config03_synthetic_metadata(meta_synth):
    """
    Config 03 has genuine sample-level metadata for all 5,070 generated
    samples. Preserve its labels and generation information exactly.

    No class ordering is inferred from the NPY array. The CSV row order is
    assumed to correspond to NPY row order, which is the required alignment
    contract for the generated dataset.
    """
    meta = meta_synth.copy()

    required = {"sample_id", "label"}
    missing = required - set(meta.columns)

    if missing:
        raise ValueError(
            f"Config 03 synthetic metadata is missing required columns: "
            f"{sorted(missing)}"
        )

    meta = add_class_columns(meta)

    # Normalize source to the standard value regardless of what the source
    # CSV used (e.g. "synthetic" lowercase) — the pipeline downstream
    # expects the literal string "GENERATED".
    meta["source"] = "GENERATED"

    if "generation_type" not in meta.columns:
        meta["generation_type"] = "TimeGAN"

    if "generation_class" not in meta.columns:
        meta["generation_class"] = meta["label"]

    if "source_sample_id" not in meta.columns:
        meta["source_sample_id"] = pd.NA

    # These fields are not supplied for synthetic samples.
    for col in [
        "subject_id",
        "fall_id",
        "fall_number",
        "segment_id",
        "center_time",
        "window_start",
        "window_end",
        "original_frame_count",
    ]:
        if col not in meta.columns:
            meta[col] = pd.NA

    if "target_frames" not in meta.columns:
        meta["target_frames"] = 32

    if "feature_count" not in meta.columns:
        meta["feature_count"] = 1137

    if "multiplier" not in meta.columns:
        meta["multiplier"] = pd.NA

    # The synthetic metadata's sample_id is local to the generated array.
    # It must not collide with real sample IDs in the final dataset.
    meta["source_sample_id"] = meta["sample_id"]

    return meta


def finalize_metadata(meta):
    """
    Create one stable metadata schema for both configurations.
    """
    meta = meta.copy()

    for col in METADATA_COLUMNS:
        if col not in meta.columns:
            meta[col] = pd.NA

    meta = meta[METADATA_COLUMNS]

    # Ensure labels/classes are internally consistent.
    meta["label"] = meta["label"].map(normalize_label)
    meta["class_id"] = meta["label"].map(CLASS_MAP).astype(np.int8)

    expected_activity = meta["label"]
    if not meta["activity_type"].astype("string").eq(
        expected_activity.astype("string")
    ).all():
        raise ValueError(
            "activity_type does not consistently represent the label."
        )

    return meta.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    config_name,
    real_npy_path,
    real_csv_path,
    synth_npy_path,
    synth_meta_path,
    out_dir,
    config02=False,
):
    print("\n" + "=" * 78)
    print(config_name)
    print("=" * 78)

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    X_real = load_npy(real_npy_path)
    meta_real_raw = load_csv(real_csv_path)
    X_synth = load_npy(synth_npy_path)
    meta_synth_raw = load_csv(synth_meta_path)

    print(f"Real array:       {X_real.shape} {X_real.dtype}")
    print(f"Real metadata:    {meta_real_raw.shape}")
    print(f"Synthetic array:  {X_synth.shape} {X_synth.dtype}")
    print(f"Synthetic metadata: {meta_synth_raw.shape}")

    # ------------------------------------------------------------------
    # 2. Validate source alignment
    # ------------------------------------------------------------------
    validate_metadata_array_alignment(
        X_real, meta_real_raw, "Real dataset"
    )

    if config02:
        # Config 02 has a generation-summary CSV with 1 row.
        # It cannot be used as sample-level metadata.
        if len(meta_synth_raw) != 1:
            raise ValueError(
                "Config 02 expected a one-row generation summary, but found "
                f"{len(meta_synth_raw)} rows."
            )

        required_summary = {
            "dataset",
            "selected_multiplier",
            "selected_N",
            "n_features",
            "modeled_dim",
        }
        missing = required_summary - set(meta_synth_raw.columns)
        if missing:
            raise ValueError(
                "Config 02 synthetic summary is missing columns: "
                f"{sorted(missing)}"
            )

        selected_n = int(meta_synth_raw.iloc[0]["selected_N"])
        if selected_n != X_synth.shape[0]:
            raise ValueError(
                "Config 02 synthetic summary selected_N does not match "
                f"the NPY sample count: selected_N={selected_n}, "
                f"NPY={X_synth.shape[0]}."
            )

        multiplier = meta_synth_raw.iloc[0]["selected_multiplier"]

        # The synthetic array is known to contain generated FALLS only.
        meta_synth = prepare_config02_synthetic_metadata(
            n_synth=X_synth.shape[0],
            start_id=next_sample_id(meta_real_raw),
            generation_type="TimeGAN",
            multiplier=multiplier,
        )

    else:
        # Config 03 MUST have one metadata row per synthetic sample.
        validate_metadata_array_alignment(
            X_synth, meta_synth_raw, "Config 03 synthetic dataset"
        )
        meta_synth = prepare_config03_synthetic_metadata(meta_synth_raw)

    validate_shape_compatibility(X_real, X_synth, config_name)

    # ------------------------------------------------------------------
    # 3. Prepare metadata
    # ------------------------------------------------------------------
    meta_real = prepare_real_metadata(meta_real_raw)

    # Assign final unique sample IDs to generated samples.
    meta_synth = assign_new_sample_ids(
        meta_synth,
        start_id=next_sample_id(meta_real),
    )

    # ------------------------------------------------------------------
    # 4. Concatenate arrays
    # ------------------------------------------------------------------
    X_final = np.concatenate([X_real, X_synth], axis=0)

    # ------------------------------------------------------------------
    # 5. Concatenate metadata in exactly the same row order
    # ------------------------------------------------------------------
    meta_final = pd.concat(
        [meta_real, meta_synth],
        ignore_index=True,
    )

    meta_final = finalize_metadata(meta_final)

    # ------------------------------------------------------------------
    # 6. Strong validation
    # ------------------------------------------------------------------
    if len(meta_final) != X_final.shape[0]:
        raise ValueError(
            f"Final array/metadata mismatch: "
            f"{X_final.shape[0]} vs {len(meta_final)}"
        )

    validate_sample_ids(meta_final)
    check_no_nan_inf(X_final, "final array")

    # Validate source/sample counts.
    expected_real = len(meta_real)
    expected_synth = len(meta_synth)

    source_counts = meta_final["source"].value_counts()
    real_count = int(source_counts.get("REAL", 0))
    generated_count = int(source_counts.get("GENERATED", 0))

    if real_count != expected_real:
        raise ValueError(
            f"REAL source count mismatch: expected {expected_real}, "
            f"got {real_count}"
        )

    if generated_count != expected_synth:
        raise ValueError(
            f"GENERATED source count mismatch: expected {expected_synth}, "
            f"got {generated_count}"
        )

    # Validate label consistency.
    label_counts = meta_final["label"].value_counts()
    real_labels = meta_real["label"].value_counts()
    synth_labels = meta_synth["label"].value_counts()

    final_non_fall = int(label_counts.get("NON_FALL", 0))
    final_fall = int(label_counts.get("FALL", 0))

    real_non_fall = int(real_labels.get("NON_FALL", 0))
    real_fall = int(real_labels.get("FALL", 0))

    synth_non_fall = int(synth_labels.get("NON_FALL", 0))
    synth_fall = int(synth_labels.get("FALL", 0))

    if final_non_fall != real_non_fall + synth_non_fall:
        raise ValueError("Final NON_FALL count is inconsistent.")

    if final_fall != real_fall + synth_fall:
        raise ValueError("Final FALL count is inconsistent.")

    # Check generation_class matches label for every row.
    if not meta_final["generation_class"].astype("string").eq(
        meta_final["label"].astype("string")
    ).all():
        raise ValueError(
            "generation_class does not consistently match label."
        )

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)

    out_npy = os.path.join(out_dir, "X_imu_radar.npy")
    out_csv = os.path.join(out_dir, "metadata.csv")

    np.save(out_npy, X_final)
    meta_final.to_csv(out_csv, index=False)

    # ------------------------------------------------------------------
    # 8. Human-readable report
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("FINAL DATASET")
    print("-" * 78)
    print(f"Shape:                 {X_final.shape}")
    print(f"dtype:                 {X_final.dtype}")
    print(f"Metadata rows:         {len(meta_final)}")
    print()
    print("LABEL DISTRIBUTION")
    print(f"  NON_FALL:            {final_non_fall}")
    print(f"  FALL:                {final_fall}")
    print()
    print("BY SOURCE")
    print(f"  REAL:                {real_count}")
    print(f"  GENERATED:           {generated_count}")
    print()
    print("REAL LABELS")
    print(f"  NON_FALL:            {real_non_fall}")
    print(f"  FALL:                {real_fall}")
    print()
    print("GENERATED LABELS")
    print(f"  NON_FALL:            {synth_non_fall}")
    print(f"  FALL:                {synth_fall}")
    print()
    print("GENERATED CLASS DISTRIBUTION")
    print(meta_synth["generation_class"].value_counts().to_string())
    print()
    print("METADATA COLUMNS")
    for col in meta_final.columns:
        print(f"  - {col}")
    print()
    print("Validation: PASSED")
    print(f"Saved array:    {out_npy}")
    print(f"Saved metadata: {out_csv}")

    return out_npy, out_csv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--base-dir",
        default=None,
        help="Directory containing the 02 and 03 configuration folders.",
    )

    parser.add_argument(
        "--out-dir",
        default="Classification_Datasets",
        help="Output root directory.",
    )

    args = parser.parse_args()

    base = args.base_dir or os.path.dirname(os.path.abspath(__file__))
    out_root = args.out_dir

    cfg02 = os.path.join(
        base, "02-All_ADLs+Generated_Falls_only"
    )
    cfg03 = os.path.join(
        base, "03-All_ADLs+All_Generated_ADLs"
    )

    outputs = []

    # ==================================================================
    # CONFIGURATION 02
    # ==================================================================
    out02 = os.path.join(
        out_root, "02-Real_ADLs+Generated_Falls"
    )

    outputs.append(
        build_dataset(
            config_name="Configuration 02: Real ADLs + Generated FALLS",
            real_npy_path=os.path.join(
                cfg02, "X_imu_radar_All_ADLs.npy"
            ),
            real_csv_path=os.path.join(
                cfg02, "metadata_All_ADLs.csv"
            ),
            synth_npy_path=os.path.join(
                cfg02, "Synthetic_imu_radar_Falls_Only.npy"
            ),
            synth_meta_path=os.path.join(
                cfg02, "metadata_Synthetic_Falls_Only.csv"
            ),
            out_dir=out02,
            config02=True,
        )
    )

    # ==================================================================
    # CONFIGURATION 03
    # ==================================================================
    out03 = os.path.join(
        out_root, "03-Real_ADLs+All_Generated_ADLs"
    )

    outputs.append(
        build_dataset(
            config_name="Configuration 03: Real ADLs + All Generated ADLs",
            real_npy_path=os.path.join(
                cfg03, "X_imu_radar_All_ADLs.npy"
            ),
            real_csv_path=os.path.join(
                cfg03, "metadata_All_ADLs.csv"
            ),
            synth_npy_path=os.path.join(
                cfg03, "Synthetic_imu_radar_All_ADLs.npy"
            ),
            synth_meta_path=os.path.join(
                cfg03, "metadata_Synthetic_All_ADLs.csv"
            ),
            out_dir=out03,
            config02=False,
        )
    )

    print("\n" + "=" * 78)
    print("ALL CONFIGURATIONS COMPLETED")
    print("=" * 78)

    for npy_path, csv_path in outputs:
        print(f"Array:    {npy_path}")
        print(f"Metadata: {csv_path}")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, ValueError, FileNotFoundError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
