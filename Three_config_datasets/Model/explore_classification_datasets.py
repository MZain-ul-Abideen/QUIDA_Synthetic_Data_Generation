#!/usr/bin/env python3
"""
explore_classification_datasets.py

Explore and compare the CSV metadata for all classification datasets.

Expected structure:

Classification_Datasets/
├── 01-Real_ADLs/
│   ├── X_imu_radar.npy
│   └── metadata.csv
├── 02-Real_ADLs+Generated_Falls/
│   ├── X_imu_radar.npy
│   └── metadata.csv
└── 03-Real_ADLs+All_Generated_ADLs/
    ├── X_imu_radar.npy
    └── metadata.csv

Usage:
    python explore_classification_datasets.py

Optional:
    python explore_classification_datasets.py --data-dir Classification_Datasets
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

EXPECTED_CONFIGS = [
    "01-Real_ADLs",
    "02-Real_ADLs+Generated_Falls",
    "03-Real_ADLs+All_Generated_ADLs",
]


# ============================================================
# Helpers
# ============================================================

def print_header(title, char="="):
    print("\n" + char * 80)
    print(title)
    print(char * 80)


def print_subheader(title):
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def load_dataset(config_dir):
    csv_path = os.path.join(config_dir, "metadata.csv")
    npy_path = os.path.join(config_dir, "X_imu_radar.npy")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing metadata.csv: {csv_path}")

    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"Missing X_imu_radar.npy: {npy_path}")

    metadata = pd.read_csv(csv_path)
    X = np.load(npy_path, mmap_mode="r")

    return metadata, X


def format_value(value):
    if pd.isna(value):
        return "NaN"
    return str(value)


def print_value_counts(series, name):
    print(f"\n{name}")
    print("-" * len(name))

    counts = series.value_counts(dropna=False)

    total = len(series)

    for value, count in counts.items():
        percentage = (count / total) * 100
        print(
            f"  {format_value(value):<30} "
            f"{count:>8} "
            f"({percentage:>6.2f}%)"
        )


def print_missing_values(df):
    missing = df.isna().sum()
    missing_pct = (missing / len(df)) * 100

    rows = []

    for col in df.columns:
        if missing[col] > 0:
            rows.append(
                (
                    col,
                    int(missing[col]),
                    float(missing_pct[col]),
                )
            )

    if not rows:
        print("  No missing values.")
        return

    print(
        f"  {'Column':<30}"
        f"{'Missing':>12}"
        f"{'Percent':>12}"
    )
    print("  " + "-" * 54)

    for col, count, pct in rows:
        print(
            f"  {col:<30}"
            f"{count:>12}"
            f"{pct:>11.2f}%"
        )


def print_unique_values(df):
    categorical_candidates = [
        "label",
        "class_id",
        "activity_type",
        "source",
        "generation_type",
        "generation_class",
    ]

    for col in categorical_candidates:
        if col not in df.columns:
            continue

        values = df[col].value_counts(dropna=False)

        print(f"\n  {col}:")
        for value, count in values.items():
            print(f"    {format_value(value):<25} {count}")


# ============================================================
# Per configuration analysis
# ============================================================

def analyze_configuration(config_name, config_dir):
    print_header(f"CONFIGURATION: {config_name}")

    metadata, X = load_dataset(config_dir)

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print_subheader("1. BASIC DATASET INFORMATION")

    print(f"  Directory       : {config_dir}")
    print(f"  Metadata rows   : {len(metadata)}")
    print(f"  Metadata cols   : {len(metadata.columns)}")
    print(f"  NPY shape       : {X.shape}")
    print(f"  NPY dtype       : {X.dtype}")

    if X.ndim == 3:
        print(f"  Samples         : {X.shape[0]}")
        print(f"  Time frames     : {X.shape[1]}")
        print(f"  Features        : {X.shape[2]}")

    # --------------------------------------------------------
    # Array / metadata alignment
    # --------------------------------------------------------

    print_subheader("2. ARRAY / METADATA ALIGNMENT")

    if X.shape[0] == len(metadata):
        print(
            f"  [OK] NPY samples ({X.shape[0]}) == "
            f"metadata rows ({len(metadata)})"
        )
    else:
        print(
            f"  [ERROR] NPY samples ({X.shape[0]}) != "
            f"metadata rows ({len(metadata)})"
        )

    # --------------------------------------------------------
    # Columns
    # --------------------------------------------------------

    print_subheader("3. METADATA COLUMNS")

    for i, col in enumerate(metadata.columns, start=1):
        dtype = metadata[col].dtype
        unique = metadata[col].nunique(dropna=False)
        print(
            f"  {i:>2}. {col:<30} "
            f"dtype={str(dtype):<12} "
            f"unique={unique}"
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    print_subheader("4. LABEL DISTRIBUTION")

    if "label" in metadata.columns:
        print_value_counts(metadata["label"], "label")

    if "class_id" in metadata.columns:
        print_value_counts(metadata["class_id"], "class_id")

    # --------------------------------------------------------
    # Activity type
    # --------------------------------------------------------

    if "activity_type" in metadata.columns:
        print_subheader("5. ACTIVITY TYPE")

        print_value_counts(
            metadata["activity_type"],
            "activity_type"
        )

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    if "source" in metadata.columns:
        print_subheader("6. REAL vs GENERATED")

        print_value_counts(
            metadata["source"],
            "source"
        )

    # --------------------------------------------------------
    # Labels by source
    # --------------------------------------------------------

    if "source" in metadata.columns and "label" in metadata.columns:

        print_subheader("7. LABEL DISTRIBUTION BY SOURCE")

        table = pd.crosstab(
            metadata["source"],
            metadata["label"],
            margins=True,
        )

        print(table.to_string())

        print("\nPercentages within each source:")

        percentage_table = pd.crosstab(
            metadata["source"],
            metadata["label"],
            normalize="index",
        ) * 100

        print(
            percentage_table.round(2).to_string()
        )

    # --------------------------------------------------------
    # Generation information
    # --------------------------------------------------------

    generation_columns = [
        "generation_type",
        "generation_class",
        "multiplier",
    ]

    existing_generation_columns = [
        col
        for col in generation_columns
        if col in metadata.columns
    ]

    if existing_generation_columns:

        print_subheader("8. GENERATION INFORMATION")

        for col in existing_generation_columns:
            print_value_counts(
                metadata[col],
                col
            )

    # --------------------------------------------------------
    # Generation class vs label
    # --------------------------------------------------------

    if (
        "label" in metadata.columns
        and "generation_class" in metadata.columns
    ):

        print_subheader("9. LABEL vs GENERATION_CLASS CONSISTENCY")

        generated_mask = (
            metadata.get("source", pd.Series(index=metadata.index))
            .astype("string")
            .eq("GENERATED")
        )

        generated = metadata[generated_mask]

        if len(generated) > 0:

            comparison = pd.crosstab(
                metadata.loc[generated_mask, "label"],
                metadata.loc[generated_mask, "generation_class"],
                margins=True,
            )

            print(comparison.to_string())

            mismatch = (
                metadata.loc[generated_mask, "label"].astype("string")
                != metadata.loc[
                    generated_mask, "generation_class"
                ].astype("string")
            )

            mismatch_count = int(mismatch.sum())

            if mismatch_count == 0:
                print("\n  [OK] No generated label mismatches.")
            else:
                print(
                    f"\n  [ERROR] {mismatch_count} generated rows "
                    f"have label != generation_class."
                )

        else:
            print("  No GENERATED samples.")

        if (
            "label" in metadata.columns
            and "generation_class" in metadata.columns
        ):

            print_subheader("9. LABEL vs GENERATION_CLASS CONSISTENCY")

            generated_mask = (
                metadata.get("source", pd.Series(index=metadata.index))
                .astype("string")
                .eq("GENERATED")
            )

            generated = metadata[generated_mask]

            if len(generated) > 0:

                comparison = pd.crosstab(
                    metadata.loc[generated_mask, "label"],
                    metadata.loc[generated_mask, "generation_class"],
                    margins=True,
                )

                print(comparison.to_string())

                mismatch = (
                    metadata.loc[generated_mask, "label"].astype("string")
                    != metadata.loc[
                        generated_mask, "generation_class"
                    ].astype("string")
                )

                mismatch_count = int(mismatch.sum())

                if mismatch_count == 0:
                    print("\n  [OK] No generated label mismatches.")
                else:
                    print(
                        f"\n  [ERROR] {mismatch_count} generated rows "
                        f"have label != generation_class."
                    )

            else:
                print("  No GENERATED samples.")

    # --------------------------------------------------------
    # Subject information
    # --------------------------------------------------------

    if "subject_id" in metadata.columns:

        print_subheader("10. SUBJECT DISTRIBUTION")

        print_value_counts(
            metadata["subject_id"],
            "subject_id"
        )

        if "label" in metadata.columns:

            print("\nLabels per subject:")

            subject_label = pd.crosstab(
                metadata["subject_id"],
                metadata["label"],
                margins=True,
            )

            print(subject_label.to_string())

    # --------------------------------------------------------
    # Fall information
    # --------------------------------------------------------

    fall_columns = [
        "fall_id",
        "fall_number",
        "segment_id",
    ]

    existing_fall_columns = [
        col
        for col in fall_columns
        if col in metadata.columns
    ]

    if existing_fall_columns:

        print_subheader("11. FALL/EVENT METADATA")

        for col in existing_fall_columns:

            non_null = metadata[col].notna().sum()

            print(
                f"  {col:<20} "
                f"non-null={non_null:<8} "
                f"unique={metadata[col].nunique(dropna=True)}"
            )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print_subheader("12. MISSING VALUES")

    print_missing_values(metadata)

    # --------------------------------------------------------
    # Duplicate IDs
    # --------------------------------------------------------

    if "sample_id" in metadata.columns:

        print_subheader("13. SAMPLE ID VALIDATION")

        duplicate_count = int(
            metadata["sample_id"].duplicated().sum()
        )

        if duplicate_count == 0:
            print("  [OK] sample_id values are unique.")
        else:
            print(
                f"  [ERROR] {duplicate_count} duplicate "
                f"sample_id values found."
            )

    # --------------------------------------------------------
    # Label/class consistency
    # --------------------------------------------------------

    if "label" in metadata.columns and "class_id" in metadata.columns:

        print_subheader("14. LABEL / CLASS_ID CONSISTENCY")

        expected = metadata["label"].map(
            {
                "NON_FALL": 0,
                "FALL": 1,
            }
        )

        mismatch = (
            expected.notna()
            & (expected != metadata["class_id"])
        )

        mismatch_count = int(mismatch.sum())

        if mismatch_count == 0:
            print(
                "  [OK] class_id correctly represents label "
                "for all recognized classes."
            )
        else:
            print(
                f"  [ERROR] {mismatch_count} label/class_id "
                f"mismatches found."
            )

    # --------------------------------------------------------
    # Compact metadata preview
    # --------------------------------------------------------

    print_subheader("15. FIRST 5 METADATA ROWS")

    print(
        metadata.head(5).to_string(index=False)
    )

    # --------------------------------------------------------
    # Last rows
    # --------------------------------------------------------

    print_subheader("16. LAST 5 METADATA ROWS")

    print(
        metadata.tail(5).to_string(index=False)
    )

    return {
        "name": config_name,
        "metadata": metadata,
        "shape": X.shape,
        "samples": X.shape[0],
        "features": X.shape[-1],
    }


# ============================================================
# Cross-configuration comparison
# ============================================================

def compare_configurations(results):

    print_header("CROSS-CONFIGURATION COMPARISON")

    # --------------------------------------------------------
    # Dataset sizes
    # --------------------------------------------------------

    print_subheader("1. DATASET SIZE")

    print(
        f"{'Configuration':<40}"
        f"{'Samples':>12}"
        f"{'Frames':>10}"
        f"{'Features':>12}"
    )

    print("-" * 74)

    for result in results:

        shape = result["shape"]

        print(
            f"{result['name']:<40}"
            f"{shape[0]:>12}"
            f"{shape[1]:>10}"
            f"{shape[2]:>12}"
        )

    # --------------------------------------------------------
    # Label comparison
    # --------------------------------------------------------

    print_subheader("2. LABEL COMPARISON")

    rows = []

    for result in results:

        df = result["metadata"]

        counts = df["label"].value_counts()

        rows.append(
            {
                "Configuration": result["name"],
                "NON_FALL": int(counts.get("NON_FALL", 0)),
                "FALL": int(counts.get("FALL", 0)),
                "TOTAL": len(df),
            }
        )

    comparison = pd.DataFrame(rows)

    print(
        comparison.to_string(index=False)
    )

    # --------------------------------------------------------
    # Label percentages
    # --------------------------------------------------------

    print_subheader("3. LABEL PERCENTAGES")

    percentage_rows = []

    for result in results:

        df = result["metadata"]

        counts = df["label"].value_counts()

        total = len(df)

        percentage_rows.append(
            {
                "Configuration": result["name"],
                "NON_FALL_%": round(
                    counts.get("NON_FALL", 0) / total * 100,
                    2,
                ),
                "FALL_%": round(
                    counts.get("FALL", 0) / total * 100,
                    2,
                ),
            }
        )

    percentage_df = pd.DataFrame(percentage_rows)

    print(
        percentage_df.to_string(index=False)
    )

    # --------------------------------------------------------
    # Source comparison
    # --------------------------------------------------------

    print_subheader("4. REAL vs GENERATED")

    rows = []

    for result in results:

        df = result["metadata"]

        if "source" not in df.columns:
            continue

        counts = df["source"].value_counts()

        rows.append(
            {
                "Configuration": result["name"],
                "REAL": int(counts.get("REAL", 0)),
                "GENERATED": int(counts.get("GENERATED", 0)),
                "TOTAL": len(df),
            }
        )

    if rows:
        print(
            pd.DataFrame(rows).to_string(index=False)
        )

    # --------------------------------------------------------
    # Source x label
    # --------------------------------------------------------

    print_subheader("5. SOURCE × LABEL")

    for result in results:

        df = result["metadata"]

        print(f"\n{result['name']}")

        if "source" not in df.columns:
            print("  No source column.")
            continue

        table = pd.crosstab(
            df["source"],
            df["label"],
            margins=True,
        )

        print(table.to_string())

    # --------------------------------------------------------
    # Generation type
    # --------------------------------------------------------

    print_subheader("6. GENERATION TYPE")

    for result in results:

        df = result["metadata"]

        print(f"\n{result['name']}")

        if "generation_type" not in df.columns:
            print("  No generation_type column.")
            continue

        print(
            df["generation_type"]
            .value_counts(dropna=False)
            .to_string()
        )

    # --------------------------------------------------------
    # Metadata schemas
    # --------------------------------------------------------

    print_subheader("7. METADATA SCHEMA COMPARISON")

    all_columns = sorted(
        set().union(
            *[
                set(result["metadata"].columns)
                for result in results
            ]
        )
    )

    print(
        f"\n{'Column':<32}",
        end="",
    )

    for result in results:
        print(
            f"{result['name'][:20]:>22}",
            end="",
        )

    print()

    print("-" * (32 + 22 * len(results)))

    for column in all_columns:

        print(f"{column:<32}", end="")

        for result in results:

            exists = column in result["metadata"].columns

            print(
                f"{'YES' if exists else 'NO':>22}",
                end="",
            )

        print()

    # --------------------------------------------------------
    # Unique categorical values
    # --------------------------------------------------------

    print_subheader("8. CATEGORICAL VALUE COMPARISON")

    categorical_columns = [
        "label",
        "class_id",
        "activity_type",
        "source",
        "generation_type",
        "generation_class",
    ]

    for column in categorical_columns:

        print(f"\n{column}")

        for result in results:

            df = result["metadata"]

            if column not in df.columns:
                print(
                    f"  {result['name']}: <missing column>"
                )
                continue

            values = sorted(
                [
                    format_value(v)
                    for v in df[column].drop_duplicates()
                ]
            )

            print(
                f"  {result['name']}: "
                f"{values}"
            )

    # --------------------------------------------------------
    # Final interpretation table
    # --------------------------------------------------------

    print_subheader("9. CLASSIFICATION DATASET SUMMARY")

    print(
        """
The table below describes what the classifier will actually receive.

Configuration 01:
    Real samples only.

Configuration 02:
    Real samples + generated FALL samples.

Configuration 03:
    Real samples + generated FALL and NON_FALL samples.

The target variable should be based on:
    label / class_id

The source and generation fields are provenance metadata.
They should NOT automatically be used as model input features.
"""
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Explore classification dataset metadata."
    )

    parser.add_argument(
        "--data-dir",
        default="Classification_Datasets",
        help="Directory containing the configuration folders.",
    )

    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)

    if not os.path.isdir(data_dir):
        print(
            f"ERROR: Directory not found: {data_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    print_header(
        "CLASSIFICATION DATASET EXPLORATION"
    )

    print(f"Root directory: {data_dir}")

    results = []

    for config_name in EXPECTED_CONFIGS:

        config_dir = os.path.join(
            data_dir,
            config_name,
        )

        if not os.path.isdir(config_dir):
            print(
                f"\nWARNING: Missing configuration directory:"
                f"\n  {config_dir}"
            )
            continue

        try:
            result = analyze_configuration(
                config_name,
                config_dir,
            )

            results.append(result)

        except Exception as exc:

            print(
                f"\nERROR while analyzing {config_name}: {exc}",
                file=sys.stderr,
            )

    if not results:
        print(
            "\nERROR: No valid classification datasets found.",
            file=sys.stderr,
        )
        sys.exit(1)

    compare_configurations(results)

    print_header(
        "EXPLORATION COMPLETE"
    )

    print(
        f"Successfully analyzed {len(results)} configuration(s)."
    )


if __name__ == "__main__":
    main()