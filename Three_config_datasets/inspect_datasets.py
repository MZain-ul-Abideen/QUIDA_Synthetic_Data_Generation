import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(".")

CONFIGS = {
    "02": BASE / "02-All_ADLs+Generated_Falls_only",
    "03": BASE / "03-All_ADLs+All_Generated_ADLs",
}


def inspect_config(name, folder):
    print("\n" + "=" * 80)
    print(f"CONFIGURATION {name}")
    print(folder)
    print("=" * 80)

    npy_files = sorted(folder.glob("*.npy"))
    csv_files = sorted(folder.glob("*.csv"))

    print("\n--- FILES ---")
    for f in npy_files + csv_files:
        print(f"  {f.name}")

    # ------------------------------------------------------------
    # Inspect NPY files
    # ------------------------------------------------------------
    print("\n--- NPY FILES ---")

    arrays = {}

    for f in npy_files:
        try:
            arr = np.load(f, mmap_mode="r")
            arrays[f.name] = arr

            print(f"\n{f.name}")
            print(f"  shape       : {arr.shape}")
            print(f"  dtype       : {arr.dtype}")
            print(f"  ndim        : {arr.ndim}")

            if arr.ndim >= 2:
                print(f"  samples     : {arr.shape[0]}")

            # Basic numerical check
            if np.issubdtype(arr.dtype, np.number):
                print(f"  NaN count   : {np.isnan(arr).sum()}")
                print(f"  Inf count   : {np.isinf(arr).sum()}")

                print(f"  min         : {np.nanmin(arr):.6f}")
                print(f"  max         : {np.nanmax(arr):.6f}")

        except Exception as e:
            print(f"\nERROR reading {f.name}")
            print(f"  {e}")

    # ------------------------------------------------------------
    # Inspect CSV files
    # ------------------------------------------------------------
    print("\n--- CSV FILES ---")

    metadata = {}

    for f in csv_files:
        try:
            df = pd.read_csv(f)
            metadata[f.name] = df

            print(f"\n{f.name}")
            print(f"  rows        : {len(df)}")
            print(f"  columns     : {len(df.columns)}")

            print("  columns:")
            for col in df.columns:
                print(f"    - {col}")

            print("\n  first 3 rows:")
            print(df.head(3).to_string(index=False))

            # Possible label columns
            possible_labels = [
                c for c in df.columns
                if any(
                    key in c.lower()
                    for key in ["label", "class", "fall", "activity", "type"]
                )
            ]

            if possible_labels:
                print("\n  possible label columns:")

                for col in possible_labels:
                    print(f"\n    {col}:")
                    print(df[col].value_counts(dropna=False).to_string())

        except Exception as e:
            print(f"\nERROR reading {f.name}")
            print(f"  {e}")

    # ------------------------------------------------------------
    # Compare real ADL files
    # ------------------------------------------------------------
    real_npy = folder / "X_imu_radar_All_ADLs.npy"
    real_csv = folder / "metadata_All_ADLs.csv"

    print("\n--- REAL ADL CONSISTENCY ---")

    if real_npy.exists() and real_csv.exists():
        X = np.load(real_npy, mmap_mode="r")
        df = pd.read_csv(real_csv)

        print(f"  X samples       : {X.shape[0]}")
        print(f"  metadata rows   : {len(df)}")

        if X.shape[0] == len(df):
            print("  STATUS          : OK - sample counts match")
        else:
            print("  STATUS          : ERROR - sample counts DO NOT match")

    # ------------------------------------------------------------
    # Compare generated files
    # ------------------------------------------------------------
    generated_npy = [
        f for f in npy_files
        if "Synthetic" in f.name
    ]

    generated_csv = [
        f for f in csv_files
        if "Synthetic" in f.name
    ]

    print("\n--- GENERATED DATA CONSISTENCY ---")

    for npy_file in generated_npy:

        X = np.load(npy_file, mmap_mode="r")

        # Try to find corresponding CSV based on "Synthetic" naming
        possible_csv = [
            f for f in generated_csv
            if f.stem.replace("metadata_", "").lower()
            in npy_file.stem.lower()
            or (
                "Falls_Only" in npy_file.name
                and "Falls_Only" in f.name
            )
            or (
                "All_ADLs" in npy_file.name
                and "All_ADLs" in f.name
            )
        ]

        print(f"\n  {npy_file.name}")
        print(f"    NPY samples : {X.shape[0]}")

        if possible_csv:
            csv_file = possible_csv[0]
            df = pd.read_csv(csv_file)

            print(f"    CSV file    : {csv_file.name}")
            print(f"    CSV rows    : {len(df)}")

            if X.shape[0] == len(df):
                print("    STATUS      : OK - sample counts match")
            else:
                print("    STATUS      : ERROR - sample counts DO NOT match")
        else:
            print("    WARNING     : Could not automatically find metadata CSV")

    # ------------------------------------------------------------
    # Shape compatibility
    # ------------------------------------------------------------
    print("\n--- SHAPE COMPATIBILITY ---")

    if len(npy_files) >= 2:
        for i in range(len(npy_files)):
            for j in range(i + 1, len(npy_files)):

                a_name = npy_files[i].name
                b_name = npy_files[j].name

                a = arrays.get(a_name)
                b = arrays.get(b_name)

                if a is None or b is None:
                    continue

                print(f"\n  {a_name}")
                print(f"  {b_name}")

                if a.ndim == b.ndim:
                    print(f"    ndim: {a.ndim} vs {b.ndim}")

                    if a.shape[1:] == b.shape[1:]:
                        print("    STATUS: COMPATIBLE for np.concatenate(axis=0)")
                    else:
                        print("    STATUS: NOT COMPATIBLE")
                        print(f"    Remaining dimensions:")
                        print(f"      A: {a.shape[1:]}")
                        print(f"      B: {b.shape[1:]}")
                else:
                    print("    STATUS: NOT COMPATIBLE - different ndim")

    print("\n" + "=" * 80)


# ================================================================
# RUN INSPECTION
# ================================================================

for name, folder in CONFIGS.items():
    inspect_config(name, folder)

print("\n\nINSPECTION COMPLETE")
print("No files were modified.")