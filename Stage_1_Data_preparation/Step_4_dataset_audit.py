# Step_4_dataset_audit
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# =====================================================
# PATHS
# =====================================================

DATA_DIR = Path("../Preprocessed_dataset")
OUTPUT_CSV = DATA_DIR / "dataset_audit_report.csv"

# =====================================================
# OUTPUT CAPTURE
# =====================================================

output_lines = []


def log(text=""):
    """Print to terminal and store the same line for CSV."""
    print(text)
    output_lines.append(text)


# =====================================================
# LOAD DATASETS
# =====================================================

X_imu = np.load(DATA_DIR / "X_imu_raw.npy")
X_imu_ir = np.load(DATA_DIR / "X_imu_ir_raw.npy")
X_all = np.load(DATA_DIR / "X_imu_ir_radar_raw.npy")

metadata = pd.read_csv(DATA_DIR / "metadata.csv")


# =====================================================
# BASIC SHAPES
# =====================================================

log("\n" + "=" * 60)
log("DATASET SHAPES")
log("=" * 60)

log(f"X_imu_raw          : {X_imu.shape}")
log(f"X_imu_ir_raw       : {X_imu_ir.shape}")
log(f"X_imu_ir_radar_raw : {X_all.shape}")


# =====================================================
# NAN CHECK
# =====================================================

log("\n" + "=" * 60)
log("NaN CHECK")
log("=" * 60)

log(f"IMU NaNs          : {np.isnan(X_imu).sum()}")
log(f"IMU+IR NaNs       : {np.isnan(X_imu_ir).sum()}")
log(f"IMU+IR+Radar NaNs : {np.isnan(X_all).sum()}")


# =====================================================
# INF CHECK
# =====================================================

log("\n" + "=" * 60)
log("INF CHECK")
log("=" * 60)

log(f"IMU Infs          : {np.isinf(X_imu).sum()}")
log(f"IMU+IR Infs       : {np.isinf(X_imu_ir).sum()}")
log(f"IMU+IR+Radar Infs : {np.isinf(X_all).sum()}")


# =====================================================
# IMU STATISTICS
# =====================================================

log("\n" + "=" * 60)
log("IMU STATISTICS")
log("=" * 60)

imu_flat = X_imu.reshape(-1, 3)

feature_names = ["ax", "ay", "az"]

for i, name in enumerate(feature_names):

    col = imu_flat[:, i]

    log(f"\n{name}")
    log(f"Mean : {np.mean(col):.6f}")
    log(f"Std  : {np.std(col):.6f}")
    log(f"Min  : {np.min(col):.6f}")
    log(f"Max  : {np.max(col):.6f}")


# =====================================================
# IR STATISTICS
# =====================================================

log("\n" + "=" * 60)
log("IR STATISTICS")
log("=" * 60)

ir_flat = X_imu_ir[:, :, 3:].reshape(-1)

log(f"Mean : {np.mean(ir_flat):.6f}")
log(f"Std  : {np.std(ir_flat):.6f}")
log(f"Min  : {np.min(ir_flat):.6f}")
log(f"Max  : {np.max(ir_flat):.6f}")


# =====================================================
# RADAR STATISTICS
# =====================================================

log("\n" + "=" * 60)
log("RADAR STATISTICS")
log("=" * 60)

radar_flat = X_all[:, :, 771:].reshape(-1)

log(f"Mean : {np.mean(radar_flat):.6f}")
log(f"Std  : {np.std(radar_flat):.6f}")
log(f"Min  : {np.min(radar_flat):.6f}")
log(f"Max  : {np.max(radar_flat):.6f}")


# =====================================================
# SUBJECT DISTRIBUTION
# =====================================================

log("\n" + "=" * 60)
log("SUBJECT DISTRIBUTION")
log("=" * 60)

subject_counts = (
    metadata["subject_id"]
    .value_counts()
    .sort_index()
)

# Reproduce the pandas-style output
log("subject_id")

for subject, count in subject_counts.items():
    log(f"{subject:<10}{count}")

log(f"Name: count, dtype: int64")


# =====================================================
# SEQUENCE LENGTH VERIFICATION
# =====================================================

log("\n" + "=" * 60)
log("SEQUENCE VERIFICATION")
log("=" * 60)

log(f"IMU sequence length       : {X_imu.shape[1]}")
log(f"IMU+IR sequence length    : {X_imu_ir.shape[1]}")
log(f"IMU+IR+Radar seq length   : {X_all.shape[1]}")


# =====================================================
# FEATURE COUNTS
# =====================================================

log("\n" + "=" * 60)
log("FEATURE COUNTS")
log("=" * 60)

log(f"IMU features          : {X_imu.shape[2]}")
log(f"IMU+IR features       : {X_imu_ir.shape[2]}")
log(f"IMU+IR+Radar features : {X_all.shape[2]}")


# =====================================================
# COMPLETION
# =====================================================

log("\nAudit completed successfully.")


# =====================================================
# SAVE EXACT TERMINAL OUTPUT TO CSV
# =====================================================

audit_df = pd.DataFrame({
    "Audit Report": output_lines
})

audit_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8"
)

# Final confirmation in terminal
print(f"\nAudit report CSV saved to:")
print(OUTPUT_CSV.resolve())