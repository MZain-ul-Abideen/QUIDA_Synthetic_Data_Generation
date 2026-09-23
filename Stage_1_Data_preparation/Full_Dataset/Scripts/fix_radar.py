# radar_fix
import pandas as pd
from pathlib import Path

# ==========================================================
# CONFIG
# ==========================================================

EXPECTED_COLS = 1135  # timestamp + 1134 radar features

# ==========================================================
# CLEAN ALL RADAR FILES
# ==========================================================

base_dir = Path(__file__).resolve().parent.parent
raw_radar_dir = base_dir / "Raw_data"
radar_files = sorted(raw_radar_dir.glob("subject_*_radar.csv"))

for filepath in radar_files:
    print(f"Processing {filepath} ...", end=" ")

    rows = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            tokens = line.split(",")

            try:
                values = [float(t) if t.strip() != "" else 0.0 for t in tokens]
            except ValueError:
                continue

            if len(values) == 0:
                continue

            # Zero-pad short rows
            if len(values) < EXPECTED_COLS:
                values += [0.0] * (EXPECTED_COLS - len(values))

            # Truncate overlong rows
            elif len(values) > EXPECTED_COLS:
                values = values[:EXPECTED_COLS]

            rows.append(values)

    df = pd.DataFrame(rows, columns=range(EXPECTED_COLS))

    # ----------------------------------------------------------
    # Overwrite original file with clean version
    # ----------------------------------------------------------

    df.to_csv(filepath, index=False, header=False)

    print(f"Done. Shape: {df.shape}")

print("\nAll radar files cleaned.")