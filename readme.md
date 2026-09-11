# Data Augmentation & Synthesis for QUIDA

This repository implements a two-stage pipeline for preparing and generating synthetic data from the QUIDA multimodal fall-detection dataset.

```text
Stage 1: Data Preparation
        ↓
Stage 2: Data Generation
        ↓
Evaluation & Comparison
```

## Dataset

Download the QUIDA dataset from:

[QUIDA Dataset on OSF](https://osf.io/yjgdv/overview?utm_source=chatgpt.com)

Place the dataset in the repository root:

```text
Data_Augmentation/
├── Dataset/
├── Stage_1_Data_preparation/
└── Stage_2_Data_generation/
```

The pipeline uses **IMU, IR Thermal Camera, and Radar** data. LiDAR is not used.

---

# Stage 1: Data Preparation

Stage 1 prepares the raw QUIDA data for synthetic-data generation.

```text
Step 1 → Step 2 → Step 3 → Step 4
```

Run from `Stage_1_Data_preparation/`:

```bash
python Step_1_fix_radar.py
python Step_2_extraction_script.py
python Step_3_sync_and_build_datasets.py
python Step_4_dataset_audit.py
```

### Step 1: Fix Radar

Cleans and rectangularises the raw radar files.

### Step 2: Extract Falls

Extracts a **6-second window** around each fall:

```text
3 seconds before + fall + 3 seconds after
```

The dataset contains **10 subjects × 10 falls = 100 expected falls**. Metadata is stored for each extracted fall.

### Step 3: Synchronise & Build

Removes invalid falls, synchronises the sensors, resamples sequences to **32 frames**, and generates four sensor configurations:

```text
X_imu_raw.npy
X_imu_ir_raw.npy
X_imu_radar_raw.npy
X_imu_ir_radar_raw.npy
```

The output is stored in:

```text
Preprocessed_dataset/
```

### Step 4: Dataset Audit

Audits the generated datasets for shapes, statistics, subject distribution, sequence lengths, and feature counts.

The report is saved to:

```text
Preprocessed_dataset/dataset_audit_report.csv
```

At this point, Stage 1 is complete, and the data is ready for Stage 2.

# Stage 2: Data Generation

Stage 2 generates synthetic data using three independent experiments. The experiments can be run **in any order**.

Each experiment is applied to four sensor configurations:

```text
1. IMU
2. IMU + IR
3. IMU + Radar
4. IMU + IR + Radar
```

This produces:

```text
3 experiments × 4 sensor configurations = 12 synthetic datasets
```

## Experiment 1: Jittering

Notebook:

```text
EXP_1_Jittering_SampleSize_repeated-run_Study.ipynb
```

Generates synthetic data using jittering and evaluates different sample sizes through repeated runs.

The previous `EXP_1.0_Jittering_Cascade_NRound_Evalaution` was removed due to methodological limitations. Details are provided in the project report.

## Experiment 2: VAE

Notebook:

```text
EXP_2_VAE_SampleSize_repeated-run_Study.ipynb
```

Generates synthetic data using a Variational Autoencoder (VAE) with repeated sample-size experiments.

## Experiment 3: TimeGAN

Notebook:

```text
EXP_3_TimeGAN_SampleSize_repeated-run_Study.ipynb
```

Generates synthetic time-series data using TimeGAN with repeated sample-size experiments.

Each experiment has its **own environment and requirements**, since the dependencies differ between stages/experiments.

---

# Evaluation & Comparison

The 12 generated datasets are passed to the evaluation and comparison pipeline.

The evaluation measures **fidelity** and **diversity**.

Statistical tests are also used to compare the generation methods and sensor configurations.

The evaluation produces a complete analysis of the synthetic datasets and enables comparison of:

* Synthetic-data generation methods
* Sensor configurations
* Sample sizes
* Overall synthetic-data quality

---

# Overall Pipeline

```text
                 QUIDA Dataset
                      │
                      ▼
             Stage 1: Preparation
                      │
                      ▼
             Preprocessed Dataset
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
      Jittering       VAE       TimeGAN
          │           │           │
          └───────────┼───────────┘
                      ▼
              12 Synthetic Datasets
                      │
                      ▼
          Evaluation & Comparison
                      │
                      ▼
             Complete Analysis
```
