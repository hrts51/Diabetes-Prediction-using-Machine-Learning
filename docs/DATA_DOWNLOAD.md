# Dataset Download Instructions

The dataset files are not included in this repository because they are 22MB each.

## Download Steps

1. Go to: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

2. Click the **Download** button (you need a free Kaggle account)

3. Extract the downloaded zip file

4. Place the following files in the `data/` folder of this project:

```
data/
├── diabetes_binary_health_indicators_BRFSS2015.csv      ← REQUIRED (main file)
├── diabetes_binary_5050split_health_indicators_BRFSS2015.csv
└── diabetes_012_health_indicators_BRFSS2015.csv
```

## Alternative — Auto-Generate Synthetic Data

If you cannot access Kaggle, run:

```bash
python src/download_data.py
```

This generates a realistic synthetic dataset with the same schema and class balance.
The pipeline will run correctly, but model performance will reflect synthetic data.

## Dataset Details

| Property | Value |
|---|---|
| Source | CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015 |
| Curator | Alex Teboul on Kaggle |
| Original | CDC.gov public data |
| License | Public domain (US government data) |
| Main file size | ~22 MB |
| Rows | 253,680 |
| Columns | 22 (21 features + 1 target) |
