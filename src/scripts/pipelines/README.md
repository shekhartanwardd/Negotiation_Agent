# Escalation Prediction Pipeline

This directory contains the escalation prediction pipeline script that processes customer support datasets and generates escalation probability predictions using a pre-trained LightGBM model.

## Overview

The pipeline performs the following steps:
1. Load dataset from CSV
2. Create derived features (IS_ND, IS_MnI, IS_PFQ, IS_OSI, IS_LATE, IS_WOD)
3. Handle missing values
4. Convert data types for LightGBM compatibility
5. Save non-log version of the dataset (before log transformations)
6. Apply log transformations to skewed features
7. Load the pre-trained LightGBM model
8. Generate escalation predictions
9. Export results to CSV

## Requirements

### Python Dependencies

```
numpy
pandas
scikit-learn
lightgbm
```

Install dependencies:
```bash
pip install numpy pandas scikit-learn lightgbm
```

### Pre-trained Model

The pipeline requires a pre-trained LightGBM model file. Default location:
```
model/v1_lgb_1125_fold3.pkl
```

## Input Data

### Required Columns

The input CSV must contain at minimum:
- `DELIVERY_ID`: Unique identifier for each delivery/order

### Recommended Columns

For best prediction accuracy, the dataset should include the following feature columns:

**Customer History Features:**
- `ORDER_COUNT_L12M`, `ORDER_COUNT_L90D`, `ORDER_COUNT_L28D`, `ORDER_COUNT_LIFETIME`
- `MTO_ORDER_COUNT_L7D`, `MTO_ORDER_COUNT_L28D`, `MTO_ORDER_COUNT_L90D`, `MTO_ORDER_COUNT_L12M`, `MTO_ORDER_COUNT_LIFETIME`
- `NEVER_DELIVERED_COUNT_L7D`, `NEVER_DELIVERED_COUNT_L28D`, `NEVER_DELIVERED_COUNT_L90D`, `NEVER_DELIVERED_COUNT_L12M`, `NEVER_DELIVERED_COUNT_LIFETIME`

**Fraud Risk Features:**
- `FRAUD_CNR_REQUEST_RATIO_L60D`, `FRAUD_CNR_REQUEST_RATIO_L180D`
- `FRAUD_CNR_APPROVED_REQUESTS_COUNT_L60D`, `FRAUD_CNR_APPROVED_REQUESTS_COUNT_L180D`
- `FRAUD_CNR_AMOUNT_L60D`, `FRAUD_CNR_AMOUNT_L180D`
- `ML_CX_CNR_RISK_V1_SCORE`

**Support History Features:**
- `SH_CNR`: Support history CNR flag
- `SH_IS_CREDITS`, `SH_IS_REFUND`, `SH_IS_REDELIVERY`, `SH_IS_REJET`
- `SH_FIRST_REPORT_ISSUE`, `SH_LATEST_REPORT_ISSUE`

**Order Features:**
- `DEFECT_CATEGORY`: Category of the issue (Never Delivered, Missing or Incorrect Items, etc.)
- `SUBTOTAL`, `TOTAL_ITEM_COUNT`, `PROMOTIONS`
- `SUBMIT_PLATFORM`

**Customer Value Features:**
- `AVG_VP_LIFETIME`, `AVG_VP_LIFETIME_CATEGORY`
- `AVG_SPEND_LIFETIME`, `AVG_SPEND_LIFETIME_CATEGORY`
- `IS_ELITE_CX`, `IS_TOP_95_PERCENT_VP`

Note: Missing columns will be filled with default values (0 for numeric, 'Unknown' for categorical).

## Output Data

The pipeline generates two output files by default:

### 1. Main Predictions File (`<output>.csv`)

Contains the final predictions with log-transformed features:
- `DELIVERY_ID`: Original delivery identifier
- All input features (log-transformed where applicable)
- `PREDICTED_ESCALATION_PROB`: Probability of escalation (0.0 to 1.0)
- `PREDICTED_ESCALATION`: Binary prediction (0 or 1) based on threshold

### 2. Non-Log Version (`<output>_non_log.csv`)

Contains the processed dataset BEFORE log transformations are applied:
- `DELIVERY_ID`: Original delivery identifier
- All input features in their original scale (not log-transformed)

**Why is the non-log version useful?**

The non-log version preserves the original scale of numeric features, which is useful for:
- **Feature analysis and interpretation**: Original values like `SUBTOTAL=$45.00` are more interpretable than log-transformed values like `SUBTOTAL=3.83`
- **Joining with other datasets**: Other datasets in your pipeline may use original (non-log) values
- **Debugging and validation**: Easier to verify data correctness with original values
- **Business reporting**: Original monetary and count values are more meaningful for stakeholders

**Data lineage for non-log version:**

```
Input CSV (dataset.csv)
    |
    v
Feature Engineering (IS_ND, IS_MnI, IS_PFQ, IS_OSI, IS_LATE, IS_WOD)
    |
    v
Handle Missing Values (numeric -> 0, categorical -> 'Unknown')
    |
    v
Convert Data Types (categorical -> str, numeric -> float64)
    |
    v
[SAVED HERE] --> <output>_non_log.csv (original feature scale)
    |
    v
Log Transformations (log1p for skewed, sign-preserving log for negative)
    |
    v
Model Predictions
    |
    v
[SAVED HERE] --> <output>.csv (log-transformed + predictions)
```

## Usage

### Basic Usage

```bash
python escalation_prediction_pipeline.py --input <input_csv> --output <output_csv>
```

This creates two files:
- `<output>.csv` - Predictions with log-transformed features
- `<output>_non_log.csv` - Dataset before log transformations

### With Custom Model Path

```bash
python escalation_prediction_pipeline.py \
    --input dataset/processed_dataset/dataset.csv \
    --output dataset/processed_dataset/predictions.csv \
    --model model/v1_lgb_1125_fold3.pkl
```

### With Custom Threshold

```bash
python escalation_prediction_pipeline.py \
    --input data.csv \
    --output predictions.csv \
    --threshold 0.3
```

### Without Saving Non-Log Version

If you do not need the non-log version:

```bash
python escalation_prediction_pipeline.py \
    --input data.csv \
    --output predictions.csv \
    --no-save-non-log
```

### Quiet Mode (Minimal Output)

```bash
python escalation_prediction_pipeline.py \
    --input data.csv \
    --output predictions.csv \
    --quiet
```

## Command Line Arguments

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| --input | -i | Yes | - | Path to input CSV dataset |
| --output | -o | Yes | - | Path to output CSV file |
| --model | -m | No | model/v1_lgb_1125_fold3.pkl | Path to pre-trained model |
| --threshold | -t | No | 0.5 | Classification threshold |
| --quiet | -q | No | False | Run in quiet mode |
| --no-save-non-log | - | No | False | Skip saving the non-log version |

## Examples

### Example 1: Standard Prediction Run

```bash
cd /path/to/NegotiatonAgent
python src/scripts/pipelines/escalation_prediction_pipeline.py \
    --input dataset/processed_dataset/dataset.csv \
    --output dataset/processed_dataset/dataset_predictions.csv
```

This creates:
- `dataset/processed_dataset/dataset_predictions.csv`
- `dataset/processed_dataset/dataset_predictions_non_log.csv`

### Example 2: With Lower Threshold for Higher Recall

```bash
python src/scripts/pipelines/escalation_prediction_pipeline.py \
    --input dataset/processed_dataset/dataset.csv \
    --output results/high_recall_predictions.csv \
    --threshold 0.3
```

### Example 3: Only Predictions (No Non-Log File)

```bash
python src/scripts/pipelines/escalation_prediction_pipeline.py \
    --input dataset/processed_dataset/dataset.csv \
    --output results/predictions.csv \
    --no-save-non-log
```

### Example 4: Batch Processing

```bash
for file in dataset/batches/*.csv; do
    output="results/predictions_$(basename $file)"
    python src/scripts/pipelines/escalation_prediction_pipeline.py \
        --input "$file" \
        --output "$output" \
        --quiet
done
```

## Pipeline Output Example

```
============================================================
ESCALATION PREDICTION PIPELINE
============================================================
Started at: 2025-01-13 10:30:00

------------------------------------------------------------
Step 1: Loading Dataset
------------------------------------------------------------
Loading dataset from: dataset/processed_dataset/dataset.csv
Dataset shape: (1414041, 195)
Columns: 195

------------------------------------------------------------
Step 2: Feature Engineering
------------------------------------------------------------
Created defect category features:
  - IS_ND (Never Delivered): 39,357 cases
  - IS_MnI (Missing/Incorrect): 78,908 cases
  ...

------------------------------------------------------------
Step 5: Applying Log Transformations
------------------------------------------------------------
Saved non-log version to: dataset/processed_dataset/dataset_predictions_non_log.csv
  (Contains original feature values before log transformation)
Identified 52 skewed features for log1p transformation
Identified 5 features with negative values
Applying log transformations...
  Applied log1p to 52 features
  Applied sign-preserving log to 5 features

...

============================================================
PIPELINE SUMMARY
============================================================
Total samples processed: 1,414,041
Features used: 48

Prediction Statistics:
  - Mean probability: 0.1323
  - Median probability: 0.0856
  - Min: 0.0016
  - Max: 0.8534

Predicted Escalations (threshold=0.5):
  - Escalated: 13,131
  - Not Escalated: 1,400,910
  - Predicted escalation rate: 0.93%

Output Files:
  - Predictions (log-transformed): dataset/processed_dataset/dataset_predictions.csv
  - Non-log version: dataset/processed_dataset/dataset_predictions_non_log.csv

Completed at: 2025-01-13 10:32:15
```

## Notes

1. The model was trained on a specific feature set. Features not seen during training will be handled with default values.

2. The prediction probability represents the likelihood that a case will escalate to a human agent or require additional intervention.

3. For production use, consider adjusting the threshold based on your precision/recall requirements:
   - Lower threshold (0.3): Higher recall, more false positives
   - Higher threshold (0.7): Higher precision, more false negatives

4. Large datasets may require significant memory. For datasets over 1M rows, ensure at least 8GB of available RAM.

5. The non-log version file can be large for big datasets. Use `--no-save-non-log` if disk space is a concern and you do not need the non-log version.

## Troubleshooting

### Model Not Found Error

If you see "Model file not found", ensure the model path is correct:
```bash
python escalation_prediction_pipeline.py --input data.csv --output out.csv --model /full/path/to/model.pkl
```

### Memory Error

For large datasets, process in batches or increase system memory.

### Feature Mismatch Warning

If you see warnings about missing features, the pipeline will use default values. For best results, ensure your dataset contains the recommended columns listed above.

### Non-Log File Not Created

Ensure you are not using the `--no-save-non-log` flag. By default, the non-log version is always saved.
