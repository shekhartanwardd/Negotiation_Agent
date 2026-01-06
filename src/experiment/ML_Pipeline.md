# ML Pipeline Documentation

## Overview

This document describes the machine learning pipeline implemented in `src/experiment/escalation_model_ml_pipeline.ipynb` for predicting customer escalation probability.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ESCALATION MODEL ML PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. LOAD DATASET                                                            │
│     Input: dataset/processed_dataset/dataset.csv                            │
│     Shape: 1,414,041 rows × 195 columns                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. FEATURE ENGINEERING                                                     │
│     Create binary indicators from DEFECT_CATEGORY                           │
│     (IS_ND, IS_MnI, IS_PFQ, IS_OSI, IS_LATE, IS_WOD)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. DATA PREPROCESSING                                                      │
│     - Handle missing values (numeric → 0, categorical → 'Unknown')          │
│     - Convert data types (categorical → str, numeric → float64)             │
│     - Apply log transformations to skewed features                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. LOAD MODEL                                                              │
│     Model: model/v1_lgb_1125_fold3.pkl                                      │
│     Type: LightGBM Booster                                                  │
│     Features: 48                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. INFERENCE                                                               │
│     - Encode categorical features with LabelEncoder                         │
│     - Generate escalation probability predictions                           │
│     - Apply threshold (0.5) for binary classification                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  6. OUTPUT                                                                  │
│     Output: dataset/processed_dataset/dataset_predictions.csv               │
│     Columns: DELIVERY_ID, features, PREDICTED_ESCALATION_PROB,              │
│              PREDICTED_ESCALATION                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Input Files

| File | Path | Description |
|------|------|-------------|
| Dataset | `dataset/processed_dataset/dataset.csv` | Processed dataset with 1.4M+ rows and 195 columns |
| Model | `model/v1_lgb_1125_fold3.pkl` | Pre-trained LightGBM Booster model |

---

## Pipeline Steps

### Step 1: Load Dataset

```python
DATASET_PATH = 'dataset/processed_dataset/dataset.csv'
df = pd.read_csv(DATASET_PATH, low_memory=False)
```

**Output Statistics:**
- Total rows: 1,414,041
- Total columns: 195
- Escalation rate (actual): 38.62%

---

### Step 2: Feature Engineering

Creates binary indicator features from `DEFECT_CATEGORY`:

| Feature | Description | Count |
|---------|-------------|-------|
| `IS_ND` | Never Delivered | 39,357 |
| `IS_MnI` | Missing or Incorrect Items | 78,908 |
| `IS_PFQ` | Order Quality Issue (Poor Food Quality) | 51,102 |
| `IS_OSI` | Order Status Inquiry | 155,767 |
| `IS_LATE` | Delivery Too Late / Early | 23,334 |
| `IS_WOD` | Wrong Order Received | 13,120 |

---

### Step 3: Data Preprocessing

#### 3.1 Handle Missing Values

| Feature Type | Strategy | Example |
|--------------|----------|---------|
| Numeric | Fill with `0` | `SH_CNR`, `MTO_ORDER_COUNT_L90D` |
| Categorical | Fill with `'Unknown'` | `SH_FIRST_REPORT_ISSUE`, `MOST_FREQ_MTO_ISSUE` |

**Key features with missing values:**
- `SH_IS_CREDITS`: 85.2% nulls → filled with 0
- `SH_FIRST_REPORT_ISSUE`: 85.2% nulls → filled with 'Unknown'
- `MTO_ORDER_COUNT_L90D`: 33.7% nulls → filled with 0

#### 3.2 Data Type Conversion

| Type | Conversion | Count |
|------|------------|-------|
| Categorical | → `str` | 10 features |
| Numeric | → `float64` | 60 features |

**Categorical Features:**
- `DEFECT_CATEGORY`
- `SH_FIRST_REPORT_ISSUE`
- `SH_LATEST_REPORT_ISSUE`
- `DEFAULT_ZIP_CODE`
- `IS_TOP_95_PERCENT_VP`
- `AVG_SPEND_LIFETIME_CATEGORY`
- `AVG_VP_LIFETIME_CATEGORY`
- `MOST_FREQ_MTO_ISSUE`
- `LATEST_MTO_ISSUE`
- `SUBMIT_PLATFORM`

#### 3.3 Log Transformations

Applied to reduce skewness in numeric features:

| Transformation | Description | Features Applied |
|----------------|-------------|------------------|
| `log1p` | Standard log(1+x) | 52 features |
| Sign-preserving log | sign(x) × log(1+\|x\|) | 5 features (with negative values) |

---

### Step 4: Load Pre-trained Model

```python
MODEL_PATH = 'model/v1_lgb_1125_fold3.pkl'

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
```

**Model Specifications:**
- Type: LightGBM `Booster`
- Features expected: 48
- Training configuration: Fold 3 cross-validation

**Model Features (48 total):**
```
SH_LATEST_REPORT_ISSUE, TOTAL_ITEM_COUNT, SH_FIRST_REPORT_ISSUE, 
PAYMENT_METHOD, DEFECT_CATEGORY, SH_CNR, FRAUD_CNR_APPROVED_REQUESTS_COUNT_L60D,
SH_IS_CREDITS, SUBTOTAL, TIP, ... and 38 more
```

---

### Step 5: Generate Predictions

The prediction process handles categorical feature encoding to avoid LightGBM compatibility issues:

```python
# Encode categorical features
from sklearn.preprocessing import LabelEncoder
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Convert to numpy array (bypasses pandas categorical mismatch)
X_array = X.values.astype(np.float64)

# Generate predictions
pred_probs = model.predict(X_array)
```

**Prediction Output:**
- Probability range: [0.0016, 0.8534]
- Mean prediction: 0.1323

---

### Step 6: Output Results

Two new columns are added:

| Column | Description |
|--------|-------------|
| `PREDICTED_ESCALATION_PROB` | Continuous probability [0, 1] |
| `PREDICTED_ESCALATION` | Binary prediction (threshold = 0.5) |

**Output File:**
```
dataset/processed_dataset/dataset_predictions.csv
```

**Prediction Distribution:**
| Prediction | Count | Percentage |
|------------|-------|------------|
| Not Escalated (0) | 1,400,910 | 99.07% |
| Escalated (1) | 13,131 | 0.93% |

---

## Feature Configuration

### Final Features List (70 Features)

The pipeline uses 70 features organized into categories:

```python
FINAL_FEATURES = [
    # Self-Help Features
    'SH_CNR', 'SH_IS_CREDITS', 'SH_IS_REFUND', 'SH_IS_REDELIVERY',
    'SH_FIRST_REPORT_ISSUE', 'SH_LATEST_REPORT_ISSUE', 'SH_IS_REJET',
    
    # Defect Category Features
    'DEFECT_CATEGORY', 'IS_MnI', 'IS_OSI', 'IS_ND', 'IS_PFQ',
    
    # MTO (Manual Takeover) Features
    'MTO_ORDER_COUNT_L7D', 'MTO_ORDER_COUNT_L28D', 'MTO_ORDER_COUNT_L90D',
    'MTO_ORDER_COUNT_L12M', 'MTO_ORDER_COUNT_LIFETIME',
    'MOST_FREQ_MTO_COUNT', 'MOST_FREQ_MTO_ISSUE', 'LATEST_MTO_ISSUE',
    
    # Fraud/Risk Features
    'FRAUD_CNR_REQUEST_RATIO_L60D', 'FRAUD_CNR_REQUEST_RATIO_L180D',
    'FRAUD_CNR_APPROVED_REQUESTS_COUNT_L60D', 'FRAUD_CNR_APPROVED_REQUESTS_COUNT_L180D',
    'FRAUD_CNR_AMOUNT_L60D', 'FRAUD_CNR_AMOUNT_L180D',
    'ML_CX_CNR_RISK_V1_SCORE',
    
    # Order/Delivery Metrics
    'ORDER_COUNT_L28D', 'ORDER_COUNT_L90D', 'ORDER_COUNT_L12M', 'ORDER_COUNT_LIFETIME',
    'NEVER_DELIVERED_COUNT_L7D', 'NEVER_DELIVERED_COUNT_L28D',
    'NEVER_DELIVERED_COUNT_L90D', 'NEVER_DELIVERED_COUNT_L12M',
    
    # Customer Value Features
    'AVG_VP_LIFETIME', 'AVG_SPEND_LIFETIME', 'IS_ELITE_CX',
    
    # And more...
]
```

---

## Model Evaluation

When labels (`IS_ESCALATED`) are available, the pipeline computes:

| Metric | Value |
|--------|-------|
| ROC AUC | Computed on test set |
| PR AUC | Computed on test set |

**Threshold Analysis:**
| Threshold | Precision | Recall |
|-----------|-----------|--------|
| 0.3 | Varies | Varies |
| 0.4 | Varies | Varies |
| 0.5 | Varies | Varies |
| 0.6 | Varies | Varies |
| 0.7 | Varies | Varies |
| 0.8 | Varies | Varies |

---

## Intermediate Outputs

The pipeline also saves intermediate files:

| File | Description |
|------|-------------|
| `dataset/processed_dataset/dataset_non_log.csv` | Dataset before log transformations |
| `dataset/processed_dataset/dataset_predictions.csv` | Final predictions with selected features |

---

## Usage

### Running the Pipeline

1. Open the notebook: `src/experiment/escalation_model_ml_pipeline.ipynb`
2. Ensure paths are configured correctly:
   ```python
   DATASET_PATH = 'dataset/processed_dataset/dataset.csv'
   MODEL_PATH = 'model/v1_lgb_1125_fold3.pkl'
   ```
3. Run all cells sequentially

### Requirements

```python
# Standard library
from datetime import datetime, timedelta
import math, warnings, os, pickle, json

# Data manipulation
import numpy as np
import pandas as pd

# Machine Learning
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, auc
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## Key Technical Notes

1. **Categorical Feature Handling**: The model expects categorical features in a specific format. The pipeline converts them to strings and uses `LabelEncoder` to avoid "train and valid dataset categorical_feature do not match" errors.

2. **Missing Features**: If a feature expected by the model is missing from the dataset, the pipeline creates it with default values (0 for numeric, 'Unknown' for categorical).

3. **Log Transformations**: Applied to 52 skewed features using `log1p(x)`, and 5 features with negative values using sign-preserving log: `sign(x) × log1p(|x|)`.

4. **Numpy Array Conversion**: The feature matrix is converted to a numpy array before prediction to bypass pandas DataFrame categorical compatibility checks.

---

## Pipeline Summary

| Step | Description | Status |
|------|-------------|--------|
| 1 | Load dataset from CSV | ✅ |
| 2 | Create derived features (IS_ND, IS_MnI, etc.) | ✅ |
| 3 | Handle missing values | ✅ |
| 4 | Convert data types for LightGBM | ✅ |
| 5 | Apply log transformations to skewed features | ✅ |
| 6 | Load pre-trained model (v1_lgb_1125_fold3.pkl) | ✅ |
| 7 | Generate escalation predictions | ✅ |
| 8 | Evaluate model performance (if labels available) | ✅ |
| 9 | Save predictions to CSV | ✅ |

