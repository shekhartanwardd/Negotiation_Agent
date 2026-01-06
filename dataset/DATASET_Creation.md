# Dataset Creation Documentation

## Overview

This document describes the creation process for `dataset/processed_dataset/dataset.csv`, which serves as the primary dataset for the Negotiation Agent project.

---

## Source Files

The dataset is created by combining four original CSV files from the `dataset/original_dataset/` directory:

| File | Description |
|------|-------------|
| `Testing_escalated.csv` | Test set - escalated cases |
| `Testing_no_escalation.csv` | Test set - non-escalated cases |
| `Training_escalated.csv` | Training set - escalated cases |
| `Training_no_escalation.csv` | Training set - non-escalated cases |

---

## Creation Process

The dataset is generated using the notebook:
```
src/experiment/historical_dataset_analysis_report.ipynb
```

### Step 1: Load and Combine Datasets

```python
testing_escalated = pd.read_csv('dataset/original_dataset/Testing_escalated.csv')
testing_non_escalated = pd.read_csv('dataset/original_dataset/Testing_no_escalation.csv')
training_escalated = pd.read_csv('dataset/original_dataset/Training_escalated.csv')
training_non_escalated = pd.read_csv('dataset/original_dataset/Training_no_escalation.csv')

dataset = pd.concat([testing_escalated, testing_non_escalated, training_escalated, training_non_escalated])
```

### Step 2: Data Cleaning

- **Deduplication**: Removes duplicate rows based on `DELIVERY_ID`
- **Missing Value Handling**: Fills missing values in the following columns with `-1`:
  - `AGENT_ISSUED_AC`
  - `CHATBOT_ISSUED_AC`
  - `SH_ISSUED_AC`

### Step 3: Feature Engineering

#### Agent-Issued AC Flag
Creates `AGENT_ISSUED_AC_FLAG` column:
- Value `1`: Agent-issued AC is greater than 0 AND a multiple of 5
- Value `0`: Otherwise

#### Apology Credit Parsing
The `filter_apology_conversations()` function scans the `CONVERSATION` column to extract apology/additional credits using regex patterns:

**Patterns Matched:**
- `"$X as apology credits"` or `"$X as additional credits"`
- `"apology credits of $X"` or `"additional credits of $X"`
- `"issued $X as apology"` or `"processed $X as apology"`
- `"additional $X credits"`
- And similar variations

**New Columns Created:**
| Column | Description |
|--------|-------------|
| `ACTUAL_AC_CONVERSATION` | Binary flag: `1` if apology credit pattern found, `0` otherwise |
| `Parsed_AC` | Dollar amount extracted from conversation (last occurrence used) |

#### Human Agent Escalation Flag
Creates `CONVERSATION_HUMAN_AGENT` column:
- Value `1`: Conversation contains "Human Agent" text
- Value `0`: Otherwise

### Step 4: Save Dataset

```python
dataset.to_csv('dataset/processed_dataset/dataset.csv')
```

---

## Output Statistics

Based on the creation run:

| Metric | Value |
|--------|-------|
| Total Unique Delivery IDs | ~1,414,041 |
| Conversations with Apology Credits | ~3.6% |
| Cases Escalated to Human Agent | ~48.1% |
| Total Apology Credits Parsed | ~$627,954 |

---

## Key Columns Added

| Column | Type | Description |
|--------|------|-------------|
| `AGENT_ISSUED_AC_FLAG` | int | Flag for valid agent-issued AC (>0 and multiple of 5) |
| `ACTUAL_AC_CONVERSATION` | int | Binary flag for apology credit presence in conversation |
| `Parsed_AC` | float | Dollar amount of apology credit extracted from conversation |
| `CONVERSATION_HUMAN_AGENT` | int | Binary flag for human agent escalation |

---

## Usage

This dataset is used by:
- `src/experiment/escalation_model_ml_pipeline.ipynb` - ML pipeline for escalation prediction
- `src/experiment/gpt_negotiation_experiment.ipynb` - GPT-based negotiation experiments
- `src/experiment/escalation_model_analysis.ipynb` - Model analysis

---

## Notes

- The `Parsed_AC` column extracts the **last** apology credit amount found in the conversation, representing the final amount applied
- Random seed `42` is used when sampling is enabled for reproducibility

