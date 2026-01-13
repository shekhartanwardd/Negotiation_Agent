# Escalation Payload Generation Pipeline

This script generates JSON payloads for cases predicted to escalate to human agents. These payloads are used as input to GPT for automated negotiation and resolution.

## Overview

The pipeline performs the following steps:
1. Load datasets (predictions, non-log, original)
2. Merge datasets to create a unified dataset with all required features
3. Extract chatbot conversation portion (CONVERSATION_CB)
4. Extract apology credits from conversations (Extracted_AC)
5. Filter cases based on escalation probability threshold
6. Generate JSON payloads for escalated cases
7. Optionally generate JSON payloads for non-escalated cases

## Input Dependencies

This script requires output files from the `escalation_prediction_pipeline.py` script:

| File | Description | Source |
|------|-------------|--------|
| `dataset_predictions.csv` | Log-transformed features with escalation predictions | Output of `escalation_prediction_pipeline.py` |
| `dataset_non_log.csv` | Original feature scale (before log transformation) | Output of `escalation_prediction_pipeline.py` |
| `dataset.csv` | Original dataset with CONVERSATION and Parsed_AC columns | Input to `escalation_prediction_pipeline.py` |

Default locations (relative to project root):
- `dataset/processed_dataset/dataset_predictions.csv`
- `dataset/processed_dataset/dataset_non_log.csv`
- `dataset/processed_dataset/dataset.csv`

## Escalation Threshold

The threshold (default 0.5) determines which cases are predicted to escalate to a human agent.

- Cases with `PREDICTED_ESCALATION_PROB >= threshold` are considered "escalated"
- Only escalated cases are sent to GPT for automated resolution
- This optimizes API costs by focusing on cases that would otherwise require human intervention

Threshold selection guidelines:
- Lower threshold (e.g., 0.3): More cases are flagged as escalated (higher recall, more API calls)
- Higher threshold (e.g., 0.7): Fewer cases are flagged (higher precision, fewer API calls)
- Default threshold (0.5): Balanced approach

## Output Files

### Escalated Payloads

Location: `dataset/escalated_payload/escalated_cases_payload.json`

JSON array containing payloads for cases predicted to escalate:

```json
[
  {
    "DELIVERY_ID": 12345678.0,
    "CONVERSATION_CB": "Chatbot: Hi, I'm your DoorDash virtual assistant...",
    "IS_CNR_ABUSER": 0.123,
    "Parsed_AC": 0.0,
    "Extracted_AC": 10.0,
    "ORDER_SUBTOTAL": 45.99,
    "IS_VIP_CUSTOMER": 1,
    "ISSUE_COUNT_LAST_10_ORDERS": 2,
    "ISSUE_COUNT_LAST_10_DAYS": 1,
    "PREDICTED_ESCALATION_PROB": 0.65,
    "SH_CNR": 0
  },
  ...
]
```

### Non-Escalated Payloads (Optional)

Location: `dataset/non_escalated_payload/non_escalated_cases_payload.json`

Same structure as escalated payloads, but for cases with `PREDICTED_ESCALATION_PROB < threshold`.

## Payload Fields

| Field | Description | Source Column |
|-------|-------------|---------------|
| DELIVERY_ID | Unique delivery identifier | DELIVERY_ID |
| CONVERSATION_CB | Chatbot portion of conversation (before human agent) | Derived from CONVERSATION |
| IS_CNR_ABUSER | CNR abuse risk score | ML_CX_CNR_RISK_V1_SCORE |
| Parsed_AC | Apology credits parsed from conversation | Parsed_AC |
| Extracted_AC | Apology credits extracted using enhanced patterns | Derived from CONVERSATION |
| ORDER_SUBTOTAL | Order subtotal amount | SUBTOTAL |
| IS_VIP_CUSTOMER | VIP/Elite customer flag | IS_ELITE_CX |
| ISSUE_COUNT_LAST_10_ORDERS | Credit/refund count in last 28 days | CREDIT_REFUND_ORDER_COUNT_L28D |
| ISSUE_COUNT_LAST_10_DAYS | Credit/refund count in last 12 months | CREDIT_REFUND_ORDER_COUNT_L12M |
| PREDICTED_ESCALATION_PROB | Model prediction probability | PREDICTED_ESCALATION_PROB |
| SH_CNR | Support history CNR flag | SH_CNR |

## Requirements

### Python Dependencies

```
numpy
pandas
```

Install dependencies:
```bash
pip install numpy pandas
```

## Usage

### Basic Usage (Default Threshold 0.5)

```bash
python generate_escalation_payload.py
```

### With Custom Threshold

```bash
# Lower threshold = more cases flagged as escalated
python generate_escalation_payload.py --threshold 0.3

# Higher threshold = fewer cases flagged
python generate_escalation_payload.py --threshold 0.7
```

### Include Non-Escalated Payloads

```bash
python generate_escalation_payload.py --include-non-escalated
```

### With Custom Paths

```bash
python generate_escalation_payload.py \
    --predictions /path/to/dataset_predictions.csv \
    --non-log /path/to/dataset_non_log.csv \
    --original /path/to/dataset.csv \
    --output-dir /path/to/output
```

### Quiet Mode

```bash
python generate_escalation_payload.py --quiet
```

## Command Line Arguments

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| --predictions | - | No | dataset/processed_dataset/dataset_predictions.csv | Path to predictions CSV |
| --non-log | - | No | dataset/processed_dataset/dataset_non_log.csv | Path to non-log CSV |
| --original | - | No | dataset/processed_dataset/dataset.csv | Path to original dataset |
| --output-dir | - | No | dataset/escalated_payload | Output directory for escalated payloads |
| --threshold | -t | No | 0.5 | Escalation probability threshold |
| --include-non-escalated | - | No | False | Also generate non-escalated payloads |
| --non-escalated-output-dir | - | No | dataset/non_escalated_payload | Output directory for non-escalated payloads |
| --quiet | -q | No | False | Run in quiet mode |

## Complete Workflow

### Step 1: Run Escalation Prediction Pipeline

First, run the escalation prediction pipeline to generate the required input files:

```bash
python src/scripts/pipelines/escalation_prediction_pipeline.py \
    --input dataset/processed_dataset/dataset.csv \
    --output dataset/processed_dataset/dataset_predictions.csv
```

This creates:
- `dataset_predictions.csv` (log-transformed features + predictions)
- `dataset_predictions_non_log.csv` (original feature scale)

Note: Rename `dataset_predictions_non_log.csv` to `dataset_non_log.csv` or update the path argument.

### Step 2: Run Payload Generation Pipeline

Then, run the payload generation pipeline:

```bash
python src/scripts/pipelines/generate_escalation_payload.py --threshold 0.5
```

This creates:
- `dataset/escalated_payload/escalated_cases_payload.json`

### Step 3: Use Payloads with GPT

The generated payloads can now be used as input to GPT for automated negotiation:

```python
import json

with open('dataset/escalated_payload/escalated_cases_payload.json', 'r') as f:
    payloads = json.load(f)

for payload in payloads:
    # Call GPT with payload
    response = call_gpt(payload)
    # Process response...
```

## Pipeline Output Example

```
============================================================
ESCALATION PAYLOAD GENERATION PIPELINE
============================================================
Started at: 2025-01-13 14:30:00
Escalation threshold: 0.5

============================================================
Loading Datasets
============================================================
Loading: dataset/processed_dataset/dataset_predictions.csv
  Shape: (1414041, 72)
Loading: dataset/processed_dataset/dataset_non_log.csv
  Shape: (1414041, 70)
Loading: dataset/processed_dataset/dataset.csv
  Shape: (1414041, 195)

------------------------------------------------------------
Merging Datasets
------------------------------------------------------------
  Merged dataset shape: (1414041, 15)
  Final dataset columns: 15

------------------------------------------------------------
Extracting Chatbot Conversation
------------------------------------------------------------
Parsing 1,414,041 conversations for chatbot portion...
  Created column 'CONVERSATION_CB'
  - Non-empty conversations: 1,414,041
  - Conversations truncated (had agent transfer): 209,059

------------------------------------------------------------
Extracting Apology Credits
------------------------------------------------------------
Extracting apology credits from 1,414,041 conversations...
  Created column 'Extracted_AC'
  - Rows with extracted AC > 0: 51,540 (3.6%)
  - Total extracted amount: $512,340.00

------------------------------------------------------------
Filtering Escalated Cases
------------------------------------------------------------
Using threshold: PREDICTED_ESCALATION_PROB >= 0.5
  Escalated cases: 13,131 (0.93%)
  Non-escalated cases: 1,400,910

------------------------------------------------------------
Generating Escalated Payloads
------------------------------------------------------------
Creating payloads for 13,131 cases...
  Created 13,131 payloads
  Saved to: dataset/escalated_payload/escalated_cases_payload.json

============================================================
PIPELINE SUMMARY
============================================================
Total records processed: 1,414,041
Escalation threshold: 0.5
Escalated cases: 13,131 (0.93%)

Output Files:
  - Escalated payloads: dataset/escalated_payload/escalated_cases_payload.json

Completed at: 2025-01-13 14:32:15
```

## Notes

1. The Extracted_AC column uses enhanced pattern matching to find apology credits mentioned in conversations. This may capture credits that Parsed_AC missed.

2. CONVERSATION_CB contains only the chatbot portion of the conversation (before human agent transfer). This is typically more relevant for GPT-based resolution.

3. Large datasets may take several minutes to process due to regex-based extraction.

4. The threshold should be tuned based on your precision/recall requirements and API budget.

## Troubleshooting

### File Not Found Error

Ensure the input files exist. Run `escalation_prediction_pipeline.py` first if needed:

```bash
python src/scripts/pipelines/escalation_prediction_pipeline.py \
    --input dataset/processed_dataset/dataset.csv \
    --output dataset/processed_dataset/dataset_predictions.csv
```

### Memory Error

For very large datasets, consider processing in batches or increasing system memory.

### Empty Payload Output

If no payloads are generated, check:
1. The threshold may be too high (no cases meet the criteria)
2. The input files may be empty or incorrectly formatted
3. PREDICTED_ESCALATION_PROB column may be missing from predictions

