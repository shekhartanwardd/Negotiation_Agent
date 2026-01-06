# Async Evaluation Pipeline

This document describes the end-to-end pipeline for running async evaluation of the Apology Credit Engine, from payload creation to full dataset evaluation.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ASYNC EVALUATION PIPELINE                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: PAYLOAD CREATION                                                   │
│  Script: process_payload_files.py                                           │
│                                                                             │
│  • Load escalated/non-escalated JSON payload files                          │
│  • Sample 50 examples based on conversation length buckets                  │
│  • Create Parsed_AC_Bucket for ground truth                                 │
│  • Generate ground truth JSON files                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: ASYNC EVALUATION                                                   │
│  Script: run_full_evaluation_async.py                                       │
│                                                                             │
│  • Load CSV dataset with payloads                                           │
│  • Filter by escalation threshold                                           │
│  • Run async GPT inference for eligible cases                               │
│  • Apply ceiling logic to credit recommendations                            │
│  • Save results with statistics                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Payload Creation

### Script: `process_payload_files.py`

This script processes raw payload files to create sampled datasets and ground truth for evaluation.

### Input Files

| File | Path |
|------|------|
| Escalated Cases | `dataset/escalated_payload/escalated_cases_payload.json` |
| Non-Escalated Cases | `dataset/non_escalated_payload/non_escalated_cases_payload.json` |

### Processing Steps

#### 1.1 Conversation Length Bucketing

Records are organized into buckets based on `CONVERSATION_CB` length:

| Bucket | Length Range | Description |
|--------|--------------|-------------|
| `very_short` | 0 - 500 chars | Very brief conversations |
| `short` | 500 - 750 chars | Short conversations |
| `medium_short` | 750 - 1000 chars | Medium-short conversations |
| `medium` | 1000 - 1500 chars | Medium conversations |
| `medium_long` | 1500 - 2000 chars | Medium-long conversations |
| `long` | 2000 - 2500 chars | Long conversations |
| `very_long` | 2500+ chars | Very long conversations |

#### 1.2 Proportional Sampling

- Samples **50 records** from each payload file
- Uses **proportional allocation** based on bucket sizes
- Ensures at least 1 sample per non-empty bucket
- Random seed: `42` for reproducibility

#### 1.3 Parsed_AC_Bucket Assignment

Assigns apology credit values to buckets:

| Bucket | Amount Range |
|--------|--------------|
| `$0` | Exactly $0 |
| `$1-5` | $1 to $5 |
| `$6-10` | $6 to $10 |
| `$11-15` | $11 to $15 |
| `$16-20` | $16 to $20 |
| ... | ... |
| `$100+` | Greater than $100 |

### Output Files

| File | Path | Description |
|------|------|-------------|
| Ground Truth (Escalated) | `dataset/ground_truth_dataset/ground_truth_escalated.json` | Sampled escalated cases with `DELIVERY_ID` and `Parsed_AC_Bucket` |
| Ground Truth (Non-Escalated) | `dataset/ground_truth_dataset/ground_truth_non_escalated.json` | Sampled non-escalated cases |

### Usage

```bash
cd src/scripts
python process_payload_files.py
```

### Example Output

```
======================================================================
PAYLOAD FILE PROCESSING
======================================================================
Random seed set to: 42
Samples per file: 50

======================================================================
Processing ESCALATED cases
======================================================================
Loading data from: .../escalated_cases_payload.json
Loaded 1,234 records

Bucket Distribution for escalated_cases_payload.json:
============================================================
  very_short      | Count:    150 | Length:   50 -   499
  short           | Count:    280 | Length:  501 -   749
  medium          | Count:    320 | Length: 1001 -  1499
  ...

Sampling Plan:
============================================================
  very_short      | Samples:   6 /   150
  short           | Samples:  11 /   280
  ...
```

---

## Step 2: Async Evaluation

### Script: `run_full_evaluation_async.py`

This script runs the Apology Credit Engine on large datasets using asynchronous API calls for high throughput.

### Features

- **Async Processing**: Uses `aiohttp` for concurrent API requests
- **Escalation Threshold**: Skips GPT inference for low-escalation cases
- **Ceiling Logic**: Converts credit ranges to ceiling values
- **Progress Tracking**: Real-time progress with `tqdm`
- **Error Handling**: Graceful handling of timeouts and API errors

### Input Files

| File | Path | Description |
|------|------|-------------|
| Dataset | `dataset/processed_dataset/dataset_final_escalated_cases.csv` | CSV with case payloads |
| System Prompt | `system_prompt/prompt_v3.txt` | GPT system prompt |
| Config | `src/scripts/config.json` | API credentials |

### Configuration File (`config.json`)

```json
{
  "api_key": "your-portkey-api-key",
  "virtual_key": "your-virtual-key"
}
```

### Payload Transformation

The script transforms raw CSV data into model-ready payloads:

| Field | Transformation |
|-------|----------------|
| `DELIVERY_ID` | Convert to string |
| `CONVERSATION_CB` | Keep as-is (string) |
| `IS_CNR_ABUSER` | Threshold at 0.5 → boolean |
| `ORDER_SUBTOTAL` | Round to 2 decimal places |
| `IS_VIP_CUSTOMER` | Convert to boolean |
| `ISSUE_COUNT_LAST_10_ORDERS` | Round to integer |
| `ISSUE_COUNT_LAST_10_DAYS` | Round to integer |
| `PREDICTED_ESCALATION_PROB` | Keep as float |
| `SH_CNR` | Max(0, value), round to 2 decimals |

### Escalation Threshold Logic

Cases are filtered based on `PREDICTED_ESCALATION_PROB`:

```
if escalation_prob < threshold (default: 0.2):
    → Skip GPT inference
    → Return $0 credit recommendation
else:
    → Run GPT inference
    → Return credit recommendation
```

### Ceiling Logic

Credit ranges are converted to ceiling values:

| Credit Range | Ceiling Value |
|--------------|---------------|
| `$0` | $0 |
| `$1-$5` | $5 |
| `$6-$10` | $10 |
| `$11-$15` | $15 |
| ... | Upper bound |

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv-file` | `dataset/.../dataset_final_escalated_cases.csv` | Input CSV path |
| `--prompt-file` | `system_prompt/prompt_v3.txt` | System prompt path |
| `--model` | `gpt-4.1-2025-04-14` | Model name |
| `--run-version` | `4` | Run version for output directory |
| `--escalation-threshold` | `0.2` | Threshold for GPT inference |
| `--limit` | `-1` | Limit cases (-1 for all) |
| `--output-base` | Project root | Base output path |
| `--max-concurrent` | `20` | Max concurrent API requests |
| `--timeout` | `120` | Request timeout (seconds) |

### Usage

#### Basic Run
```bash
cd src/scripts
python run_full_evaluation_async.py
```

#### With Custom Parameters
```bash
python run_full_evaluation_async.py \
    --csv-file /path/to/dataset.csv \
    --prompt-file /path/to/prompt.txt \
    --model gpt-4.1-2025-04-14 \
    --escalation-threshold 0.3 \
    --max-concurrent 30 \
    --limit 1000 \
    --run-version 5
```

#### Limit to First 100 Cases
```bash
python run_full_evaluation_async.py --limit 100
```

#### Higher Concurrency
```bash
python run_full_evaluation_async.py --max-concurrent 50
```

### Output

Results are saved to `results/run{version}/async_evaluation_{timestamp}.json`:

```json
{
  "metadata": {
    "csv_file": "...",
    "prompt_file": "...",
    "model": "gpt-4.1-2025-04-14",
    "escalation_threshold": 0.2,
    "max_concurrent": 20,
    "total_cases": 10000,
    "gpt_inference": 2500,
    "threshold_skip": 7500,
    "failures": 5,
    "elapsed_time_seconds": 180.5,
    "timestamp": "20251230_130606"
  },
  "summary": {
    "total_credit_ceiling": 25000.00,
    "credit_distribution": {
      "$0": 7505,
      "$1-$5": 1200,
      "$6-$10": 800,
      ...
    }
  },
  "results": [...],
  "failures": [...]
}
```

### Example Console Output

```
======================================================================
ASYNC FULL DATASET EVALUATION - APOLOGY CREDIT ENGINE
======================================================================

Loading dataset from: .../dataset_final_escalated_cases.csv
Total rows in dataset: 511224
Expected GPT calls: 102245
Expected threshold skips: 408979

Model: gpt-4.1-2025-04-14
Escalation Threshold: 0.2
Max Concurrent Requests: 20
Timeout: 120s

Starting evaluation of 511224 cases...

Evaluating: 100%|██████████████████████| 511224/511224 [45:23<00:00, 187.6it/s]

======================================================================
EVALUATION COMPLETE
======================================================================

Total Cases Processed: 511224
  - Successful: 511200
  - GPT Inference: 102200
  - Threshold Skip ($0): 409000
  - Failures: 24

Elapsed Time: 2723.5 seconds (45.4 minutes)
Rate: 187.6 cases/second

============================================================
TOTAL APOLOGY CREDIT (Ceiling Logic): $1,234,567.00
============================================================

Credit Distribution:
  $0: 409000 cases -> $0 each = $0.00
  $1-$5: 45000 cases -> $5 each = $225,000.00
  $6-$10: 32000 cases -> $10 each = $320,000.00
  ...

Results saved to: results/run4/async_evaluation_20251230_130606.json
======================================================================
```

---

## Requirements

### Python Packages

```txt
aiohttp>=3.8.0
pandas>=1.5.0
tqdm>=4.64.0
```

### Environment

- Python 3.8+
- Valid API credentials in `config.json`
- Network access to API endpoint

---

## File Structure

```
src/scripts/
├── README.md                      # This file
├── config.json                    # API credentials (not in git)
├── process_payload_files.py       # Payload creation script
├── run_full_evaluation_async.py   # Async evaluation script
├── run_full_evaluation.py         # Sync evaluation script (legacy)
├── apology_credit_engine.py       # Core engine module
└── parse_apology_credits.py       # Credit parsing utilities
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `config.json not found` | Create `config.json` with API credentials |
| API timeout errors | Increase `--timeout` or decrease `--max-concurrent` |
| Memory issues with large datasets | Use `--limit` to process in batches |
| Rate limiting | Decrease `--max-concurrent` |

### Debugging

Run with limited cases to test:
```bash
python run_full_evaluation_async.py --limit 10
```

---

## Performance Tips

1. **Optimal Concurrency**: Start with `--max-concurrent 20`, increase if API allows
2. **Threshold Tuning**: Higher `--escalation-threshold` reduces GPT calls but may miss cases
3. **Batch Processing**: For very large datasets, use `--limit` to process in chunks
4. **Network**: Ensure stable network connection for async requests

---

## Pipeline Summary

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `process_payload_files.py` | Raw JSON payloads | Sampled ground truth JSON |
| 2 | `run_full_evaluation_async.py` | CSV dataset | Evaluation results JSON |

