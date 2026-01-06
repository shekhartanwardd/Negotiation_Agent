"""
Process Payload Files Script

This script performs the following steps:
1. Sample 50 examples from escalated and non-escalated payload files (based on CONVERSATION_CB length)
2. Create Parsed_AC_Bucket column based on Parsed_AC value
3. Remove Parsed_AC from the original JSON files
4. Create ground truth JSON files with DELIVERY_ID and Parsed_AC_Bucket

Usage:
    python process_payload_files.py

Output:
    - Modified escalated_cases_payload.json and non_escalated_cases_payload.json (Parsed_AC removed)
    - ground_truth_escalated.json and ground_truth_non_escalated.json (sampled ground truth data)
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Tuple, List, Dict

# Configuration
RANDOM_SEED = 42
TOTAL_SAMPLES = 50

# File paths
ESCALATED_INPUT = Path("/Users/shekhar.tanwar/Documents/Projects/NegotiatonAgent/dataset/escalated_payload/escalated_cases_payload.json")
NON_ESCALATED_INPUT = Path("/Users/shekhar.tanwar/Documents/Projects/NegotiatonAgent/dataset/non_escalated_payload/non_escalated_cases_payload.json")
GROUND_TRUTH_DIR = Path("/Users/shekhar.tanwar/Documents/Projects/NegotiatonAgent/dataset/ground_truth_dataset")

# Define conversation length buckets
LENGTH_BUCKETS = [
    (0, 500, "very_short"),
    (500, 750, "short"),
    (750, 1000, "medium_short"),
    (1000, 1500, "medium"),
    (1500, 2000, "medium_long"),
    (2000, 2500, "long"),
    (2500, float('inf'), "very_long")
]


def load_data(filepath: Path) -> List[Dict]:
    """Load the JSON payload file."""
    print(f"Loading data from: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data):,} records")
    return data


def save_data(data: List[Dict], filepath: Path) -> None:
    """Save data to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data):,} records to: {filepath}")


def get_conversation_length(record: Dict) -> int:
    """Extract the length of the conversation from a record."""
    conv = record.get('CONVERSATION_CB', '')
    if conv is None:
        return 0
    return len(str(conv))


def assign_length_bucket(length: int) -> str:
    """Assign a conversation to a bucket based on its length."""
    for min_len, max_len, bucket_name in LENGTH_BUCKETS:
        if min_len <= length < max_len:
            return bucket_name
    return LENGTH_BUCKETS[-1][2]


def create_buckets(data: List[Dict]) -> Dict[str, List[Dict]]:
    """Organize records into buckets based on conversation length."""
    buckets = defaultdict(list)
    for record in data:
        length = get_conversation_length(record)
        bucket_name = assign_length_bucket(length)
        buckets[bucket_name].append(record)
    return dict(buckets)


def calculate_samples_per_bucket(buckets: Dict[str, List[Dict]], total_samples: int) -> Dict[str, int]:
    """Calculate how many samples to take from each bucket using proportional allocation."""
    total_records = sum(len(records) for records in buckets.values())
    non_empty_buckets = {k: v for k, v in buckets.items() if len(v) > 0}
    
    samples_per_bucket = {}
    for bucket_name, records in non_empty_buckets.items():
        proportion = len(records) / total_records
        samples = max(1, round(proportion * total_samples))
        samples_per_bucket[bucket_name] = min(samples, len(records))
    
    # Adjust to match exactly total_samples
    current_total = sum(samples_per_bucket.values())
    
    if current_total > total_samples:
        sorted_buckets = sorted(samples_per_bucket.items(), key=lambda x: -x[1])
        for bucket_name, _ in sorted_buckets:
            if current_total <= total_samples:
                break
            if samples_per_bucket[bucket_name] > 1:
                samples_per_bucket[bucket_name] -= 1
                current_total -= 1
    elif current_total < total_samples:
        sorted_buckets = sorted(
            samples_per_bucket.items(),
            key=lambda x: len(buckets[x[0]]) - x[1],
            reverse=True
        )
        for bucket_name, _ in sorted_buckets:
            if current_total >= total_samples:
                break
            capacity = len(buckets[bucket_name]) - samples_per_bucket[bucket_name]
            if capacity > 0:
                add_samples = min(capacity, total_samples - current_total)
                samples_per_bucket[bucket_name] += add_samples
                current_total += add_samples
    
    return samples_per_bucket


def get_parsed_ac_bucket(parsed_ac: float) -> str:
    """
    Assign Parsed_AC value to a bucket.
    
    Buckets:
    - $0: exactly 0
    - $1-5: 1 to 5
    - $6-10: 6 to 10
    - $11-15: 11 to 15
    - etc.
    """
    if parsed_ac is None:
        return "$0"
    
    parsed_ac = float(parsed_ac)
    
    if parsed_ac == 0:
        return "$0"
    elif parsed_ac <= 5:
        return "$1-5"
    elif parsed_ac <= 10:
        return "$6-10"
    elif parsed_ac <= 15:
        return "$11-15"
    elif parsed_ac <= 20:
        return "$16-20"
    elif parsed_ac <= 25:
        return "$21-25"
    elif parsed_ac <= 30:
        return "$26-30"
    elif parsed_ac <= 35:
        return "$31-35"
    elif parsed_ac <= 40:
        return "$36-40"
    elif parsed_ac <= 45:
        return "$41-45"
    elif parsed_ac <= 50:
        return "$46-50"
    elif parsed_ac <= 75:
        return "$51-75"
    elif parsed_ac <= 100:
        return "$76-100"
    else:
        return "$100+"


def sample_from_buckets(buckets: Dict[str, List[Dict]], samples_per_bucket: Dict[str, int]) -> List[Dict]:
    """Sample records from each bucket with metadata."""
    sampled_records = []
    
    for bucket_name, num_samples in samples_per_bucket.items():
        if bucket_name in buckets and num_samples > 0:
            records = buckets[bucket_name]
            sampled = random.sample(records, min(num_samples, len(records)))
            for record in sampled:
                record_copy = record.copy()
                record_copy['_sampling_metadata'] = {
                    'conversation_length': get_conversation_length(record),
                    'length_bucket': bucket_name
                }
                sampled_records.append(record_copy)
    
    # Shuffle to mix buckets
    random.shuffle(sampled_records)
    return sampled_records


def process_file(input_path: Path, total_samples: int) -> Tuple[List[Dict], List[Dict]]:
    """
    Process a payload file:
    1. Load data
    2. Sample based on conversation length
    3. Create Parsed_AC_Bucket for sampled records
    4. Return (sampled_with_bucket, original_without_parsed_ac)
    """
    # Load data
    data = load_data(input_path)
    
    # Create buckets based on conversation length
    buckets = create_buckets(data)
    
    print(f"\nBucket Distribution for {input_path.name}:")
    print("=" * 60)
    for bucket_name, records in sorted(buckets.items(), 
            key=lambda x: next((i for i, b in enumerate(LENGTH_BUCKETS) if b[2] == x[0]), 999)):
        if records:
            lengths = [get_conversation_length(r) for r in records]
            print(f"  {bucket_name:15} | Count: {len(records):6} | Length: {min(lengths):4} - {max(lengths):5}")
    
    # Calculate samples per bucket
    samples_per_bucket = calculate_samples_per_bucket(buckets, total_samples)
    
    print(f"\nSampling Plan:")
    print("=" * 60)
    for bucket_name, num_samples in sorted(samples_per_bucket.items(),
            key=lambda x: next((i for i, b in enumerate(LENGTH_BUCKETS) if b[2] == x[0]), 999)):
        pool_size = len(buckets[bucket_name])
        print(f"  {bucket_name:15} | Samples: {num_samples:3} / {pool_size:5}")
    
    # Sample records
    sampled_records = sample_from_buckets(buckets, samples_per_bucket)
    
    # Add Parsed_AC_Bucket to sampled records
    sampled_with_bucket = []
    for record in sampled_records:
        record_copy = record.copy()
        parsed_ac = record_copy.get('Parsed_AC', 0)
        record_copy['Parsed_AC_Bucket'] = get_parsed_ac_bucket(parsed_ac)
        sampled_with_bucket.append(record_copy)
    
    # Create version with Parsed_AC removed
    data_without_parsed_ac = []
    for record in data:
        record_copy = record.copy()
        if 'Parsed_AC' in record_copy:
            del record_copy['Parsed_AC']
        data_without_parsed_ac.append(record_copy)
    
    return sampled_with_bucket, data_without_parsed_ac


def create_ground_truth(sampled_records: List[Dict]) -> List[Dict]:
    """Create ground truth records with only DELIVERY_ID and Parsed_AC_Bucket."""
    ground_truth = []
    for record in sampled_records:
        gt_record = {
            'DELIVERY_ID': record.get('DELIVERY_ID'),
            'Parsed_AC_Bucket': record.get('Parsed_AC_Bucket')
        }
        ground_truth.append(gt_record)
    return ground_truth


def main():
    """Main processing function."""
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    print(f"{'='*70}")
    print("PAYLOAD FILE PROCESSING")
    print(f"{'='*70}")
    print(f"Random seed set to: {RANDOM_SEED}")
    print(f"Samples per file: {TOTAL_SAMPLES}")
    
    # Create ground truth directory
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process escalated cases
    print(f"\n{'='*70}")
    print("Processing ESCALATED cases")
    print(f"{'='*70}")
    escalated_sampled, escalated_without_parsed_ac = process_file(ESCALATED_INPUT, TOTAL_SAMPLES)
    
    # Process non-escalated cases
    print(f"\n{'='*70}")
    print("Processing NON-ESCALATED cases")
    print(f"{'='*70}")
    non_escalated_sampled, non_escalated_without_parsed_ac = process_file(NON_ESCALATED_INPUT, TOTAL_SAMPLES)
    
    # Step 3: Save modified original files (without Parsed_AC)
    print(f"\n{'='*70}")
    print("Step 3: Removing Parsed_AC from original files")
    print(f"{'='*70}")
    save_data(escalated_without_parsed_ac, ESCALATED_INPUT)
    save_data(non_escalated_without_parsed_ac, NON_ESCALATED_INPUT)
    
    # Step 4: Create and save ground truth files
    print(f"\n{'='*70}")
    print("Step 4: Creating ground truth files")
    print(f"{'='*70}")
    
    escalated_ground_truth = create_ground_truth(escalated_sampled)
    non_escalated_ground_truth = create_ground_truth(non_escalated_sampled)
    
    escalated_gt_path = GROUND_TRUTH_DIR / "ground_truth_escalated.json"
    non_escalated_gt_path = GROUND_TRUTH_DIR / "ground_truth_non_escalated.json"
    
    save_data(escalated_ground_truth, escalated_gt_path)
    save_data(non_escalated_ground_truth, non_escalated_gt_path)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nEscalated cases:")
    print(f"  - Original file updated: {ESCALATED_INPUT}")
    print(f"  - Ground truth saved: {escalated_gt_path}")
    print(f"  - Sampled: {len(escalated_sampled)} records")
    
    # Show Parsed_AC_Bucket distribution for escalated
    bucket_dist = defaultdict(int)
    for record in escalated_sampled:
        bucket_dist[record.get('Parsed_AC_Bucket', 'Unknown')] += 1
    print(f"  - Parsed_AC_Bucket distribution:")
    for bucket, count in sorted(bucket_dist.items()):
        print(f"      {bucket}: {count}")
    
    print(f"\nNon-escalated cases:")
    print(f"  - Original file updated: {NON_ESCALATED_INPUT}")
    print(f"  - Ground truth saved: {non_escalated_gt_path}")
    print(f"  - Sampled: {len(non_escalated_sampled)} records")
    
    # Show Parsed_AC_Bucket distribution for non-escalated
    bucket_dist = defaultdict(int)
    for record in non_escalated_sampled:
        bucket_dist[record.get('Parsed_AC_Bucket', 'Unknown')] += 1
    print(f"  - Parsed_AC_Bucket distribution:")
    for bucket, count in sorted(bucket_dist.items()):
        print(f"      {bucket}: {count}")
    
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

