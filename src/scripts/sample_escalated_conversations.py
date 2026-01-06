"""
Sampling Script for Escalated Conversations

This script samples diverse conversations from the escalated cases payload
based on conversation length distribution. It creates buckets by conversation
length and samples from each bucket to ensure diversity.

Usage:
    python sample_escalated_conversations.py

Output:
    Saves sampled conversations to sampled_payload.json
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# Configuration
RANDOM_SEED = 42
TOTAL_SAMPLES = 50
INPUT_FILE = Path("/Users/shekhar.tanwar/Documents/Projects/NegotiatonAgent/dataset/escalated_payload/escalated_cases_payload.json")
OUTPUT_FILE = Path("/Users/shekhar.tanwar/Documents/Projects/NegotiatonAgent/dataset/escalated_payload/sampled_payload.json")

# Define conversation length buckets based on the data distribution
# Using percentile-based boundaries for diverse sampling
LENGTH_BUCKETS = [
    (0, 500, "very_short"),        # ~10th percentile and below
    (500, 750, "short"),           # 10th to ~30th percentile
    (750, 1000, "medium_short"),   # ~30th to 50th percentile
    (1000, 1500, "medium"),        # 50th to 75th percentile
    (1500, 2000, "medium_long"),   # 75th to 90th percentile
    (2000, 2500, "long"),          # 90th to 95th percentile
    (2500, float('inf'), "very_long")  # 95th percentile and above
]


def load_data(filepath: Path) -> list[dict]:
    """Load the JSON payload file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_conversation_length(record: dict) -> int:
    """Extract the length of the conversation from a record."""
    return len(record.get('CONVERSATION_CB', ''))


def assign_bucket(length: int) -> str:
    """Assign a conversation to a bucket based on its length."""
    for min_len, max_len, bucket_name in LENGTH_BUCKETS:
        if min_len <= length < max_len:
            return bucket_name
    return LENGTH_BUCKETS[-1][2]  # Default to last bucket


def create_buckets(data: list[dict]) -> dict[str, list[dict]]:
    """Organize records into buckets based on conversation length."""
    buckets = defaultdict(list)
    
    for record in data:
        length = get_conversation_length(record)
        bucket_name = assign_bucket(length)
        buckets[bucket_name].append(record)
    
    return dict(buckets)


def calculate_samples_per_bucket(buckets: dict[str, list[dict]], total_samples: int) -> dict[str, int]:
    """
    Calculate how many samples to take from each bucket.
    
    Uses proportional allocation with a minimum of 1 sample per non-empty bucket,
    then adjusts to ensure exactly total_samples are selected.
    """
    total_records = sum(len(records) for records in buckets.values())
    non_empty_buckets = {k: v for k, v in buckets.items() if len(v) > 0}
    
    # Calculate proportional allocation
    samples_per_bucket = {}
    for bucket_name, records in non_empty_buckets.items():
        proportion = len(records) / total_records
        # Use ceiling to ensure minimum representation, but at least 1
        samples = max(1, round(proportion * total_samples))
        samples_per_bucket[bucket_name] = min(samples, len(records))
    
    # Adjust to match exactly total_samples
    current_total = sum(samples_per_bucket.values())
    
    if current_total > total_samples:
        # Reduce from largest buckets
        sorted_buckets = sorted(samples_per_bucket.items(), key=lambda x: -x[1])
        for bucket_name, _ in sorted_buckets:
            if current_total <= total_samples:
                break
            if samples_per_bucket[bucket_name] > 1:
                samples_per_bucket[bucket_name] -= 1
                current_total -= 1
    
    elif current_total < total_samples:
        # Add to buckets that have capacity
        sorted_buckets = sorted(
            samples_per_bucket.items(), 
            key=lambda x: len(buckets[x[0]]) - x[1],  # Sort by remaining capacity
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


def sample_from_buckets(buckets: dict[str, list[dict]], samples_per_bucket: dict[str, int]) -> list[dict]:
    """Sample records from each bucket."""
    sampled_records = []
    
    for bucket_name, num_samples in samples_per_bucket.items():
        if bucket_name in buckets and num_samples > 0:
            records = buckets[bucket_name]
            # Sample without replacement
            sampled = random.sample(records, min(num_samples, len(records)))
            sampled_records.extend(sampled)
    
    return sampled_records


def add_metadata(record: dict, bucket_name: str) -> dict:
    """Add sampling metadata to a record."""
    record_copy = record.copy()
    record_copy['_sampling_metadata'] = {
        'conversation_length': get_conversation_length(record),
        'length_bucket': bucket_name
    }
    return record_copy


def main():
    """Main sampling function."""
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    print(f"Random seed set to: {RANDOM_SEED}")
    
    # Load data
    print(f"\nLoading data from: {INPUT_FILE}")
    data = load_data(INPUT_FILE)
    print(f"Loaded {len(data)} records")
    
    # Create buckets
    buckets = create_buckets(data)
    print(f"\n{'='*60}")
    print("Bucket Distribution:")
    print(f"{'='*60}")
    for bucket_name, records in sorted(buckets.items(), key=lambda x: LENGTH_BUCKETS.index(next(b for b in LENGTH_BUCKETS if b[2] == x[0]))):
        lengths = [get_conversation_length(r) for r in records]
        print(f"  {bucket_name:15} | Count: {len(records):6} | Length range: {min(lengths):4} - {max(lengths):5}")
    
    # Calculate samples per bucket
    samples_per_bucket = calculate_samples_per_bucket(buckets, TOTAL_SAMPLES)
    print(f"\n{'='*60}")
    print("Sampling Plan:")
    print(f"{'='*60}")
    for bucket_name, num_samples in sorted(samples_per_bucket.items(), key=lambda x: LENGTH_BUCKETS.index(next(b for b in LENGTH_BUCKETS if b[2] == x[0]))):
        pool_size = len(buckets[bucket_name])
        print(f"  {bucket_name:15} | Samples: {num_samples:3} / {pool_size:5} ({100*num_samples/pool_size:.1f}%)")
    
    # Perform sampling
    sampled_records = []
    for bucket_name, num_samples in samples_per_bucket.items():
        if bucket_name in buckets and num_samples > 0:
            records = buckets[bucket_name]
            sampled = random.sample(records, min(num_samples, len(records)))
            # Add metadata for tracking
            for record in sampled:
                record_with_meta = add_metadata(record, bucket_name)
                sampled_records.append(record_with_meta)
    
    # Shuffle final list to mix buckets
    random.shuffle(sampled_records)
    
    print(f"\n{'='*60}")
    print(f"Sampled {len(sampled_records)} conversations")
    print(f"{'='*60}")
    
    # Summary statistics of sampled data
    sampled_lengths = [get_conversation_length(r) for r in sampled_records]
    print(f"\nSampled conversation length statistics:")
    print(f"  Min: {min(sampled_lengths)}")
    print(f"  Max: {max(sampled_lengths)}")
    print(f"  Mean: {sum(sampled_lengths)/len(sampled_lengths):.1f}")
    
    # Save sampled data
    print(f"\nSaving sampled data to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(sampled_records, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Successfully saved {len(sampled_records)} sampled conversations")
    
    return sampled_records


if __name__ == "__main__":
    main()

