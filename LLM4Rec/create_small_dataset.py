"""
Create a smaller version of the DPO dataset for testing
"""
import json
import random

def create_small_dataset(input_path, output_path, num_users=100):
    """
    Create a smaller dataset with specified number of users.
    
    Args:
        input_path: Path to full DPO dataset
        output_path: Path to save small dataset
        num_users: Number of users to sample
    """
    print(f"Loading full dataset from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        full_data = [json.loads(line) for line in f if line.strip()]
    
    print(f"Full dataset size: {len(full_data)} users")
    
    # Randomly sample users
    random.seed(42)
    small_data = random.sample(full_data, min(num_users, len(full_data)))
    
    print(f"Sampled {len(small_data)} users")
    
    # Save small dataset
    print(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in small_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✅ Small dataset created successfully!")
    print(f"   Total examples: {len(small_data)}")
    
    # Show rating distribution
    from collections import Counter
    chosen_ratings = [item['chosen']['rating'] for item in small_data]
    rejected_ratings = [item['rejected']['rating'] for item in small_data]
    
    print(f"\nChosen ratings distribution:")
    for rating, count in sorted(Counter(chosen_ratings).items(), reverse=True):
        print(f"  {rating} stars: {count}")
    
    print(f"\nRejected ratings distribution:")
    for rating, count in sorted(Counter(rejected_ratings).items()):
        print(f"  {rating} stars: {count}")

if __name__ == "__main__":
    create_small_dataset(
        input_path="data/dpo_preference_dataset.jsonl",
        output_path="data/dpo_preference_dataset_small.jsonl",
        num_users=100
    )

