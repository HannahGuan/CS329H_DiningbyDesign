"""
Create evaluation split by extracting the last N users from the DPO dataset
"""
import json
import argparse

def create_eval_split(input_path, output_path, num_users=20):
    """
    Extract the last N users from the DPO dataset for evaluation.
    
    Args:
        input_path: Path to full DPO dataset
        output_path: Path to save evaluation split
        num_users: Number of users to extract from the end
    """
    print(f"Loading DPO dataset from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    print(f"Total samples in dataset: {len(data)}")
    
    # Get last N samples
    eval_data = data[-num_users:]
    
    print(f"Extracted last {len(eval_data)} samples for evaluation")
    print(f"First eval user ID: {eval_data[0]['user_id']}")
    print(f"Last eval user ID: {eval_data[-1]['user_id']}")
    
    # Save evaluation split
    print(f"\nSaving evaluation split to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in eval_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✅ Evaluation split created successfully!")
    print(f"   File: {output_path}")
    print(f"   Samples: {len(eval_data)}")
    
    # Show rating distribution
    from collections import Counter
    chosen_ratings = [item['chosen']['rating'] for item in eval_data]
    rejected_ratings = [item['rejected']['rating'] for item in eval_data]
    
    print(f"\nRating distribution in evaluation set:")
    print(f"Chosen ratings:")
    for rating, count in sorted(Counter(chosen_ratings).items(), reverse=True):
        print(f"  {rating} stars: {count}")
    
    print(f"\nRejected ratings:")
    for rating, count in sorted(Counter(rejected_ratings).items()):
        print(f"  {rating} stars: {count}")
    
    return eval_data

def main():
    parser = argparse.ArgumentParser(description="Create evaluation split from DPO dataset")
    parser.add_argument("--input_path", type=str, default="data/dpo_preference_dataset.jsonl",
                        help="Path to full DPO dataset")
    parser.add_argument("--output_path", type=str, default="data/dpo_eval_last20.jsonl",
                        help="Path to save evaluation split")
    parser.add_argument("--num_users", type=int, default=20,
                        help="Number of users to extract from the end")
    
    args = parser.parse_args()
    
    create_eval_split(args.input_path, args.output_path, args.num_users)

if __name__ == "__main__":
    main()

