"""
Script to create a DPO preference dataset from user profiles and business data.
For each user, selects their most positive and most negative reviews as chosen/rejected pairs.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

def load_jsonl(filepath):
    """Load data from JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def create_business_lookup(business_data: List[Dict]) -> Dict[str, Dict]:
    """Create a lookup dictionary for businesses by business_id."""
    return {business['business_id']: business for business in business_data}

def find_best_and_worst_reviews(reviews: List[Dict]) -> Tuple[Dict, Dict]:
    """
    Find the most positive (highest stars) and most negative (lowest stars) reviews.
    Returns (best_review, worst_review) or (None, None) if not enough variety.
    """
    if not reviews or len(reviews) < 2:
        return None, None
    
    # Sort by stars
    sorted_reviews = sorted(reviews, key=lambda x: x['stars'])
    worst_review = sorted_reviews[0]
    best_review = sorted_reviews[-1]
    
    # Only return if there's a difference in ratings
    if best_review['stars'] == worst_review['stars']:
        return None, None
    
    return best_review, worst_review

def create_dpo_dataset(user_profiles_path: str, business_path: str, output_path: str):
    """
    Create DPO preference dataset.
    
    For each user:
    - Find their highest-rated review (chosen/positive)
    - Find their lowest-rated review (rejected/negative)
    - Combine user profile with corresponding business profiles
    """
    print("Loading user profiles...")
    users = load_jsonl(user_profiles_path)
    print(f"Loaded {len(users)} users")
    
    print("Loading business data...")
    businesses = load_jsonl(business_path)
    business_lookup = create_business_lookup(businesses)
    print(f"Loaded {len(businesses)} businesses")
    
    print("\nCreating DPO dataset...")
    dpo_data = []
    skipped_count = 0
    missing_business_count = 0
    
    for user in users:
        user_id = user['user_id']
        user_profile = user.get('profile', '')
        reviews = user.get('reviews', [])
        
        # Find best and worst reviews
        best_review, worst_review = find_best_and_worst_reviews(reviews)
        
        if best_review is None or worst_review is None:
            skipped_count += 1
            continue
        
        # Get business profiles
        best_business_id = best_review['business_id']
        worst_business_id = worst_review['business_id']
        
        best_business = business_lookup.get(best_business_id)
        worst_business = business_lookup.get(worst_business_id)
        
        if not best_business or not worst_business:
            missing_business_count += 1
            continue
        
        best_business_profile = best_business.get('profile', '')
        worst_business_profile = worst_business.get('profile', '')
        
        # Create DPO example
        dpo_example = {
            'user_id': user_id,
            'user_profile': user_profile,
            'chosen': {
                'business_id': best_business_id,
                'business_name': best_review['name'],
                'business_profile': best_business_profile,
                'text': f"{user_profile}\n\n{best_business_profile}",
                'rating': best_review['stars'],
                'review_text': best_review['text']
            },
            'rejected': {
                'business_id': worst_business_id,
                'business_name': worst_review['name'],
                'business_profile': worst_business_profile,
                'text': f"{user_profile}\n\n{worst_business_profile}",
                'rating': worst_review['stars'],
                'review_text': worst_review['text']
            }
        }
        
        dpo_data.append(dpo_example)
    
    # Save to JSONL
    print(f"\nSaving DPO dataset to {output_path}...")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dpo_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"\n{'='*60}")
    print(f"DPO Dataset Creation Complete!")
    print(f"{'='*60}")
    print(f"Total users processed: {len(users)}")
    print(f"Valid DPO pairs created: {len(dpo_data)}")
    print(f"Skipped (no rating variance): {skipped_count}")
    print(f"Skipped (missing business data): {missing_business_count}")
    print(f"Output saved to: {output_path}")
    
    # Print sample
    if dpo_data:
        print(f"\n{'='*60}")
        print("Sample DPO Example:")
        print(f"{'='*60}")
        sample = dpo_data[0]
        print(f"\nUser ID: {sample['user_id']}")
        print(f"\nUser Profile (first 200 chars):")
        print(f"{sample['user_profile'][:200]}...")
        print(f"\n--- CHOSEN (Rating: {sample['chosen']['rating']}) ---")
        print(f"Business: {sample['chosen']['business_name']}")
        print(f"Business Profile (first 200 chars):")
        print(f"{sample['chosen']['business_profile'][:200]}...")
        print(f"\n--- REJECTED (Rating: {sample['rejected']['rating']}) ---")
        print(f"Business: {sample['rejected']['business_name']}")
        print(f"Business Profile (first 200 chars):")
        print(f"{sample['rejected']['business_profile'][:200]}...")

def main():
    """Main function."""
    user_profiles_path = "data/user_profiles.jsonl"
    business_path = "data/business.jsonl"
    output_path = "data/dpo_preference_dataset.jsonl"
    
    create_dpo_dataset(user_profiles_path, business_path, output_path)

if __name__ == "__main__":
    main()

