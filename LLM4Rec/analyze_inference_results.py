"""
Analyze inference results from the evaluation
"""
import json
import argparse
import numpy as np

def analyze_results(results_file):
    """Analyze the inference results."""
    print(f"Loading results from {results_file}...")
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    total = len(results)
    
    print("\n" + "="*80)
    print("INFERENCE RESULTS ANALYSIS")
    print("="*80)
    
    # Overall statistics
    print(f"\nTotal samples: {total}")
    print(f"Accuracy (model prefers chosen): {data.get('accuracy', 0):.2f}%")
    print(f"Correct preferences: {data.get('correct_preferences', 0)}/{total}")
    print(f"\nAverage Perplexities:")
    print(f"  Chosen: {data.get('avg_chosen_perplexity', 0):.2f}")
    print(f"  Rejected: {data.get('avg_rejected_perplexity', 0):.2f}")
    print(f"  Difference: {data.get('avg_perplexity_diff', 0):.2f}")
    
    # Perplexity statistics
    chosen_perplexities = [r['chosen_perplexity'] for r in results]
    rejected_perplexities = [r['rejected_perplexity'] for r in results]
    perplexity_diffs = [r['perplexity_diff'] for r in results]
    
    print(f"\n--- Perplexity Statistics ---")
    print(f"Chosen perplexity:")
    print(f"  Mean: {np.mean(chosen_perplexities):.2f}")
    print(f"  Median: {np.median(chosen_perplexities):.2f}")
    print(f"  Min: {np.min(chosen_perplexities):.2f}")
    print(f"  Max: {np.max(chosen_perplexities):.2f}")
    
    print(f"\nRejected perplexity:")
    print(f"  Mean: {np.mean(rejected_perplexities):.2f}")
    print(f"  Median: {np.median(rejected_perplexities):.2f}")
    print(f"  Min: {np.min(rejected_perplexities):.2f}")
    print(f"  Max: {np.max(rejected_perplexities):.2f}")
    
    print(f"\nPerplexity difference (rejected - chosen):")
    print(f"  Mean: {np.mean(perplexity_diffs):.2f}")
    print(f"  Median: {np.median(perplexity_diffs):.2f}")
    
    # Rating analysis
    print(f"\n--- Rating Analysis ---")
    rating_gaps = [r['chosen_rating'] - r['rejected_rating'] for r in results]
    print(f"Rating gap (chosen - rejected):")
    print(f"  Mean: {np.mean(rating_gaps):.2f} stars")
    print(f"  Median: {np.median(rating_gaps):.2f} stars")
    
    # Show cases where model got it wrong
    incorrect = [r for r in results if not r['prefers_chosen']]
    if incorrect:
        print(f"\n--- Cases Where Model Prefers Rejected ({len(incorrect)} cases) ---")
        for i, case in enumerate(incorrect[:5], 1):  # Show first 5
            print(f"\nCase {i}:")
            print(f"  User: {case['user_id']}")
            print(f"  Chosen: {case['chosen_business']} (rating: {case['chosen_rating']}, ppl: {case['chosen_perplexity']:.2f})")
            print(f"  Rejected: {case['rejected_business']} (rating: {case['rejected_rating']}, ppl: {case['rejected_perplexity']:.2f})")
            print(f"  Perplexity diff: {case['perplexity_diff']:.2f}")
    
    # Show strong correct predictions
    correct = [r for r in results if r['prefers_chosen']]
    if correct:
        # Sort by perplexity difference (larger = more confident)
        correct_sorted = sorted(correct, key=lambda x: x['perplexity_diff'], reverse=True)
        print(f"\n--- Strongest Correct Predictions (Top 3) ---")
        for i, case in enumerate(correct_sorted[:3], 1):
            print(f"\nCase {i}:")
            print(f"  User: {case['user_id']}")
            print(f"  Chosen: {case['chosen_business']} (rating: {case['chosen_rating']}, ppl: {case['chosen_perplexity']:.2f})")
            print(f"  Rejected: {case['rejected_business']} (rating: {case['rejected_rating']}, ppl: {case['rejected_perplexity']:.2f})")
            print(f"  Perplexity diff: {case['perplexity_diff']:.2f} (strong preference for chosen)")
    
    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description="Analyze inference results")
    parser.add_argument("--results_file", type=str, required=True,
                        help="Path to inference results JSON file")
    
    args = parser.parse_args()
    analyze_results(args.results_file)

if __name__ == "__main__":
    main()

