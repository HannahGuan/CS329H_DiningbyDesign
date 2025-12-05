"""
Inference script for trained DPO model with updated prompt format
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import argparse
import numpy as np
from tqdm import tqdm

def load_model_and_tokenizer(base_model_name, adapter_path=None):
    """Load the model and tokenizer."""
    print(f"Loading tokenizer from {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading base model from {base_model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    if adapter_path:
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    
    return model, tokenizer

def calculate_perplexity(model, tokenizer, prompt, response):
    """
    Calculate perplexity of a response given a prompt.
    Lower perplexity means the model finds the response more likely.
    """
    # Combine prompt and response
    full_text = prompt + response
    
    # Tokenize
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    prompt_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Get the length of prompt tokens to mask them out
    prompt_length = prompt_inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        
        # Get logits and shift for next token prediction
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs['input_ids'][..., 1:].contiguous()
        
        # Calculate loss only on the response tokens (not the prompt)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Reshape losses back and mask out prompt tokens
        losses = losses.view(shift_labels.shape)
        
        # Only consider losses for response tokens (after prompt)
        response_losses = losses[:, prompt_length-1:]
        
        # Calculate average loss and perplexity
        avg_loss = response_losses.mean().item()
        perplexity = np.exp(avg_loss)
    
    return perplexity, avg_loss

def generate_recommendation(model, tokenizer, user_profile, max_length=512):
    """
    Generate a restaurant recommendation for a given user profile.
    Uses the new prompt format matching the training data.
    """
    # System prompt (same as training)
    system_prompt = """You are a preference-aware recommendation assistant trained to understand user preferences for restaurants and businesses. Given a user's profile describing their preferences, analyze business profiles to determine compatibility. Consider all aspects: cuisine preferences, atmosphere, price point, service style, amenities, and location characteristics. Your goal is to identify businesses that would provide the best experience for each specific user."""
    
    # Create prompt (same format as training)
    user_prompt = (
        "Task: Given a user's dining preferences and a restaurant's characteristics, "
        "determine how well the restaurant matches the user's preferences.\n\n"
        f"The user profile is: {user_profile}"
    )
    
    # Format with chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()
    elif "assistant\n" in response:
        response = response.split("assistant\n")[-1].strip()
    
    return response, formatted_prompt

def evaluate_business_perplexity(model, tokenizer, user_profile, business_name, business_profile):
    """
    Evaluate perplexity for a specific business recommendation.
    Returns the perplexity and formatted prompt.
    """
    # System prompt (same as training)
    system_prompt = """You are a preference-aware recommendation assistant trained to understand user preferences for restaurants and businesses. Given a user's profile describing their preferences, analyze business profiles to determine compatibility. Consider all aspects: cuisine preferences, atmosphere, price point, service style, amenities, and location characteristics. Your goal is to identify businesses that would provide the best experience for each specific user."""
    
    # Create prompt (same format as training)
    user_prompt = (
        "Task: Given a user's dining preferences and a restaurant's characteristics, "
        "determine how well the restaurant matches the user's preferences.\n\n"
        f"The user profile is: {user_profile}"
    )
    
    # Format with chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Create the expected response (same format as training)
    response = f"The recommended restaurant is: {business_name}\n\n{business_profile}"
    
    # Calculate perplexity
    perplexity, avg_loss = calculate_perplexity(model, tokenizer, formatted_prompt, response)
    
    return perplexity, avg_loss

def test_from_dpo_dataset(model, tokenizer, dpo_data_path, num_samples=5, use_last=False, output_file=None):
    """Test the model on samples from the DPO dataset."""
    print(f"\nLoading test data from {dpo_data_path}...")
    with open(dpo_data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    if use_last:
        # Use the last N samples
        samples = data[-num_samples:]
        print(f"Testing on LAST {num_samples} samples...\n")
    else:
        # Random sampling
        import random
        samples = random.sample(data, min(num_samples, len(data)))
        print(f"Testing on {num_samples} random samples...\n")
    
    # Store results if output file is specified
    results = []
    
    # Use tqdm for progress tracking
    for i, sample in enumerate(tqdm(samples, desc="Running inference", unit="sample"), 1):
        # Generate recommendation
        response, formatted_prompt = generate_recommendation(
            model, tokenizer,
            sample['user_profile'],
            max_length=512
        )
        
        # Calculate perplexity for chosen business
        chosen_perplexity, chosen_loss = evaluate_business_perplexity(
            model, tokenizer,
            sample['user_profile'],
            sample['chosen']['business_name'],
            sample['chosen']['business_profile']
        )
        
        # Calculate perplexity for rejected business
        rejected_perplexity, rejected_loss = evaluate_business_perplexity(
            model, tokenizer,
            sample['user_profile'],
            sample['rejected']['business_name'],
            sample['rejected']['business_profile']
        )
        
        # Determine preference
        prefers_chosen = chosen_perplexity < rejected_perplexity
        perplexity_diff = rejected_perplexity - chosen_perplexity
        
        # Store result
        result = {
            'sample_id': i,
            'user_id': sample['user_id'],
            'user_profile': sample['user_profile'],
            'chosen_business': sample['chosen']['business_name'],
            'chosen_rating': sample['chosen']['rating'],
            'chosen_profile': sample['chosen']['business_profile'],
            'chosen_perplexity': float(chosen_perplexity),
            'chosen_loss': float(chosen_loss),
            'rejected_business': sample['rejected']['business_name'],
            'rejected_rating': sample['rejected']['rating'],
            'rejected_profile': sample['rejected']['business_profile'],
            'rejected_perplexity': float(rejected_perplexity),
            'rejected_loss': float(rejected_loss),
            'model_response': response,
            'prefers_chosen': bool(prefers_chosen),
            'perplexity_diff': float(perplexity_diff)
        }
        results.append(result)
    
    print("\n" + "="*80)
    
    # Calculate and display summary statistics
    if results:
        correct_preferences = sum(1 for r in results if r['prefers_chosen'])
        accuracy = (correct_preferences / len(results)) * 100
        avg_perplexity_diff = np.mean([r['perplexity_diff'] for r in results])
        
        avg_chosen_ppl = np.mean([r['chosen_perplexity'] for r in results])
        avg_rejected_ppl = np.mean([r['rejected_perplexity'] for r in results])
        
        print("SUMMARY STATISTICS")
        print(f"{'='*80}")
        print(f"Total samples: {len(results)}")
        print(f"Model prefers chosen: {correct_preferences} ({accuracy:.2f}%)")
        print(f"Model prefers rejected: {len(results) - correct_preferences} ({100-accuracy:.2f}%)")
        print(f"\nPerplexity:")
        print(f"  Avg chosen perplexity: {avg_chosen_ppl:.2f}")
        print(f"  Avg rejected perplexity: {avg_rejected_ppl:.2f}")
        print(f"  Avg perplexity difference: {avg_perplexity_diff:.2f}")
        
        # Save results to file if specified
        if output_file:
            print(f"\n{'='*80}")
            print(f"Saving results to {output_file}...")
            output_data = {
                'total_samples': len(samples),
                'accuracy': accuracy,
                'correct_preferences': correct_preferences,
                'avg_chosen_perplexity': float(avg_chosen_ppl),
                'avg_rejected_perplexity': float(avg_rejected_ppl),
                'avg_perplexity_diff': float(avg_perplexity_diff),
                'results': results
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Results saved to: {output_file}")
        
        print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description="Run inference with trained DPO model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Base model name")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to LoRA adapter (if trained)")
    parser.add_argument("--dpo_data", type=str, default="data/dpo_preference_dataset.jsonl",
                        help="Path to DPO dataset for testing")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to test")
    parser.add_argument("--use_last", action="store_true",
                        help="Use last N samples instead of random sampling")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save inference results as JSON")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("DPO MODEL INFERENCE")
    print("="*80)
    print(f"Base model: {args.base_model}")
    print(f"Adapter: {args.adapter_path if args.adapter_path else 'None (base model only)'}")
    print(f"Dataset: {args.dpo_data}")
    print(f"Samples: {args.num_samples}")
    print(f"Output: {args.output_file if args.output_file else 'No file output'}")
    print("="*80)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter_path)
    
    # Test on DPO dataset
    test_from_dpo_dataset(model, tokenizer, args.dpo_data, args.num_samples, args.use_last, args.output_file)

if __name__ == "__main__":
    main()
