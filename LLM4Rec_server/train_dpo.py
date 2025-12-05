"""
Configurable DPO Training Script for User-Business Preference Learning
Usage: python train_dpo_configurable.py --help
"""
import torch
import json
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import os
import wandb

# Target modules for LoRA
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "fc1", "fc2",
    "w1", "w2", "w3",
]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DPO Training with configurable hyperparameters")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="HuggingFace model name or path")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token (or use HUGGINGFACE_TOKEN env var)")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to DPO preference dataset (JSONL file)")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion of data to use for validation (0.0-1.0)")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for data splitting")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./dpo_model_output",
                        help="Directory to save model checkpoints")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Per-device eval batch size (defaults to batch_size)")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Number of warmup steps")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")
    
    # Quantization arguments
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization")
    parser.add_argument("--no_4bit", action="store_false", dest="use_4bit",
                        help="Don't use 4-bit quantization")
    
    # Logging arguments
    parser.add_argument("--logging_steps", type=int, default=4,
                        help="Log every X steps")
    parser.add_argument("--eval_steps", type=int, default=50,
                        help="Evaluate every X steps")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every X steps")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="Maximum number of checkpoints to keep")
    
    # Wandb arguments
    parser.add_argument("--use_wandb", action="store_true", default=False,
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="LLM4Rec-DPO",
                        help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Wandb run name (auto-generated if not provided)")
    parser.add_argument("--wandb_api_key", type=str, default=None,
                        help="Wandb API key (or use WANDB_API_KEY env var)")
    
    return parser.parse_args()

#############################
#### LOAD DATA ####
#############################

def load_dpo_data(filepath):
    """Load DPO preference dataset from JSONL file."""
    print(f"Loading DPO data from {filepath}...")
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} preference pairs")
    return data

def format_preference_data(dpo_data):
    """
    Format DPO data into the structure needed for training.
    Creates prompts with structured format: problem intro + user profile + recommended restaurant.
    """
    formatted_data = []
    
    for item in tqdm(dpo_data, desc="Formatting data"):
        # Create structured prompt with problem introduction
        prompt = (
            "Task: Given a user's dining preferences and a restaurant's characteristics, "
            "determine how well the restaurant matches the user's preferences.\n\n"
            f"The user profile is: {item['user_profile']}"
        )
        
        # Format chosen response (high-rated business)
        chosen = (
            f"The recommended restaurant is: {item['chosen']['business_name']}\n\n"
            f"{item['chosen']['business_profile']}"
        )
        
        # Format rejected response (low-rated business)
        rejected = (
            f"The recommended restaurant is: {item['rejected']['business_name']}\n\n"
            f"{item['rejected']['business_profile']}"
        )
        
        formatted_data.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected,
            'user_id': item['user_id'],
            'chosen_rating': item['chosen']['rating'],
            'rejected_rating': item['rejected']['rating']
        })
    
    return formatted_data

def data_formulate(data, tokenizer):
    """Apply chat template to format the prompt."""
    # System prompt for restaurant/business recommendation task
    system_prompt = """You are a preference-aware recommendation assistant trained to understand user preferences for restaurants and businesses. Given a user's profile describing their preferences, analyze business profiles to determine compatibility. Consider all aspects: cuisine preferences, atmosphere, price point, service style, amenities, and location characteristics. Your goal is to identify businesses that would provide the best experience for each specific user."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": data['prompt']},
    ]
    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def print_trainable_parameters(model):
    """Print the number of trainable parameters."""
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"\n🚀 Trainable parameters: {trainable_params:,}")
    print(f"📦 Total parameters:     {all_param:,}")
    print(f"📈 Percentage:           {100 * trainable_params / all_param:.4f}%\n")

def main():
    args = parse_args()
    
    # Print configuration
    print("\n" + "="*80)
    print("DPO TRAINING CONFIGURATION")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"LoRA (r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout})")
    print(f"4-bit Quantization: {args.use_4bit}")
    print(f"Test Split: {args.test_size}")
    print(f"Use Wandb: {args.use_wandb}")
    print("="*80 + "\n")
    
    # Setup tokens
    hf_token = args.hf_token or os.environ.get("HUGGINGFACE_TOKEN", None)
    
    # Setup Wandb if requested
    if args.use_wandb:
        wandb_api_key = args.wandb_api_key or os.environ.get("WANDB_API_KEY", None)
        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key
            wandb.login(key=wandb_api_key)
            
            # Auto-generate run name if not provided
            if args.wandb_run_name is None:
                args.wandb_run_name = f"dpo-{args.model_name.split('/')[-1]}-lr{args.learning_rate}-bs{args.batch_size}-ep{args.num_epochs}"
            
            print(f"Weights & Biases project: {args.wandb_project}")
            print(f"Run name: {args.wandb_run_name}\n")
        else:
            print("⚠️  Warning: Wandb requested but no API key found. Disabling wandb.")
            args.use_wandb = False
    
    #############################
    #### LOAD DATA ####
    #############################
    
    PREFERENCE_DATA = load_dpo_data(args.data_path)
    prefs = format_preference_data(PREFERENCE_DATA)
    
    # Split into train and test
    train_data, test_data = train_test_split(
        prefs, 
        test_size=args.test_size, 
        random_state=args.random_seed
    )
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}\n")
    
    #############################
    #### MODEL SETUP ####
    #############################
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    if args.use_4bit:
        print("Using 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            token=hf_token,
            quantization_config=quantization_config
        )
    else:
        print("Loading full precision model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            token=hf_token,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    #############################
    #### DATA PREPARATION ####
    #############################
    
    print("\nPreparing training dataset...")
    prompt_list = [data_formulate(data, tokenizer) for data in tqdm(train_data, desc="Formatting prompts")]
    chosen_list = [data['chosen'] for data in train_data]
    rejected_list = [data['rejected'] for data in train_data]
    
    train_dataset = Dataset.from_dict({
        'prompt': prompt_list,
        'chosen': chosen_list,
        'rejected': rejected_list
    })
    
    print(f"Train dataset: {train_dataset}")
    
    # Prepare test dataset
    print("\nPreparing test dataset...")
    test_prompt_list = [data_formulate(data, tokenizer) for data in tqdm(test_data, desc="Formatting test prompts")]
    test_chosen_list = [data['chosen'] for data in test_data]
    test_rejected_list = [data['rejected'] for data in test_data]
    
    test_dataset = Dataset.from_dict({
        'prompt': test_prompt_list,
        'chosen': test_chosen_list,
        'rejected': test_rejected_list
    })
    
    print(f"Test dataset: {test_dataset}")
    
    #############################
    #### TRAINING CONFIG ####
    #############################
    
    print("\nSetting up training configuration...")
    
    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
    
    training_args = DPOConfig(
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_only_model=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        eval_strategy="steps" if args.test_size > 0 else "no",
        eval_steps=args.eval_steps if args.test_size > 0 else None,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="wandb" if args.use_wandb else "none",
        run_name=args.wandb_run_name if args.use_wandb else None,
        remove_unused_columns=False,
        max_length=args.max_length,
    )
    
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model": args.model_name,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "epochs": args.num_epochs,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "data_path": args.data_path,
                "use_4bit": args.use_4bit,
            }
        )
    
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        target_modules=TARGET_MODULES,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    #############################
    #### TRAINER SETUP ####
    #############################
    
    print("\nInitializing DPO trainer...")
    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if args.test_size > 0 else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    print_trainable_parameters(dpo_trainer.model)
    
    #############################
    #### TRAINING ####
    #############################
    
    print("\n" + "="*80)
    print("Starting DPO Training...")
    print("="*80 + "\n")
    
    # Train the model
    dpo_trainer.train()
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80 + "\n")
    
    # Save the final model
    print(f"Saving final model to {args.output_dir}/final_model...")
    dpo_trainer.save_model(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")
    
    print("\n✅ Training complete! Model saved successfully.")

if __name__ == "__main__":
    main()

