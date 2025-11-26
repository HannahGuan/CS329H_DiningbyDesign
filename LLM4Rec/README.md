# LLM4Rec - DPO Training for User-Business Preference Learning

This repository contains code for training a language model using Direct Preference Optimization (DPO) to learn user preferences for businesses based on user profiles and business characteristics.

## Dataset

The project uses two Hugging Face datasets:
- `zetianli/CS329H_Project_user_profiles` - User profiles with reviews
- `zetianli/CS329H_Project_business` - Business information

## Files

### Data Preparation
- **`download_data.py`** - Downloads datasets from Hugging Face and saves as JSONL
- **`create_dpo_dataset.py`** - Creates DPO preference pairs from user profiles and business data
- **`verify_dpo_dataset.py`** - Analyzes and shows statistics about the DPO dataset

### Training
- **`train_dpo.py`** - Main training script using DPO with LoRA fine-tuning
- **`inference_dpo.py`** - Inference script to test the trained model

### Data Files
- `data/user_profiles.jsonl` - 20,000 user profiles
- `data/business.jsonl` - 78,123 businesses
- `data/dpo_preference_dataset.jsonl` - 17,517 preference pairs

## Quick Start

### 1. Download Data

```bash
python download_data.py
```

This will download the datasets from Hugging Face and save them in the `data/` directory.

### 2. Create DPO Dataset

```bash
python create_dpo_dataset.py
```

This creates preference pairs where:
- **Chosen**: User's highest-rated business (with user profile + business profile)
- **Rejected**: User's lowest-rated business (with user profile + business profile)

### 3. Train the Model

```bash
python train_dpo.py
```

**Training Configuration:**
- **Model**: Qwen/Qwen2.5-3B-Instruct
- **Quantization**: 4-bit with BitsAndBytes
- **LoRA**: r=8, alpha=16, dropout=0.1
- **Learning Rate**: 5e-5
- **Batch Size**: 4
- **Epochs**: 3
- **Train/Test Split**: 80/20

**Note:** Set your Hugging Face token as an environment variable:
```bash
export HUGGINGFACE_TOKEN=your_token_here
```

Or leave it as `None` if the model is public.

### 4. Run Inference

Test the base model (before training):
```bash
python inference_dpo.py --base_model Qwen/Qwen2.5-3B-Instruct --num_samples 5
```

Test the fine-tuned model:
```bash
python inference_dpo.py \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --adapter_path ./dpo_model_output/final_model \
    --num_samples 5
```

## DPO Dataset Format

Each entry in `dpo_preference_dataset.jsonl` contains:

```json
{
    "user_id": "user_123",
    "user_profile": "User's preference description...",
    "chosen": {
        "business_id": "business_456",
        "business_name": "Great Restaurant",
        "business_profile": "Business description...",
        "text": "User profile\n\nBusiness profile",
        "rating": 5.0,
        "review_text": "Original review text..."
    },
    "rejected": {
        "business_id": "business_789",
        "business_name": "Poor Restaurant",
        "business_profile": "Business description...",
        "text": "User profile\n\nBusiness profile",
        "rating": 1.0,
        "review_text": "Original review text..."
    }
}
```

## Dataset Statistics

- **Total preference pairs**: 17,517
- **Average rating gap**: 2.81 stars
- **Rating gap distribution**:
  - 4.0 stars: 40.7%
  - 3.0 stars: 19.5%
  - 2.0 stars: 19.3%
  - 1.0 stars: 20.4%

## Requirements

```bash
pip install torch transformers datasets peft trl bitsandbytes scikit-learn tqdm
```

For CUDA support, ensure you have the appropriate PyTorch version installed.

## Model Architecture

The training uses:
- **Base Model**: Qwen2.5-3B-Instruct
- **Quantization**: 4-bit NF4 quantization with double quantization
- **LoRA Adapters**: Applied to attention and MLP layers
- **Training Method**: Direct Preference Optimization (DPO)

## Output

Training outputs are saved to `./dpo_model_output/`:
- Checkpoints during training
- Final model in `final_model/`
- Training logs

## Notes

- The training uses 4-bit quantization to reduce memory usage
- LoRA allows efficient fine-tuning with ~1% trainable parameters
- DPO directly optimizes for preference alignment without separate reward model
- The model learns to predict which businesses align better with user profiles

## Citation

If you use this code or dataset, please cite the original data sources:
- CS329H Project datasets by zetianli on Hugging Face

