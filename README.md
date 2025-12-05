# CS329H_DiningbyDesign

This repository contains the implementation of a personalized restaurant recommendation system using Direct Preference Optimization (DPO) to fine-tune large language models on user preferences derived from Yelp data.

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Datasets](#datasets)
- [Reproducing Results](#reproducing-results)
- [Trained Models](#trained-models)
- [Computational Requirements](#computational-requirements)
- [Reproducibility](#reproducibility)

## Overview

This project fine-tunes small language models (Qwen3-0.6B, LFM2-350M) using DPO on personalized restaurant recommendation tasks. We generate natural language user and restaurant profiles from Yelp review data, then train models to predict user preferences by comparing their perplexity on preferred vs. non-preferred restaurants.

**Key Components:**
- Profile generation from Yelp reviews using GPT-based models
- DPO training with LoRA parameter-efficient fine-tuning
- Multiple experimental settings (single-item vs. full-list recommendations)
- Baseline comparison with GPT-4o-mini
- Evaluation metrics: Perplexity (chosen/rejected), MAP, NDCG, Pairwise Accuracy

## Repository Structure

```
CS329H_DiningbyDesign/
├── LLM4Rec/                          # Main experimentation directory (Google Colab)
│   ├── data/                         # Shared data files
│   ├── hannah_qwen_list/             # Qwen3-0.6B length2-list recommendation
│   │   └── dpo_train_hannah_list.ipynb
│   ├── hannah_qwen_single/           # Qwen3-0.6B single-item recommendation
│   │   └── dpo_train_hannah_single.ipynb
│   ├── hannah_GPT4o_baseline/        # GPT-4o-mini baseline experiments
│   │   └── dpo_train_hannah3_baseline.ipynb
│   ├── justin_LFM_fulllist/          # LFM2-350M full-list recommendation
│   │   └── justin_LFM_Fulllist.ipynb
│   ├── justin_Qwen_fulllist/         # Qwen3-0.6B full-list (alternative)
│   │   └── justin_Qwen_Fulllist.ipynb
│   ├── yanzhen_final_list/           # LFM2 length2-list recommendation
│   │   └── dpo_train_yanzhen_list.ipynb
│   └── yanzhen_final_single/         # LFM2 single-item recommendation
│       └── dpo_train_yanzhen_single.ipynb
│
├── LLM4Rec_server/                   # Server-based training (basic DPO)
│   ├── script/                       # Training and inference scripts
│   │   ├── train_full.sh
│   │   ├── train_small.sh
│   │   └── run_inference_eval.sh
│   ├── train_dpo.py                  # DPO training script
│   ├── inference_dpo.py              # Inference and evaluation
│   ├── create_dpo_dataset.py         # Dataset preparation
│   ├── create_eval_split.py          # Evaluation split creation
│   ├── analyze_inference_results.py  # Results analysis
│   ├── download_data.py              # Data download utility
│   └── requirements.txt              # Python dependencies
│
├── profile_generation/               # Profile generation scripts
│   ├── generate_profiles.py          # Main profile generation script
│   ├── generate_profile_user.sh      # User profile generation
│   ├── generate_profile_resaurants.sh
│   └── constants.py                  # Configuration constants
│
├── data_processing_code/             # Data preprocessing notebooks
│   ├── data_processing.ipynb
│   └── data_combine_clean.ipynb
│
└── README.md                         # This file
```

## Environment Setup

### For Google Colab (LLM4Rec)

The notebooks in `LLM4Rec/` are designed to run on Google Colab with A100 GPUs. Each notebook contains installation cells:

```python
!pip install trl
!pip install bitsandbytes
!pip install huggingface_hub
```

Required packages (versions will be auto-installed):
- torch>=2.0.0
- transformers>=4.40.0
- datasets>=2.14.0
- peft>=0.10.0
- trl>=0.8.0
- bitsandbytes>=0.43.0
- accelerate>=0.27.0
- wandb>=0.16.0

### For Server Environment (LLM4Rec_server)

```bash
cd LLM4Rec_server
pip install -r requirements.txt
```

See [LLM4Rec_server/requirements.txt](LLM4Rec_server/requirements.txt) for exact dependency specifications.

## Datasets

All datasets are hosted on Hugging Face and will be automatically downloaded when running the scripts.

### User & Restaurant Profiles

1. **[zetianli/CS329H_Project_user_profiles](https://huggingface.co/datasets/zetianli/CS329H_Project_user_profiles)**
   - 20,000 natural language user profiles generated from Yelp reviews
   - Includes user review history with ratings
   - Generated using GPT-based models (Openai/gpt-oss-120b)

2. **[zetianli/CS329H_Project_business](https://huggingface.co/datasets/zetianli/CS329H_Project_business)**
   - ~78,000 restaurant and non-restaurant business profiles
   - Includes sampled review comments and business metadata
   - Generated using GPT-based models

### DPO Training & Evaluation Data

3. **[zetianli/CS329H_DPO_FullList_train](https://huggingface.co/datasets/zetianli/CS329H_DPO_FullList_train)**
   - DPO training set with user-restaurant preference pairs
   - Constructed from users with rating gaps ≥ 2 stars

4. **[zetianli/CS329H_DPO_LFM_FullList_test_output](https://huggingface.co/datasets/zetianli/CS329H_DPO_LFM_FullList_test_output)**
   - Test set outputs for base and DPO-trained LFM2 models

5. **[zetianli/CS329H_DPO_Qwen_FullList_test_output](https://huggingface.co/datasets/zetianli/CS329H_DPO_Qwen_FullList_test_output)**
   - Test set outputs for base and DPO-trained Qwen3 models

6. **[HannahGrj/LLM4Rec_DPO_List_test_with_responses](https://huggingface.co/datasets/HannahGrj/LLM4Rec_DPO_List_test_with_responses)**
   - GPT-4o-mini baseline outputs on test set

### Data Format

**Training Input Format:**
```
User profile + Restaurant1 profile → Rating1
User profile + Restaurant2 profile → Rating2
...
User profile + RestaurantN profile → RatingN
```

## Reproducing Results

### Quick Start (Server Environment)

```bash
# 1. Download data
cd LLM4Rec_server
python download_data.py

# 2. Create DPO dataset
python create_dpo_dataset.py

# 3. Train on small dataset (for testing)
bash script/train_small.sh

# 4. Train on full dataset
bash script/train_full.sh

# 5. Run inference and evaluation
bash script/run_inference_eval.sh

# 6. Analyze results
python analyze_inference_results.py
```

### Reproducing Paper Results (Google Colab)

Each experimental setting in `LLM4Rec/` has a dedicated notebook:

1. **Open the desired notebook** in Google Colab
2. **Mount Google Drive** (for saving checkpoints)
3. **Run all cells sequentially**

| Notebook | Model | Setting | Produces |
|----------|-------|---------|----------|
| `hannah_qwen_list/dpo_train_hannah_list.ipynb` | Qwen3-0.6B | Length2-list | Metrics & model checkpoints |
| `hannah_qwen_single/dpo_train_hannah_single.ipynb` | Qwen3-0.6B | Single-item | Metrics & model checkpoints |
| `hannah_GPT4o_baseline/dpo_train_hannah3_baseline.ipynb` | GPT-4o-mini | Baseline | Baseline metrics |
| `justin_LFM_fulllist/justin_LFM_Fulllist.ipynb` | LFM2-350M | Full-list | Metrics & model checkpoints |
| `justin_Qwen_fulllist/justin_Qwen_Fulllist.ipynb` | Qwen3-0.6B | Full-list (alt) | Metrics & model checkpoints |
| `yanzhen_final_list/dpo_train_yanzhen_list.ipynb` | LFM2-350M | Length2-list | Metrics & model checkpoints |
| `yanzhen_final_single/dpo_train_yanzhen_single.ipynb` | LFM2-350M | Single-item | Metrics & model checkpoints |

**Expected outputs:**
- Training logs in Weights & Biases
- Model checkpoints saved to Google Drive
- Evaluation metrics (PPL, MAP, NDCG, Pairwise Accuracy) printed in notebook

### Generating Profiles (Optional)

To regenerate user/restaurant profiles from raw Yelp data:

```bash
cd profile_generation

# Generate user profiles
bash generate_profile_user.sh

# Generate restaurant profiles
bash generate_profile_resaurants.sh
```

Note: Profile generation requires access to GPT-based models and may incur API costs.

## Trained Models

The trained DPO model checkpoints are available and referenced in the notebooks. However, due to the size limitation, we cannot upload everything to Github. Links to use trained models via HuggingFace:

- [LFM2-350M DPO (full-list)](https://huggingface.co/HannahGrj/dpo-lfm-fullList-gap2-n5000)
- [LFM2-350M DPO (length2-list)](https://huggingface.co/HannahGrj/dpo-lfm2-350m-length2list-gap2-5000)
- [LFM2-350M DPO (single-item)](https://huggingface.co/HannahGrj/dpo-lfm2-350m-single-gap2-n5000)
- [Qwen3-0.6B DPO (full-list)](https://huggingface.co/HannahGrj/dpo-qwen-fullList-gap2-n5000)
- [Qwen3-0.6B DPO (length2-list)](https://huggingface.co/HannahGrj/dpo-qwen-length2list-gap2-n5000)
- [Qwen3-0.6B DPO (single-item)](https://huggingface.co/HannahGrj/dpo-qwen-single-gap2-n5000)

## Computational Requirements

### Hardware

**Recommended (used in our experiments):**
- GPU: NVIDIA A100 (40GB) on Google Colab
- RAM: 12GB+ system memory
- Storage: 20GB+ free space (for datasets and checkpoints)

**Minimum:**
- GPU: NVIDIA T4 (16GB) or equivalent
- RAM: 8GB+ system memory
- 4-bit quantization can reduce memory requirements

### Runtime Estimates

| Task | Hardware | Approximate Time |
|------|----------|------------------|
| Full DPO training (2 epochs, 2000 examples) | A100 (40GB) | 2-3 hours |
| Small dataset training (100 examples) | A100 (40GB) | 10-15 minutes |
| Inference on test set (200 examples) | A100 (40GB) | 5-10 minutes |
| Profile generation (20k users) | CPU/GPU | 4-6 hours |

**Total time to reproduce all experiments:** ~20 hours on A100 GPUs (excluding pre-trained model loading)

### Software

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- Google Colab (for LLM4Rec notebooks)

## Reproducibility

### Random Seeds

All stochastic processes use fixed random seeds for reproducibility:

- **Python/NumPy:** `RANDOM_SEED = 42`
- **PyTorch:** `seed = 42` (set in training configs)
- **Dataset splitting:** `random_state=42`
- **Data sampling:** `random.seed(42)`

Seeds are set in:
- [LLM4Rec_server/train_dpo.py:40](LLM4Rec_server/train_dpo.py#L40) (`--random_seed=42`)
- [LLM4Rec_server/create_small_dataset.py:23](LLM4Rec_server/create_small_dataset.py#L23)
- Each notebook in `LLM4Rec/` (cell with `RANDOM_SEED = 42`)

### Deterministic Execution

Training uses deterministic algorithms where possible:
- `use_seedable_sampler: true` in training configs
- Gradient checkpointing enabled for memory consistency
- Fixed learning rate schedules (no adaptive scheduling)

### Package Versions

While `requirements.txt` specifies minimum versions (e.g., `torch>=2.0.0`), our experiments used:
- torch: 2.9.0+cu126
- transformers: 4.57.2
- datasets: 4.0.0
- peft: 0.18.0
- trl: 0.25.1
- bitsandbytes: 0.48.2

For exact reproducibility, these versions are recommended (though not strictly required).

### Running Scripts End-to-End

All scripts and notebooks are designed to run without modification:

1. **No manual data preprocessing required** - datasets auto-download from Hugging Face
2. **No hardcoded paths** - all paths are configurable or relative
3. **No missing dependencies** - all requirements specified in `requirements.txt`
4. **Consistent output formats** - all scripts produce standard JSON/JSONL outputs

### Mapping Scripts to Paper Results

| Paper Section/Figure | Script/Notebook | Output |
|----------------------|-----------------|--------|
| Training metrics (Loss, PPL) | All training notebooks | Weights & Biases logs + notebook output |
| Evaluation metrics (MAP, NDCG) | `LLM4Rec_server/analyze_inference_results.py` | Console output + JSON files |
| Baseline comparison | `hannah_GPT4o_baseline/dpo_train_hannah3_baseline.ipynb` | Notebook output |
| Model performance comparison | All notebooks in `LLM4Rec/` | Combined metrics in W&B |




