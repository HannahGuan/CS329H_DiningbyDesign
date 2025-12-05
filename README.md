# CS329H_DiningbyDesign

### Information Extraction

### Hugging Face
[**zetianli/CS329H_Project_user_profiles**](https://huggingface.co/datasets/zetianli/CS329H_Project_user_profiles) : user profiles & their reviews

[**zetianli/CS329H_Project_business**](https://huggingface.co/datasets/zetianli/CS329H_Project_business) : basic information of business involved, with sampled comments


Input to training: 

User profile + restaurant1's profile + score1

User profile + restaurant2's profile + score2

User profile + restaurant3's profile + score3

User profile + restaurant4's profile + score4

User profile + restaurant5's profile + score5

Inference:
Perplexity of the User profile + the restaurant's profile in the trained model.

### Data Size

User Profiles = 20k

Restaurants + Non-Restaurants Profiles = ~78k

### Model to Generate Profiles and Processing: **Openai/gpt-oss-120b**
### Model Candidates to fine tune:
  - LFM2s: 350M
  - Qwen3s: 0.6B
### Parameters Update: LoRA
### Evaluation: PPL chosen&rejected, diff; MAP, NDCG, PairwiseAcc




