# CS329H_DiningbyDesign

### Information Extraction

`user_reviews_filtered.json`: (user+review combe) for users with >=5 reviews

`business_with_reviews.json`: basic information of business involved, with sampled comments

`user_profiles.json`: basic information of users involved

All files in the link: https://drive.google.com/drive/folders/1Hlr-ADl5-UPdKwpL1gAmNRnVSo1Ohxxo?usp=share_link

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




# Next Step
Create each user's profile (Possibly by creating a prompt for each user based on reviews). ✅️ 11.22

Create each restaurant's profile, based on users' reviews and basic info. ✅️ 11.22

Implement listwise preference training; Run listwise training + baselines (direct prompting, DPO, BERT models?) 11.24 - 12.1 

Analyze result, what works and what doesn't, write a report. 12.1-12.3



