from openai import AsyncOpenAI
import asyncio
import os
import re
from constants import (
    USER_PROFILE_GENERATION_PROMPT,
    RESTAURANT_PROFILE_GENERATION_PROMPT,
)

from tqdm import tqdm
import argparse
from datasets import load_dataset
import json
# get the port from the environment variable
port = os.getenv("PORT", 8000)

model =  AsyncOpenAI(
    base_url=os.getenv("dummy_url", f"http://localhost:{port}/v1"),
    api_key=os.getenv("dummy_api_key", "none"),
)

batch_size = 256
semaphore = asyncio.Semaphore(batch_size)


async def get_profile(generation_prompt: str, index: int) -> str:
    proflie = ""
    while(proflie == ""):
        response = await model.chat.completions.create(
            model="gpt-oss-120b",
            messages=[
                {"role": "user", "content": generation_prompt},
            ],
            timeout=60,
        )
        text = response.choices[0].message.content
        match = re.search(r"<PROFILE>([\s\S]*?)<\/PROFILE>", text)
        if match:
            proflie = match.group(1).strip()
        else:
            print(f"Get prasing error for row {index+1}: Response: {text}\nretrying...")
    print(f"Generated profile for row {index+1}:\n{proflie}\n\n")
    return proflie

async def get_profile_batch(prompt: str, index: int) -> str:
    async with semaphore:
        return await get_profile(prompt, index)

def removed_nulls(review_data: list[dict]) -> list[dict]:
    new_reviews = []
    for review in review_data:
        att_key = list(review['attributes'].keys()) if review['attributes'] is not None else []
        for k in att_key:
            if review['attributes'][k] is None:
                del review['attributes'][k]
        if att_key == []:
            del review['attributes']
        new_reviews.append(review)
    return new_reviews

def process_user_prompt(data: dict) -> str:
    review_data = removed_nulls(data['reviews'])
    removed_keys = ['reviews', '__index_level_0__', 'user_id']
    user_name = data['name']
    for key in removed_keys:
        del data[key]

    def generate_review_contents(user_name: str, review: dict) -> str:
        contents = "---------------------------------------------------\n"
        business_name = review['name']
        contents += f"{user_name} reviewed {business_name} on {review['date']}:\n{review['text']}\n\n"

        additional_info = ""
        if 'attributes' in review:
            additional_info = json.dumps(review['attributes'], indent=2)
            del review['attributes']
        del review['text']

        basic_business_info = "Basic business information:\n" + json.dumps(review, indent=2)
        additional_info = "\nAdditional business information:\n" + additional_info
        contents += basic_business_info + "\n\n" + additional_info + "\n\n"
        return contents

    user_basic_information = json.dumps(data, indent=2)
    user_review_info = ""
    for i, review in enumerate(review_data):
        user_review_info += generate_review_contents(user_name, review)
    return USER_PROFILE_GENERATION_PROMPT.format(user_basic_information=user_basic_information, user_review_info=user_review_info)



def process_business_prompt(data: dict) -> str:
    sample_reviews = data['sample_reviews']
    atturbutes = data['attributes']
    removed_keys = ['sample_reviews', '__index_level_0__', 'business_id', "stars", "attributes"]
    for key in removed_keys:
        del data[key]

    def generate_review_contents(reviews: list[str]) -> str:
        contents = ""
        for i in range(len(reviews)):
            contents += "---------------------------------------------------\n"
            contents += f"Review {i+1}:\n{reviews[i]}\n\n"
        contents += "---------------------------------------------------\n"
        return contents

    def remove_null_attributes(attributes: dict) -> dict:
        att_key = list(attributes.keys()) if attributes is not None else []
        for k in att_key:
            if attributes[k] is None:
                del attributes[k]
        if att_key == []:
            return {}
        return attributes
        
    additional_info = ""
    atturbutes = remove_null_attributes(atturbutes)
    if atturbutes:
        additional_info = "\nAdditional business information:\n" + json.dumps(atturbutes, indent=2) + "\n\n"
    
    basic_business_info = "Basic business information:\n" + json.dumps(data, indent=2) + "\n\n"
    basic_business_info += additional_info

    sample_reviews_contents = generate_review_contents(sample_reviews)
    return RESTAURANT_PROFILE_GENERATION_PROMPT.format(business_basic_information=basic_business_info, sample_reviews=sample_reviews_contents)



async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile_type",
        type=str,
        default="user",
        choices=["user", "restaurant"],
        help="Type of profile to generate (e.g., user, restaurant)"
    )
    profile_type = parser.parse_args().profile_type

    dataset = None
    dataset_name = None
    if profile_type == "user":
        dataset_name = "zetianli/CS329H_Project_user_profiles"
        dataset = load_dataset(dataset_name)
    elif profile_type == "restaurant":
        dataset_name = "zetianli/CS329H_Project_business"
        dataset = load_dataset(dataset_name)
    else:
        raise ValueError(f"Invalid profile type: {profile_type}")
    
    prompts = []
    print(f"Generating {profile_type} prompts")
    for i, point in tqdm(enumerate(dataset['train'])):
        if profile_type == "user":
            prompt = process_user_prompt(point)
        elif profile_type == "restaurant":
            prompt = process_business_prompt(point)
        else:
            raise ValueError(f"Invalid profile type: {profile_type}")
        prompts.append(prompt)
    print(f"{profile_type} prompts generation completed")
    
    # generate profiles
    print(f"Generating {profile_type} profiles")
    tasks = [get_profile_batch(prompt, i) for i, prompt in tqdm(enumerate(prompts), desc="Generating profiles")]
    profiles = await asyncio.gather(*tasks)
    print(f"{profile_type} profiles generation completed")
    
    # save profiles to dataset
    dataset['train'] = dataset['train'].add_column("profile", profiles)
    #print(dataset['train'][:5])
    dataset.push_to_hub(dataset_name)
    print(f"{profile_type} profiles  updated to {dataset_name}")

# still working
if __name__ == "__main__":
    asyncio.run(main())
