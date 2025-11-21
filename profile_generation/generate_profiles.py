from openai import OpenAI
import os
import re
from constants import (
    USER_PROFILE_GENERATION_PROMPT,
    RESTAURANT_PROFILE_GENERATION_PROMPT,
)

import argparse
from datasets import load_dataset
import json
# get the port from the environment variable
port = os.getenv("PORT", 8693)
host = "liquid-gpu-012"

model =  OpenAI(
    base_url=os.getenv("dummy_url", f"http://{host}:{port}/v1"),
    api_key=os.getenv("dummy_api_key", "none"),
)



# need HPC version
def get_profile(generation_prompt: str) -> str:
    proflie = ""
    while(proflie == ""):
        response = model.chat.completions.create(
            model="gpt-oss-120b",
            messages=[
                {"role": "user", "content": generation_prompt},
            ],
        )
        text = response.choices[0].message.content
        match = re.search(r"<PROFILE>([\s\S]*?)<\/PROFILE>", text)
        if match:
            proflie = match.group(1).strip()

    return proflie

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
    removed_keys = ['sample_reviews', '__index_level_0__', 'business_id', "stars"]

    def generate_review_contents(reviews: list[str]) -> str:
        contents = ""
        for i in range(len(reviews)):
            contents += f"Review {i+1}:\n{reviews[i]}\n\n"
        return contents

    def remove_null_attributes(additional_info: dict) -> dict:
        pass

    sample_reviews_contents = generate_review_contents(sample_reviews)
    business_basic_information = ""
    return RESTAURANT_PROFILE_GENERATION_PROMPT.format(business_basic_information=business_basic_information, sample_reviews=sample_reviews_contents)

# still working
if __name__ == "__main__":
    #print(get_profile("Hello, are you running correctly?"))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile_type",
        type=str,
        default="user",
        choices=["user", "restaurant"],
        help="Type of profile to generate (e.g., user, restaurant)"
    )
    profile_type = parser.parse_args().profile_type
    print(f"Generating {profile_type} profiles")

    dataset = None
    if profile_type == "user":
        dataset = load_dataset("zetianli/CS329H_Project_user_profiles")
    elif profile_type == "restaurant":
        dataset = load_dataset("zetianli/CS329H_Project_business")

    point = dataset['train'][0]
    #print(process_user_prompt(point))
    print(get_profile(process_user_prompt(point)))
    