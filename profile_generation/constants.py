USER_PROFILE_GENERATION_PROMPT = USER_PROFILE_GENERATION_PROMPT = """You are a writer who creates concise, human-readable user profiles from raw data.

You will be given JSON-like data of user's basic information:


<|USER's BASIC INFORMATION|>
{user_basic_information}
<|USER's BASIC INFORMATION|>



Also, you will be given a list of reviews to the current user from different restaurants or businesses:

<|USER's REVIEWS|>
{user_review_info}
<|USER's REVIEWS|>

Your task is to infer the user's tastes and personality and express them as a natural-language description.

# Output requirements
- Output exactly ONE paragraph, wrapped in <PROFILE> and </PROFILE> tags.
- The paragraph must start with the user's name (if available), e.g., "Jonathon enjoys..." or "Matthew is the kind of person who...".
- Length: about 3-6 sentences.
- Do NOT mention that you are reading JSON or data, and do not list field names or quote raw keys.

# Content guidelines
1. Focus on the user's preferences and habits inferred from the reviews, such as:
   - Types of places they enjoy (e.g., seafood, burgers, wine bars, hotels, casual spots, family-friendly venues).
   - Price sensitivity (e.g., prefers casual mid-range places vs. high-end spots).
   - Atmosphere preferences (casual vs. upscale, lively vs. quiet, family-friendly vs. adult-oriented).
   - Attitude toward service, food quality, and consistency (e.g., values friendly service, hot and fresh food, etc.).
   - Any consistent patterns (e.g., often returns to places they like, appreciates variety, tends to recommend places to others).

2. Infer personality traits from their tone and comments:
   - Are they easy-going or demanding?
   - Do they emphasize friendliness, cleanliness, speed, value, or atmosphere?
   - Do they sound enthusiastic, critical, adventurous, etc.?

3. You must NOT:
   - Mention any numeric values explicitly (no star ratings, no counts, no dates, no exact prices).
   - Mention specific field names like "review_count", "business_id", "attributes", or "categories".
   - Refer to the text as "reviews", "data", or "JSON"; just describe the person.
   - Reveal identifiable information like exact dates or IDs.

4. If information is limited or inconsistent:
   - Still write a brief, coherent profile using hedged language (e.g., "seems to enjoy", "appears to value").
   - Do not invent detailed facts that clearly conflict with the data.

Your final answer must be ONLY the <PROFILE> paragraph and nothing else."""



# EXAMPLE USER PROFILE
# Jonathon enjoys a relaxed, casual dining scene, ranging from fresh seafood and hearty pastas to juicy burgers and 
# classic cheesesteaks, and he often highlights hot, fresh food and friendly service as the deciding factors. He gravitates
# toward moderately priced venues that offer a comfortable vibe, free Wi‑Fi and easy parking, and he’s quick to return to spots
# that deliver consistent quality. When it comes to accommodations, he appreciates spacious, clean rooms and attentive staff, 
# especially when the location is convenient for local events. Overall, his tone suggests an easy‑going, social personality
# that values reliability, good value and a welcoming atmosphere.




RESTAURANT_PROFILE_GENERATION_PROMPT = """
You are a writer creating concise profiles of restaurants from structured data.

You are given a JSON-like object describing one business:

<|BUSINESS BASIC INFORMATION|>
{business_basic_information}
<|BUSINESS BASIC INFORMATION|>



Also, you will be given a list of reviews to the current business:

<|SAMPLE REVIEWS TO CURRENT BUSINESS|>
{sample_reviews}
<|SAMPLE REVIEWS TO CURRENT BUSINESS|>

TASK
Write a single, fluent paragraph that summarizes what this restaurant is like.

REQUIREMENTS
1. The output must be exactly one paragraph (no headings, bullet points, or line breaks).
2. Clearly mention the restaurant's **name** and **location** (city and state, if available).
3. Refer to the **categories** to describe the type of restaurant (e.g., cuisine, setting, or business type).
4. Use relevant **attributes** to describe practical details such as price level, ambiance, noise, reservations, parking, takeout/delivery, outdoor seating, suitability for groups/families, etc.
5. Read the **sample_reviews** and use them to infer the overall atmosphere, service style, food quality, and typical customer experience, but:
   - Do NOT mention “reviews”, “reviewers”, “ratings”, “stars”, or “Yelp”.
   - Paraphrase instead of copying text verbatim.
6. Keep the tone neutral and descriptive, not overly positive or negative.
7. Do not invent specific details that are not suggested by the JSON (e.g., menu items, exact prices, or décor themes that are not implied).
8. If some fields (attributes, reviews, etc.) are missing or empty, simply omit them; do not say that the information is missing.

OUTPUT FORMAT
- Return only the final paragraph as plain text, with no quotes around it and no additional explanation.
"""
