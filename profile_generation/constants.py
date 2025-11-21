USER_PROFILE_GENERATION_PROMPT = USER_PROFILE_GENERATION_PROMPT = """You are a writer who creates concise, human-readable user profiles from raw data.

You will be given JSON-like data of user's basic information:


<|USER's BASIC INFORMATION|>
{user_basic_information}
<|USER's BASIC INFORMATION|>



Also, you will be given a list of reviews to the current user from different restaurants or businesses:

<|USER's REVIEWS|>
{user_review_info}
<|USER's REVIEWS|>


Your task is to infer the user's **dining preferences** and express them as a natural-language description.

Write a concise user profile that can be used to **optimize restaurant recommendation**
Focus on clear, explicit preference signals.

# Instructions:
1. Begin with 1-2 sentences summarizing the user's overall dining style and personality.
2. Describe preferred **cuisines and food types** (e.g., American comfort food, seafood, Mexican, sushi, burgers, vegetarian options), based only on restaurant-like businesses.
3. Describe preferred **ambiance and atmosphere** (e.g., casual, family-friendly, quiet, lively, sports-bar, upscale).
4. Describe expectations around **service**, **speed**, and **cleanliness**, using both restaurant and non-restaurant reviews when helpful.
5. Indicate typical **price range and value sensitivity** (e.g., budget-friendly, mid-range, fine dining, generous portions vs. small plates), if inferable.
6. Mention relevant **practical preferences** when clear (parking, takeout/delivery, reservations, outdoor seating, kid-friendliness, late-night hours, etc.).
7. Note how critical or forgiving the user tends to be (strict vs. easygoing), without mentioning “stars” or “ratings”.
8. Do not mention Yelp, reviews, or “other reviewers”.
9. Do not invent details that are not supported by the data.
10. Mention the user's **preferences** and **dislikes** in a neutral tone.

# Output requirements
- Output exactly ONE paragraph, wrapped in <PROFILE> and </PROFILE> tags.
- The paragraph must start with the user's name (if available).
- Length: One long paragraph of plain text

Your final answer must be ONLY the <PROFILE> paragraph and nothing else."""




USER_PROFILE_GENERATION_EXAMPLE = """
Jonathon is a relaxed, social eater who gravitates toward hearty American comfort foods such as seafood pastas, burgers, fries, cobb salads, and classic sandwiches like Philly cheesesteak, preferring venues that serve these dishes well. He favors casual, family‑friendly spots with a laid‑back bar vibe—often sports‑oriented or lively pubs that offer free Wi‑Fi, outdoor seating, and ample parking (bike racks or surface lots) and that accommodate groups, kids, and takeout or delivery. Service matters to him: he values friendly, prompt staff and will note delays (e.g., waiting for a check) but remains forgiving if the food is hot, fresh, and flavorful. He typically chooses mid‑range establishments (price level around 2) that provide good value and moderate portions, and he likes full‑bar or beer‑and‑wine options, casual attire, and average noise levels. Practical comforts such as credit‑card acceptance, wheelchair accessibility, and a clean, well‑maintained environment are important, while ambience that feels overly dingy is a minor drawback. Overall, Jonathon is easygoing but attentive to service quality and a welcoming atmosphere.
"""


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
