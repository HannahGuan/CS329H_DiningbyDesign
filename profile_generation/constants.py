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

Your task is to infer the restaurant's key characteristics and customer-relevant qualities, and express them as a natural-language description.

Write a concise restaurant profile that can be used to **optimize restaurant recommendation**.
Focus on clear, explicit signals about cuisine, atmosphere, service, pricing, and practical features.

# Instructions:
1. Briefly identify the restaurant (mentioning its name is optional); only include location if it naturally supports understanding the setting.
2. Use the provided categories to describe the restaurant's cuisine, style, and overall identity.
3. Incorporate relevant attributes such as price level, ambiance, noise level, service style, parking, reservations, takeout/delivery, outdoor seating, kid- or group-friendliness, accessibility, and other practical conveniences.
4. Use the customer review snippets to infer the typical dining experience:
   - never referring to the snippets as “reviews,” “reviewers,” “ratings,” or “stars,”
   - paraphrasing insights rather than copying text,
   - keeping the tone neutral, balanced, and descriptive.
5. Do not invent details that are not supported by the data; avoid specifying dishes or décor features unless clearly implied.
6. If some fields are missing or sparse, simply omit them; do not comment on missing information.
7. Mention advantages and disadvantages in an impersonal, attribute-focused way. Do not use phrases like “some guests say,” “customers noted,” “people mentioned,” or any wording that references opinions or groups of diners. Instead, express drawbacks as general characteristics only when they are clearly implied (e.g., “portions may be modest” or “service can slow during peak hours”).
8. The paragraph should be smooth and natural, optimized for restaurant recommendation: concrete, attribute-rich, and focused on what guests can expect.


# Output requirements
- Output exactly ONE paragraph, wrapped in <PROFILE> and </PROFILE> tags.
- The paragraph must start with the restaurant's name (if available).
- Length: One long paragraph of plain text

Your final answer must be ONLY the <PROFILE> paragraph and nothing else.
"""

RESTAURANT_PROFILE_GENERATION_EXAMPLE = """
Freddy's Frozen Custard & Steakburgers in Meridian, ID offers a casual fast‑casual experience centered on classic burgers, hot dogs, and a variety of frozen custard and sundae desserts, positioned at the lowest price tier. The venue features counter service with no table reservations, a drive‑thru, and both takeout and delivery options, complemented by free Wi‑Fi, indoor TVs, and average‑level noise that suits families and groups. Practical conveniences include bike parking, a public parking lot, outdoor seating, and acceptance of credit cards, while the atmosphere remains relaxed and kid‑friendly. Guests can expect quick service even during busy periods, though burger patties may be on the thinner side; overall the menu’s strong points are the highly praised custard treats and solid burger flavors at budget‑friendly prices.
"""