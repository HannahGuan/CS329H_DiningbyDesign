USER_PROFILE_GENERATION_PROMPT = USER_PROFILE_GENERATION_PROMPT = """You are a writer who creates concise, human-readable user profiles from raw data.

You will be given JSON-like data about a single user's data. This data may include:
- One or more reviews (with fields like "text", "stars", "business categories", "city", "state").
- Optional activity data (e.g., review_count, useful/funny/cool counts).
- Optional business attributes for the places they reviewed (e.g., ambience, price range, alcohol, outdoor seating).

Your task is to infer the user's tastes and personality and express them as a natural-language description.

<|USER's BASIC INFORMATION|>
{user_basic_information}
<|USER's BASIC INFORMATION|>

<|USER's REVIEWS|>
{user_review_info}
<|USER's REVIEWS|>

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




RESTAURANT_PROFILE_GENERATION_PROMPT = """You are a summarizer to elaborate on the fragmented information about a restaurant. Your goal is to generate a fluent description of a business restaurant based on the given json-like description. 

<|The Start of JSON Information>
{json_info}
<|The End of JSON Information|>

# Guidelines:
1. Your response should be only one paragraph. 
2. You should mention categories, attributes, and name of the restaurant. 
3. You should learn from the sampled reviews. 
4. You should not mention stars and ‘other reviewers’ explicitly. Instead, your style should be neutral."""

# Example Restaurant
"""
Target in Tucson, AZ is a large department store offering a broad range of products across categories such as fashion, home and garden, electronics, and furniture. The store is equipped with practical amenities including bike parking, wheelchair accessibility, a parking lot, and credit card acceptance, though it lacks services like takeout, delivery, catering, reservations, or outdoor seating. While the store is generally clean and organized, the customer experience is often marred by inefficient front-end operations, particularly around checkout. Shoppers report long waits and confusion due to a heavy reliance on self-checkout stations and limited staffed lanes, with employees sometimes appearing disengaged or unprepared to handle more complex transactions such as coupon redemption or regulated item purchases. Although inventory includes toys and collectible cards, some customers have noted a lack of variety and infrequent restocking. Overall, the store provides the expected convenience of a department store but may fall short in providing seamless in-person service.
"""
