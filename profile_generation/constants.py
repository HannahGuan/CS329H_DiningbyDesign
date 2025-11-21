USER_PROFILE_GENERATION_PROMPT = """You are a summarizer to elaborate on the fragmented information about a user. Your goal is to generate a fluent description based on the given json-like description. 

<|The Start of JSON Information>
{user_json}
<|The End of JSON Information|> 

# Guidelines: 
1. Your response should be only one paragraph, starting with user’s name.
2. You should infer the user’s personality and preference from their reviews and activity data.
3. You should not mention numeric data explicitly. """

# Example User 
"""
Jonathon is a seasoned and sociable Yelp user who has been part of the community for many years, sharing thoughtful feedback across a wide range of dining and hospitality experiences. With a large network of friends and a consistently high average rating, Jonathon appears to appreciate quality service, flavorful food, and relaxed, casual atmospheres. His reviews often highlight hot, fresh meals and attentive service, though he is not hesitant to deduct a star when customer service falls short. He enjoys trying a mix of traditional American fare, seafood, and pub-style eateries, often favoring establishments that offer reliable comfort food, good value, and a clean setting. His style is straightforward and sincere, with a strong focus on practicality and personal satisfaction rather than flair, and he seems to enjoy revisiting places that leave a positive impression. Overall, Jonathon comes across as an easygoing but discerning customer who values consistency, cleanliness, and hearty meals in friendly environments.
"""

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
