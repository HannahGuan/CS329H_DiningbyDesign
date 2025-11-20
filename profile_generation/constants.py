USER_PROFILE_GENERATION_PROMPT = """fill in"""


RESTAURANT_PROFILE_GENERATION_PROMPT = """You are a summarizer to elaborate on the fragmented information about a restaurant. Your goal is to generate a fluent description of a business restaurant based on the given json-like description. 

<|The Start of JSON Information>
{json_info}
<|The End of JSON Information|>

# Guidelines:
1. Your response should be only one paragraph. 
2. You should mention categories, attributes, and name of the restaurant. 
3. You should learn from the sampled reviews. 
4. You should not mention stars and ‘other reviewers’ explicitly. Instead, your style should be neutral."""
