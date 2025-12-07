import os
from groq import Groq
import json
from dotenv import load_dotenv


# load the environment variables from the .env file.
load_dotenv()

# create a tag extractor class.
class TagExtractor():
    
    # load the groq api key from the env file.
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    
    client = Groq(api_key=GROQ_API_KEY)
    
    query:str
    
    models = [
    "llama-3.1-8b-instant",
    ]

    token_class = {"short": 150, "moderate": 700, "long": 1536}

    system_prompt = """
                You are an intelligent generative assistant specialized in extracting relevant tags from multimedia content.
                Your goal is to provide concise, accurate, and descriptive tags that effectively capture the essence of the content.
                These tags will be used for indexing, search, and categorization.

                Rules:
                - Generate **3 to 4 tags** per content.
                - Tags must be concise, highly relevant, and effectively describe the content.
                - Return tags as a **JSON list of strings**.

                Few-shot examples:

                Example 1:
                Content: A high-tempo Afrobeats song designed for workouts.
                Tags: ["afrobeats", "high-tempo", "workout"]

                Example 2:
                Content: A serene landscape photo of mountains during sunset.
                Tags: ["landscape", "mountains", "sunset", "nature"]

                Example 3:
                Content: Educational video about AI and machine learning fundamentals.
                Tags: ["education", "AI", "machine learning", "tutorial"]

                Instructions:
                - Always return **3-4 tags**. If content is simple, choose the most descriptive tags.
                - Tags will be used for on-chain storage (NFT attributes), AI DB indexing, and UI/search filtering.
                - Make sure the tags **effectively capture the content**.
                - Finally, return only the tags. Don't include any content, just return only the tags as the model output.
                """
        
    def generate_tags(self, input_content, token, model=models[0], temperature=0):
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": input_content}
                ],
                temperature=temperature,
                response_format={"type": "text"},
                max_tokens=token,
                stream=False
            )  

            raw_output = response.choices[0].message.content
            
            # remove code block fences.
            cleaned = raw_output.replace("```json", "").replace("```", "").strip() 
            
            # convert to python list.
            tags = json.loads(cleaned)
            
            return tags
            
        
        except Exception as e:
            return {"Error": str(e)}