import os
import json
from groq import Groq
from dotenv import load_dotenv

# load environment variables.
load_dotenv()

class QueryExtractor:

    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=GROQ_API_KEY)

    models = ["llama-3.1-8b-instant"]

    token_class = {"short": 150, "moderate": 700, "long": 1536}

    system_prompt = """
    You are an intelligent semantic search query generator. 
    Your job is to take a user's natural-language query and produce a small set of highly relevant, 
    concise, meaningful search phrases that best represent the user’s true intent.

    Guidelines:
    - Generate 2 to 4 short search queries.
    - Each query must be a short phrase, not a full sentence.
    - Phrases must be descriptive, meaningful, and excellent for vector-based semantic search.
    - Capture the underlying topic, themes, and intent of the user query.
    - Avoid unrelated or overly broad keywords.
    - Do NOT generate tags, categories, labels, or hashtags.
    - Return ONLY a JSON list of strings. No explanations or commentary.

    Examples:

    Example 1:
    User Query: "Show me philosophical writings about the nature of suffering."
    Output: ["philosophy of suffering", "human condition", "existential pain"]

    Example 2:
    User Query: "I want images of futuristic cities with neon lights."
    Output: ["futuristic neon city", "cyberpunk skyline", "sci-fi metropolis"]

    Example 3:
    User Query: "Give me content about African culture, traditions, and folklore."
    Output: ["african traditions", "cultural heritage africa", "african folklore"]

    Follow the instructions strictly.
    Return only a JSON list of 2–4 search phrases.
    """

    def extract_queries(self, user_query: str, token: int = 150, model=None, temperature: float = 0):
        if model is None:
            model = self.models[0]

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_query},
                ],
                temperature=temperature,
                max_tokens=token,
                response_format={"type": "text"},
                stream=False
            )

            raw_output = response.choices[0].message.content.strip()

            # cleanup the ai raw output.
            cleaned = raw_output.replace("```json", "").replace("```", "").strip()

            # parse json list.
            queries = json.loads(cleaned)

            return queries

        except Exception as e:
            return {"Error": str(e)}
