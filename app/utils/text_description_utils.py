import os
from groq import Groq
from dotenv import load_dotenv

# load environment variables.
load_dotenv()

class TextDescription:

    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=GROQ_API_KEY)

    models = [
        "llama-3.1-8b-instant",
    ]

    token_class = {"short": 150, "moderate": 700, "long": 1536}

    # system prompt.
    system_prompt = """
    You are an advanced text–understanding AI. Your job is to read a user-provided text
    and produce a brief, clear, and concise description that preserves all the important,
    relevant, and meaningful information contained in the text.

    Guidelines:
    - “Brief” does NOT mean short. It means the description should avoid unnecessary sentences, redundancy, and filler.
    - Capture the core meaning, key points, and essential details of the text.
    - Preserve important context, intent, entities, actions, and relationships.
    - Do NOT summarize into overly short bullet points.
    - The output must be a coherent natural-language paragraph.
    - The description must be information-dense, accurate, and representative of the original content.
    - Do not invent or hallucinate information not present in the text.
    - Return only the description as plain text without labels or formatting.
    """

    def describe(self, input_content: str, token: int = 700, model=None, temperature: float = 0.0):
        """
        Reads a text, generates a clear, concise, but information-preserving description.
        """

        if model is None:
            model = self.models[0]

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": input_content}
                ],
                temperature=temperature,
                max_tokens=token,
                stream=False
            )

            # plain text output.
            description = response.choices[0].message.content.strip()
            return description

        except Exception as e:
            return f"Error generating text description: {str(e)}"