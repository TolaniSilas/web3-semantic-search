import os
from groq import Groq
import base64
from PIL import Image
from dotenv import load_dotenv


# load the environment variables from the .env file.
load_dotenv()


def encode_image_to_base64(image_path: str) -> str:
    """
    reads an image file and converts it into a base64-encoded string.
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def AnalyzeImageWithAI(image_path: str, api_key: str) -> str:
    """
    sends an image to the groq vision model (Llama 4 Scout 17B)
    and returns the description of an image.

    Args:
        image_path (str): path to the image file
        api_key (str): your Groq API key

    Returns:
        str: image description result returned by the model
    """

    # convert image to base64 string.
    base64_image = encode_image_to_base64(image_path)

    # initialize Groq client.
    client = Groq(api_key=api_key)

    # query groq vision model.
    chat_completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image? Provide a detailed and thorough description of every visible element. Examine the image meticulously and describe all objects, people, actions, surroundings, text, colors, and any subtle or fine details present."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
    )

    # extract description text.
    return chat_completion.choices[0].message.content