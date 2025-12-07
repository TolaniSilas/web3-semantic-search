import os

# where Chroma will persist its DB.
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_data")
