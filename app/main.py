# import the necessary libraries.
import io
import os
from typing import Optional, Dict, List
from PIL import Image
import tempfile
import shutil

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from groq import Groq

from app.config import CHROMA_DIR
from app.models.schemas import AnalyzeResponse, IndexRequest, IndexResponse, SearchRequest, SearchResultItem, SearchResponse
from app.utils.img_description_utils import AnalyzeImageWithAI
from app.utils.tag_extraction import TagExtractor
from app.utils.text_description_utils import TextDescription
from app.utils.extract_user_query import QueryExtractor



# instantiate fastapi.
app = FastAPI(title="Web3 Semantic Search - AI Service")

# # configure the ChromaDB client
# settings = Settings(
#     chroma_db_impl="duckdb+parquet",   # backend storage type.
#     persist_directory=CHROMA_DIR,    # folder to store the DB files.
# )

# initialize and configure the ChromaDB client (keep persist_directory as CHROMA_DIR).

client = chromadb.PersistentClient(path=CHROMA_DIR)
COLLECTION_NAME = "semantic-contents"

# ensure collection exists.
try:
    # try to get the existing collection.
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Collection {COLLECTION_NAME} retrieved successfully.")
except Exception:
    # if the collection doesn't exist, create it.
    collection = client.create_collection(name=COLLECTION_NAME)
    print(f"Collection {COLLECTION_NAME} created successfully.")


# load the environment variables from the .env file.
load_dotenv()



@app.get("/health")
async def health():
    return {
        "application": "AI Service",
        "message": "Running Successfully"
    }



@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(None),
    textContent: Optional[str] = Form(None), 
    userDescription: str = Form(...), 
    title: str = Form(...),          
):
    """
    analyze content (image or typed text) before minting.
    requires one content source (file OR textContent), a user description, and a title.
    """

    has_file = file and file.filename
    has_text = textContent is not None and len(textContent.strip()) > 0

    if has_file and has_text:
        raise HTTPException(status_code=400, 
            detail="Cannot submit both a file and typed text content. Please choose one source."
        )
    if not has_file and not has_text:
        raise HTTPException(status_code=400, 
            detail="Must provide either an image file or textual content. Use the file upload for images and text field for blog content")

    # first scenario is for the image content.
    if has_file:
        file_bytes = await file.read()
        
        # perform image verification using PIL.
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img.verify()
    
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. File upload is reserved for images in PNG, JPG, or JPEG format. Please use the text field for blog content.")
        
            
        # temporary file handling, and perform ai analysis.
        ai_description = ""
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        try:
            # write the image bytes to the temporary path.
            with open(temp_file_path, "wb") as buffer:
                buffer.write(file_bytes)

            # load the groq api key from the env file.
            GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
            
            # pass the temporary path to your AI model handler.
            ai_description = AnalyzeImageWithAI(temp_file_path, api_key=GROQ_API_KEY)
            
        except Exception as e:
            # log the exception for debugging.
            print(f"Error during image analysis: {e}") 
            ai_description = ""
            
        finally:
            # clean up the temporary directory and file.
            shutil.rmtree(temp_dir)

        # generate tags from the image description provided by the vision models.
        tag_extractor = TagExtractor()
        generated_tags = tag_extractor.generate_tags(ai_description, tag_extractor.token_class["short"])

        enhanced = ""
        if userDescription:
            enhanced += userDescription.strip() + " "
        enhanced += ai_description.strip()

        metadata = {
            "type": "image",
            "title": title,
            "description": enhanced,
            "tags": generated_tags,
        }

        return AnalyzeResponse(
            enhancedDescription=enhanced,
            tags=generated_tags,
            metadata=metadata,
            success=True
        )

    # second scenario is for the text content.
    elif has_text:
        text_content = textContent

        # get the ai description using the llm model.
        tag_extractor = TagExtractor()
        text_model = TextDescription()
        ai_description = text_model.describe(text_content, text_model.token_class["long"])

        # generate tags with llm.
        generated_tags = tag_extractor.generate_tags(ai_description, tag_extractor.token_class["long"])

        # generate an enhanced description.
        enhanced = ""
        if userDescription:
            enhanced += userDescription.strip() + " "
        enhanced += ai_description.strip()

        metadata = {
            "type": "blog_text", 
            "title": title,
            "description": enhanced,
            "tags": generated_tags
        }

        return AnalyzeResponse(
            enhancedDescription=enhanced,
            tags=generated_tags,
            metadata=metadata,
            success=True
        )
    
    # should not be reachable due to initial validation.
    return JSONResponse(status_code=500, content={"detail": "Content processing failed unexpectedly."})



# post minting stage.
@app.post("/index", response_model=IndexResponse)
async def index_content(req: IndexRequest):
    """
    store metadata after minting.
    uses token_id as ChromaDB document ID.
    """

    tags_string = " ".join(req.tags)

    try:
        collection.add(
            ids=[req.token_id],
            documents=[req.description],
            metadatas=[{
                "tags": tags_string, 
                "title": req.title
                }]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        

    return IndexResponse(
        success=True,
        message="Metadata successfully stored in ChromaDB."
    )


# api endpoint to search for related contents in the chroma db.
@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """
    convert user query to ai tags, which will be used to search the ChromaDB and return
    only ids, documents, and metadatas.
    """
    # ai converts query to semantic tags.
    query_model = QueryExtractor()
    obtained_queries = query_model.extract_queries(req.query, query_model.token_class["moderate"])


    try:
        # query ChromaDB for matches.
        raw = collection.query(
            query_texts=obtained_queries,
            n_results=10,
            include=["documents", "metadatas"]
    )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    result_items = []

    # chroma returns lists inside lists, then flatten them.
    ids = raw.get("ids", [[]])[0]
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]

    for i in range(len(ids)):
        result_items.append(
            SearchResultItem(
                tokenId=ids[i],
                description=docs[i],
                metadata=metas[i]
            )
        )

    # return structured response.
    return SearchResponse(
        queryTags=obtained_queries,
        results=result_items
    )




if __name__ == "__main__":
    
    # import the uvicorn module for asynchronous running.
    import uvicorn
    
    print("preparing api endpoints...")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)