from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class AnalyzeResponse(BaseModel):
    enhancedDescription: str           
    tags: List[str]                
    metadata: Dict[str, Any]       
    success: bool               


class IndexRequest(BaseModel):
    token_id: str      
    title: str               
    description: str                 
    tags: List[str]                 


class IndexResponse(BaseModel):
    success: bool
    message: str


class SearchRequest(BaseModel):
    query: str                        


class SearchResultItem(BaseModel):
    tokenId: str
    description: str
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    queryTags: List[str]
    results: List[SearchResultItem]

