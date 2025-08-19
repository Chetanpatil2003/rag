"""
State definitions for LangGraph RAG pipeline
"""
from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from src.models.api_models import Citation

class RAGState(TypedDict):
    """State for RAG pipeline graph"""
    question: str
    retrieved_facts: List[Document]
    retrieved_external: List[Document]
    answer: str
    citations: List[Citation]
    status: str
    needs_external: bool
    is_sensitive: bool