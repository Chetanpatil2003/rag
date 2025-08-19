"""
API data models for BYD Seal RAG system
"""
from typing import List
from pydantic import BaseModel, Field

class QuestionRequest(BaseModel):
    """Request model for asking questions"""
    question: str = Field(..., min_length=1, description="The question to ask about BYD Seal")

class Citation(BaseModel):
    """Citation information for sources used in answers"""
    source: str = Field(..., description="Source type (facts/external)")
    doc_id: str = Field(..., description="Document identifier")
    chunk_id: str = Field(..., description="Chunk identifier within document")

class AnswerResponse(BaseModel):
    """Response model for answers"""
    answer: str = Field(..., description="The generated answer")
    status: str = Field(..., description="Status of the response (answered/refused_sensitive/insufficient_info/error)")
    citations: List[Citation] = Field(default_factory=list, description="Sources used for the answer")

class EmbedResponse(BaseModel):
    """Response model for embedding endpoint"""
    status: str = Field(..., description="Status of embedding operation")
    message: str = Field(..., description="Descriptive message")
    
class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Health status")
    pipeline_ready: bool = Field(..., description="Whether pipeline is initialized")
    vectorstores_ready: bool = Field(default=False, description="Whether vectorstores are ready")

class StatusResponse(BaseModel):
    """Detailed status response"""
    pipeline_ready: bool = Field(..., description="Whether pipeline is initialized")
    vectorstores_ready: bool = Field(..., description="Whether vectorstores are ready")
    facts_vectorstore: bool = Field(..., description="Whether facts vectorstore exists")
    external_vectorstore: bool = Field(..., description="Whether external vectorstore exists")