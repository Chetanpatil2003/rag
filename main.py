"""
BYD Seal RAG API - Main FastAPI Application
"""
import logging
import uvicorn
from fastapi import FastAPI, HTTPException

from src.models.api_models import QuestionRequest, AnswerResponse
from src.pipeline.rag_pipeline import RAGPipeline
from src.config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI application
app = FastAPI(
    title="BYD Seal RAG API", 
    version="1.0.0",
    description="A RAG-based Q&A system for BYD Seal vehicle information"
)

# Global pipeline instance
pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline (without embeddings)"""
    global pipeline
    try:
        pipeline = RAGPipeline()
        logger.info("Pipeline initialized (vectorstores not ready yet)")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about BYD Seal"""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    if not pipeline.vectorstores_ready:
        raise HTTPException(
            status_code=400, 
            detail="Vectorstores not ready. Please call /embed first."
        )
    
    try:
        response = pipeline.ask(request.question)
        return response
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail="Error processing question")

@app.post("/embed")
async def embed_documents():
    """Trigger embedding and vectorstore creation"""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        pipeline.initialize_vectorstores()
        return {"status": "success", "message": "Documents embedded successfully"}
    except Exception as e:
        logger.error(f"Error embedding documents: {e}")
        raise HTTPException(status_code=500, detail="Error embedding documents")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "pipeline_ready": pipeline is not None,
        "vectorstores_ready": pipeline.vectorstores_ready if pipeline else False
    }

@app.get("/status")
async def status():
    """Get detailed system status"""
    if not pipeline:
        return {"status": "pipeline_not_initialized"}
    
    return {
        "pipeline_ready": True,
        "vectorstores_ready": pipeline.vectorstores_ready,
        "facts_vectorstore": pipeline.doc_processor.facts_vectorstore is not None,
        "external_vectorstore": pipeline.doc_processor.external_vectorstore is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)