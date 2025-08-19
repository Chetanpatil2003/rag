"""
Configuration management for BYD Seal RAG API
"""
import os
from dataclasses import dataclass
from typing import List

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, skip
    pass

@dataclass
class Config:
    """Configuration class for the RAG system"""
    
    # API Configuration
    azure_deployment: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-35-turbo")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
    embedding_model: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
    
    # Model Configuration
    temperature: float = 0.0
    max_retries: int = 3
    retry_delay: float = 2.0
    
    # Document Processing
    chunk_size: int = 800
    chunk_overlap: int = 100
    batch_size: int = 20
    
    # Retrieval Configuration
    top_k_retrieval: int = 3
    
    # File Paths
    facts_file: str = "data/byd_seal_facts.md"
    external_file: str = "data/byd_seal_external.json"
    cache_dir: str = "cache"
    
    # Guardrails Configuration
    sensitive_topics: List[str] = None
    
    def __post_init__(self):
        """Initialize default sensitive topics if not provided"""
        if self.sensitive_topics is None:
            self.sensitive_topics = [
                "price", "pricing", "cost", "warranty", "guarantee", 
                "availability", "stock", "delivery", "shipping",
                "technical specifications", "performance numbers"
            ]

# Global config instance
config = Config()