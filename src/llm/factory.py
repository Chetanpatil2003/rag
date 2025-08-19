"""
LLM and Embeddings factory for different providers
"""
import os
import logging
from typing import Optional

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.config import config

logger = logging.getLogger(__name__)

class LLMProvider:
    """Enum-like class for LLM providers"""
    AZURE = "azure"
    GOOGLE = "google"

def get_llm_provider() -> str:
    """Determine which LLM provider to use based on environment"""
    if os.getenv("GOOGLE_API_KEY"):
        return LLMProvider.GOOGLE
    elif os.getenv("AZURE_OPENAI_API_KEY"):
        return LLMProvider.AZURE
    else:
        # Default to Google (as in original code)
        return LLMProvider.GOOGLE

def initialize_llm(provider: Optional[str] = None):
    """Initialize LLM based on provider"""
    if provider is None:
        provider = get_llm_provider()
    
    logger.info(f"Initializing LLM with provider: {provider}")
    
    if provider == LLMProvider.GOOGLE:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=config.temperature,
            max_tokens=None,
            timeout=None,
            max_retries=config.max_retries,
        )
    elif provider == LLMProvider.AZURE:
        return AzureChatOpenAI(
            deployment_name=config.azure_deployment,
            api_version=config.api_version,
            temperature=config.temperature,
            max_retries=config.max_retries,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def initialize_embeddings(provider: Optional[str] = None):
    """Initialize embeddings based on provider"""
    if provider is None:
        provider = get_llm_provider()
    
    logger.info(f"Initializing embeddings with provider: {provider}")
    
    if provider == LLMProvider.GOOGLE:
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
    elif provider == LLMProvider.AZURE:
        return AzureOpenAIEmbeddings(
            deployment=config.embedding_model,
            api_version=config.api_version,
        )
    else:
        raise ValueError(f"Unsupported embeddings provider: {provider}")