"""
Document processing module for loading and chunking documents
"""
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

from src.config import config
from src.llm.factory import initialize_embeddings

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading, processing, and vectorstore creation"""
    
    def __init__(self):
        self.embeddings = initialize_embeddings()
        self.text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self.facts_vectorstore: Optional[FAISS] = None
        self.external_vectorstore: Optional[FAISS] = None
        
    def load_facts(self, file_path: str) -> List[Document]:
        """Load and process facts from markdown file"""
        try:
            logger.info(f"Loading facts from {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split the content into chunks
            texts = self.text_splitter.split_text(content)
            
            # Create documents with metadata
            documents = []
            for i, text in enumerate(texts):
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": "facts",
                        "doc_id": f"F{i+1}",
                        "chunk_id": f"c{i+1}",
                        "file_path": file_path
                    }
                )
                documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} fact documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading facts: {e}")
            return []
    
    def load_external(self, file_path: str) -> List[Document]:
        """Load and process external data from JSON file"""
        try:
            logger.info(f"Loading external data from {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            for i, item in enumerate(data):
                # Extract transcript content
                transcript_content = self._extract_transcript_content(item)
                
                if not transcript_content or len(transcript_content.strip()) < 20:
                    continue  # Skip empty or very short content
                
                # Create metadata
                metadata = {
                    "source": "external",
                    "doc_id": f"E{i+1}",
                    "chunk_id": "c1",
                    "file_path": file_path,
                    "video_id": item.get('video_id', ''),
                    "title": item.get('title', ''),
                    "brand": item.get('brand', ''),
                    "product": item.get('product', '')
                }
                
                # Split content if too long
                documents.extend(self._create_document_chunks(transcript_content, metadata))
            
            logger.info(f"Loaded {len(documents)} external documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading external data: {e}")
            return []
    
    def _extract_transcript_content(self, item: dict) -> str:
        """Extract transcript content from various possible formats"""
        if 'transcriptText' in item and 'content' in item['transcriptText']:
            return item['transcriptText']['content']
        elif 'transcript' in item:
            return item['transcript']
        return ""
    
    def _create_document_chunks(self, content: str, base_metadata: dict) -> List[Document]:
        """Create document chunks from content"""
        documents = []
        
        if len(content) > config.chunk_size:
            texts = self.text_splitter.split_text(content)
            for j, text in enumerate(texts):
                if len(text.strip()) > 20:  # Skip very short chunks
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata["chunk_id"] = f"c{j+1}"
                    documents.append(Document(
                        page_content=text,
                        metadata=chunk_metadata
                    ))
        else:
            documents.append(Document(
                page_content=content,
                metadata=base_metadata
            ))
        
        return documents
    
    def create_vectorstores(self):
        """Main method to create vector stores - use cache if available"""
        cache_dir = Path(config.cache_dir)
        cache_dir.mkdir(exist_ok=True)
        
        facts_cache = cache_dir / "facts_vectorstore"
        external_cache = cache_dir / "external_vectorstore"
        
        # Try to load from cache first
        self._load_from_cache(facts_cache, external_cache)
        
        # Create from scratch if cache loading failed
        if self.facts_vectorstore is None:
            logger.info("Creating vector stores from scratch...")
            self._create_vectorstores_from_scratch()
        elif self.external_vectorstore is None and Path(config.external_file).exists():
            logger.info("Creating external vector store from scratch...")
            external_docs = self.load_external(config.external_file)
            if external_docs:
                self.external_vectorstore = self._create_vectorstore_batched(external_docs)
        
        # Save to cache
        self._save_to_cache(facts_cache, external_cache)
    
    def _load_from_cache(self, facts_cache: Path, external_cache: Path):
        """Load vectorstores from cache"""
        if facts_cache.exists():
            logger.info("Loading facts vector store from cache...")
            try:
                self.facts_vectorstore = FAISS.load_local(
                    str(facts_cache), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Facts vector store loaded from cache")
            except Exception as e:
                logger.warning(f"Failed to load facts cache: {e}")
                self.facts_vectorstore = None
        
        if external_cache.exists():
            logger.info("Loading external vector store from cache...")
            try:
                self.external_vectorstore = FAISS.load_local(
                    str(external_cache), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("External vector store loaded from cache")
            except Exception as e:
                logger.warning(f"Failed to load external cache: {e}")
                self.external_vectorstore = None
    
    def _save_to_cache(self, facts_cache: Path, external_cache: Path):
        """Save vectorstores to cache"""
        try:
            if self.facts_vectorstore:
                self.facts_vectorstore.save_local(str(facts_cache))
                logger.info("Facts vector store saved to cache")
            
            if self.external_vectorstore:
                self.external_vectorstore.save_local(str(external_cache))
                logger.info("External vector store saved to cache")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def _create_vectorstores_from_scratch(self):
        """Create vectorstores from scratch with rate limiting"""
        # Load documents
        facts_docs = self.load_facts(config.facts_file)
        external_docs = self.load_external(config.external_file)
        
        if not facts_docs:
            raise ValueError("No facts documents loaded")
        
        logger.info("Creating vector stores with rate limiting...")
        
        # Create facts vector store
        logger.info(f"Processing {len(facts_docs)} facts documents")
        try:
            self.facts_vectorstore = FAISS.from_documents(facts_docs, self.embeddings)
            logger.info("Facts vector store created successfully")
        except Exception as e:
            logger.error(f"Error creating facts vector store: {e}")
            raise
        
        # Create external vector store with batching
        if external_docs:
            logger.info(f"Processing {len(external_docs)} external documents in batches")
            self.external_vectorstore = self._create_vectorstore_batched(external_docs)
        
        logger.info("All vector stores created successfully")
    
    def _create_vectorstore_batched(self, documents: List[Document]) -> Optional[FAISS]:
        """Create vector store with batching to handle rate limits"""
        if not documents:
            return None
            
        try:
            # Start with first batch
            batch_size = min(config.batch_size, len(documents))
            first_batch = documents[:batch_size]
            
            logger.info(f"Creating initial vector store with {len(first_batch)} documents")
            vectorstore = FAISS.from_documents(first_batch, self.embeddings)
            
            # Add remaining documents in batches
            remaining_docs = documents[batch_size:]
            
            for i in range(0, len(remaining_docs), batch_size):
                batch = remaining_docs[i:i + batch_size]
                logger.info(f"Adding batch {i//batch_size + 2}: {len(batch)} documents")
                
                try:
                    # Add batch to existing vector store
                    batch_texts = [doc.page_content for doc in batch]
                    batch_metadatas = [doc.metadata for doc in batch]
                    vectorstore.add_texts(batch_texts, batch_metadatas)
                    
                    # Add delay between batches to respect rate limits
                    if i + batch_size < len(remaining_docs):
                        logger.info(f"Waiting {config.retry_delay} seconds before next batch...")
                        time.sleep(config.retry_delay)
                        
                except Exception as e:
                    logger.warning(f"Error processing batch {i//batch_size + 2}: {e}")
                    logger.info("Waiting longer before retry...")
                    time.sleep(config.retry_delay * 2)
                    continue
            
            logger.info(f"External vector store created with {len(documents)} total documents")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error creating batched vector store: {e}")
            return None