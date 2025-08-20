# BYD Seal RAG API

A Retrieval-Augmented Generation (RAG) system for answering questions about BYD Seal vehicles with built-in guardrails for safe and reliable responses.

## Features

- **Multi-source RAG**: Combines authoritative facts with external data sources
- **Intelligent Guardrails**: Prevents hallucination and ensures safe responses
- **Rate-limited Embeddings**: Handles large datasets with batching and caching
- **Modular Architecture**: Clean separation of concerns for maintainability
- **Multiple LLM Support**: Works with Azure OpenAI and Google Gemini
- **Comprehensive Logging**: Full observability of the RAG pipeline

## Architecture

```
src/
├── config.py                    # Configuration management
├── main.py                      # FastAPI application entry point
├── models/
│   ├── api_models.py            # Pydantic models for API
│   └── graph_state.py           # LangGraph state definitions
├── llm/
│   └── factory.py               # LLM and embeddings factory
├── processing/
│   └── document_processor.py    # Document loading and processing
├── guardrails/
│   └── checker.py               # Safety and validation logic
└── pipeline/
    └── rag_pipeline.py          # Main RAG pipeline orchestration
```

## Quick Start

### Prerequisites

- Python 3.8+
- API keys for your chosen LLM provider:
  - **Google**: `GOOGLE_API_KEY`
  - **Azure**: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`

### Installation

1. setup venv
```bash
mac-
python3 -m venv venv
source ./venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
rename .env.example to .env and add required credentials

or
# For Google Gemini (default)
export GOOGLE_API_KEY="your-api-key"

# Or for Azure OpenAI
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_CHAT_DEPLOYMENT=""
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT=""
```



### Running the Application

1. Start the API server:
```bash
python main.py
```

Or 


2.  Run with Docker Dekstop (Optional):
```bash
docker build -t rag-app .
```
```bash
docker run -d -p 8001:8000 --name my-rag-container rag-app

```


2. Initialize the vectorstores (one-time setup):
```bash
curl -X POST "http://localhost:8001/embed"
```

3. Ask questions:
```bash
curl -X POST "http://localhost:8001/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the safety features of BYD Seal?"}'
```

## API Endpoints

### Core Endpoints

- **POST /ask**: Ask questions about BYD Seal
- **POST /embed**: Initialize document embeddings
- **GET /health**: Basic health check
- **GET /status**: Detailed system status

### Example Usage

```python
import requests

# Initialize embeddings (do this once)
response = requests.post("http://localhost:8001/embed")
print(response.json())

# Ask a question
response = requests.post(
    "http://localhost:8001/ask",
    json={"question": "What are the key features of BYD Seal?"}
)
answer = response.json()
print(f"Answer: {answer['answer']}")
print(f"Status: {answer['status']}")
print(f"Citations: {answer['citations']}")
```

## Configuration

The system can be configured via environment variables or by modifying `src/config.py`:

### Key Configuration Options

```python
# Model settings
AZURE_OPENAI_CHAT_DEPLOYMENT = "gpt-35-turbo"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = "text-embedding-3-large"

# Processing settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K_RETRIEVAL = 3
BATCH_SIZE = 20

# File paths
FACTS_FILE = "data/byd_seal_facts.md"
EXTERNAL_FILE = "data/byd_seal_external.json"
```


## Guardrails System

The system implements multiple layers of safety:

1. **Sensitive Topic Detection**: Identifies questions about pricing, warranties, etc.
2. **Source Validation**: Ensures answers are grounded in provided sources
3. **Fabrication Prevention**: Detects and blocks potentially fabricated information
4. **Authoritative Source Prioritization**: Prefers official facts over external sources

See [DESIGN.md](DESIGN.md) for detailed guardrails documentation.

