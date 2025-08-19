"""
Main RAG pipeline implementation using LangGraph
"""
import logging
from typing import List

from langgraph.graph import StateGraph, END
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from src.llm.factory import initialize_llm
from src.processing.document_processor import DocumentProcessor
from src.guardrails.checker import GuardrailsChecker
from src.models.api_models import AnswerResponse, Citation
from src.models.graph_state import RAGState
from src.config import config

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline orchestrating the entire process"""
    
    def __init__(self):
        self.llm = initialize_llm()
        self.doc_processor = DocumentProcessor()
        self.guardrails = GuardrailsChecker()
        
        # Don't create vectorstores on init
        self.vectorstores_ready = False
        self.graph = self._create_graph()

    def initialize_vectorstores(self):
        """Initialize vectorstores manually via /embed route"""
        self.doc_processor.create_vectorstores()
        self.vectorstores_ready = True
        logger.info("Vectorstores initialized successfully")
    
    def _create_graph(self):
        """Create LangGraph for RAG pipeline"""
        
        # Build graph
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("retrieve_facts", self._retrieve_facts)
        workflow.add_node("check_sensitivity", self._check_sensitivity)
        workflow.add_node("retrieve_external", self._retrieve_external)
        workflow.add_node("generate_answer", self._generate_answer)
        
        # Add edges
        workflow.add_edge("retrieve_facts", "check_sensitivity")
        workflow.add_edge("check_sensitivity", "retrieve_external")
        workflow.add_edge("retrieve_external", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        # Set entry point
        workflow.set_entry_point("retrieve_facts")
        
        return workflow.compile()
    
    def _retrieve_facts(self, state: RAGState) -> RAGState:
        """Retrieve relevant facts from authoritative sources"""
        question = state["question"]
        
        if self.doc_processor.facts_vectorstore:
            docs = self.doc_processor.facts_vectorstore.similarity_search(
                question, k=config.top_k_retrieval
            )
            state["retrieved_facts"] = docs
            
            # Check if we have sufficient facts
            if docs and any(len(doc.page_content.strip()) > 50 for doc in docs):
                state["needs_external"] = False
            else:
                state["needs_external"] = True
        else:
            state["retrieved_facts"] = []
            state["needs_external"] = True
        
        return state
    
    def _check_sensitivity(self, state: RAGState) -> RAGState:
        """Check if question is about sensitive topics"""
        state["is_sensitive"] = self.guardrails.is_sensitive_question(state["question"])
        return state
    
    def _retrieve_external(self, state: RAGState) -> RAGState:
        """Retrieve from external sources if needed and safe"""
        logger.info("Checking if external sources are needed")
        
        if state["needs_external"] and not state["is_sensitive"]:
            if self.doc_processor.external_vectorstore:
                docs = self.doc_processor.external_vectorstore.similarity_search(
                    state["question"], k=config.top_k_retrieval
                )
                state["retrieved_external"] = docs
            else:
                state["retrieved_external"] = []
        else:
            state["retrieved_external"] = []
        
        return state
    
    def _generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer based on retrieved documents"""
        question = state["question"]
        facts = state.get("retrieved_facts", [])
        external = state.get("retrieved_external", [])
        
        # Prioritize facts over external
        all_sources = facts + external
        
        if not all_sources:
            state["answer"] = self.guardrails.get_refusal_message("insufficient_info")
            state["status"] = "insufficient_info"
            state["citations"] = []
            self.guardrails.log_guardrail_action("refuse", question, "no_sources")
            return state
        
        # If sensitive topic and no facts, refuse
        if self.guardrails.should_refuse_sensitive(state["is_sensitive"], bool(facts)):
            state["answer"] = self.guardrails.get_refusal_message("sensitive")
            state["status"] = "refused_sensitive"
            state["citations"] = []
            self.guardrails.log_guardrail_action("refuse", question, "sensitive_no_facts")
            return state
        
        # Generate answer
        try:
            answer = self._generate_llm_response(question, all_sources)
            
            # Validate answer
            if self.guardrails.validate_answer(answer, all_sources):
                state["answer"] = answer
                state["status"] = "answered"
                state["citations"] = self._extract_citations(all_sources)
            else:
                state["answer"] = self.guardrails.get_refusal_message("validation_failed")
                state["status"] = "validation_failed"
                state["citations"] = []
                self.guardrails.log_guardrail_action("refuse", question, "validation_failed")
                
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            state["answer"] = "I encountered an error while processing your question."
            state["status"] = "error"
            state["citations"] = []
        
        return state
    
    def _generate_llm_response(self, question: str, sources: List[Document]) -> str:
        """Generate LLM response using sources"""
        # Create prompt
        prompt = PromptTemplate(
            template="""You are a helpful assistant answering questions about BYD Seal vehicles. 

STRICT RULES:
1. Only use the provided context to answer
2. Never make up or hallucinate information
3. If context doesn't contain the answer, say so
4. Cite sources for each claim using [source:doc_id:chunk_id] format
5. Be concise and accurate

Context:
{context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )
        
        # Prepare context
        context = "\n\n".join([
            f"Source: {doc.metadata['source']} | Doc: {doc.metadata['doc_id']} | Chunk: {doc.metadata['chunk_id']}\n{doc.page_content}"
            for doc in sources
        ])
        
        # Generate answer
        response = self.llm.invoke(
            prompt.format(context=context, question=question)
        )
        return response.content
    
    def _extract_citations(self, sources: List[Document]) -> List[Citation]:
        """Extract citations from source documents"""
        citations = []
        for doc in sources:
            citations.append(Citation(
                source=doc.metadata["source"],
                doc_id=doc.metadata["doc_id"],
                chunk_id=doc.metadata["chunk_id"]
            ))
        return citations
    
    def ask(self, question: str) -> AnswerResponse:
        """Process question through RAG pipeline"""
        if not self.vectorstores_ready:
            return AnswerResponse(
                answer="Vectorstores not initialized. Please call /embed endpoint first.",
                status="not_ready",
                citations=[]
            )
        
        initial_state = RAGState(
            question=question,
            retrieved_facts=[],
            retrieved_external=[],
            answer="",
            citations=[],
            status="processing",
            needs_external=False,
            is_sensitive=False
        )
        
        # Run through graph
        final_state = self.graph.invoke(initial_state)
        
        return AnswerResponse(
            answer=final_state["answer"],
            status=final_state["status"],
            citations=final_state["citations"]
        )