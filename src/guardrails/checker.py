"""
Guardrails implementation for safe RAG responses
"""
import logging
from typing import List

from langchain.schema import Document
from src.config import config

logger = logging.getLogger(__name__)

class GuardrailsChecker:
    """Implements guardrails for safe and reliable answers"""
    
    def __init__(self):
        self.sensitive_topics = config.sensitive_topics
        self.fabrication_indicators = [
            "approximately", "around", "roughly", "about $", "₹",
            "starting from", "priced at", "costs", "estimated",
            "likely", "probably", "might be", "could be"
        ]
    
    def is_sensitive_question(self, question: str) -> bool:
        """
        Check if question is about sensitive topics that require 
        authoritative sources
        
        Args:
            question: The user's question
            
        Returns:
            True if the question is about sensitive topics
        """
        question_lower = question.lower()
        is_sensitive = any(topic in question_lower for topic in self.sensitive_topics)
        
        if is_sensitive:
            logger.info(f"Question flagged as sensitive: {question}")
        
        return is_sensitive
    
    def validate_answer(self, answer: str, sources: List[Document]) -> bool:
        """
        Validate that answer is grounded in sources and doesn't contain
        fabricated information
        
        Args:
            answer: Generated answer
            sources: Source documents used
            
        Returns:
            True if answer passes validation checks
        """
        if not sources:
            logger.warning("No sources provided for answer validation")
            return False
        
        # Check if answer contains fabrication indicators not in sources
        source_content = " ".join([doc.page_content for doc in sources]).lower()
        answer_lower = answer.lower()
        
        fabrication_found = []
        for indicator in self.fabrication_indicators:
            if indicator in answer_lower and indicator not in source_content:
                fabrication_found.append(indicator)
        
        if fabrication_found:
            logger.warning(f"Potential fabrication indicators found: {fabrication_found}")
            return False
        
        # Additional validation: Check for specific patterns that indicate hallucination
        if self._contains_specific_numbers(answer, source_content):
            logger.warning("Answer contains specific numbers not found in sources")
            return False
        
        return True
    
    def _contains_specific_numbers(self, answer: str, source_content: str) -> bool:
        """
        Check if answer contains specific numerical claims not in sources
        This is a simplified check - in production, you'd use more sophisticated NLP
        """
        import re
        
        # Find numerical patterns in answer
        answer_numbers = re.findall(r'[\d,]+(?:\.\d+)?', answer)
        
        # If no numbers, no concern
        if not answer_numbers:
            return False
        
        # Check if these numbers appear in source content
        for number in answer_numbers:
            if number not in source_content:
                # Allow common numbers that are likely safe
                try:
                    num_val = float(number.replace(',', ''))
                    # Skip very common numbers (years, etc.)
                    if num_val < 100 or num_val > 2030:
                        continue
                    return True
                except ValueError:
                    continue
        
        return False
    
    def should_refuse_sensitive(self, is_sensitive: bool, has_facts: bool) -> bool:
        """
        Determine if we should refuse to answer a sensitive question
        based on available authoritative sources
        
        Args:
            is_sensitive: Whether question is about sensitive topics
            has_facts: Whether we have facts (authoritative sources)
            
        Returns:
            True if we should refuse to answer
        """
        return is_sensitive and not has_facts
    
    def get_refusal_message(self, question_type: str = "sensitive") -> str:
        """
        Get appropriate refusal message based on question type
        
        Args:
            question_type: Type of question being refused
            
        Returns:
            Appropriate refusal message
        """
        if question_type == "sensitive":
            return ("I cannot provide information about pricing, warranty, or technical "
                   "specifications as this information is not available in my reliable sources.")
        elif question_type == "insufficient_info":
            return "I don't have enough information to answer this question safely."
        elif question_type == "validation_failed":
            return ("I cannot provide a reliable answer based on the available sources. "
                   "Please try rephrasing your question or ask about topics covered in "
                   "the reliable documentation.")
        else:
            return "I cannot provide information on this topic."
    
    def log_guardrail_action(self, action: str, question: str, reason: str):
        """Log guardrail actions for monitoring and improvement"""
        logger.info(f"Guardrail action: {action} | Question: {question[:100]}... | Reason: {reason}")