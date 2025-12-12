"""LLM-based term negotiator for semantic handshake protocol."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NegotiationResult:
    """Result of a term negotiation."""
    
    term: str
    agreed_definition: str
    original_definition_a: str
    original_definition_b: str
    negotiation_rounds: int = 1
    success: bool = True
    explanation: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "term": self.term,
            "agreed_definition": self.agreed_definition,
            "original_definition_a": self.original_definition_a,
            "original_definition_b": self.original_definition_b,
            "negotiation_rounds": self.negotiation_rounds,
            "success": self.success,
            "explanation": self.explanation,
            "timestamp": self.timestamp.isoformat(),
        }


class TermNegotiator:
    """LLM-based negotiator for resolving semantic conflicts.
    
    Uses an LLM to propose unified definitions when two agents
    have different understandings of a term.
    """
    
    NEGOTIATION_PROMPT = """You are a semantic mediator helping two AI agents agree on a common definition for a term.

Term: "{term}"

Agent A's definition: "{definition_a}"
Agent A's context: {context_a}

Agent B's definition: "{definition_b}"
Agent B's context: {context_b}

Your task:
1. Analyze both definitions and their contexts
2. Propose a unified definition that:
   - Captures the essential meaning from both perspectives
   - Is precise and unambiguous
   - Works for both agents' use cases
   - Is concise (1-2 sentences max)

Respond with ONLY the agreed definition, nothing else."""

    EXPLANATION_PROMPT = """Explain briefly (1 sentence) why this unified definition works for both agents:

Term: "{term}"
Agent A's definition: "{definition_a}"
Agent B's definition: "{definition_b}"
Unified definition: "{unified_definition}"

Explanation:"""

    def __init__(
        self,
        llm: Optional[Any] = None,
        max_rounds: int = 3,
        fallback_strategy: str = "merge",
    ):
        """Initialize the negotiator.
        
        Args:
            llm: LLM instance (from src.llm.provider).
            max_rounds: Maximum negotiation rounds before fallback.
            fallback_strategy: Strategy when LLM fails ("merge", "first", "second").
        """
        self._llm = llm
        self.max_rounds = max_rounds
        self.fallback_strategy = fallback_strategy
        self._initialized = False
    
    def _ensure_llm(self) -> None:
        """Ensure LLM is available."""
        if self._llm is None:
            try:
                from src.llm.provider import get_llm
                self._llm = get_llm()
                self._initialized = True
            except Exception as e:
                logger.warning(f"Could not initialize LLM: {e}")
                self._initialized = False
    
    async def negotiate_term(
        self,
        term: str,
        definition_a: str,
        definition_b: str,
        context_a: str = "",
        context_b: str = "",
    ) -> NegotiationResult:
        """Negotiate a unified definition for a term.
        
        Args:
            term: The term to negotiate.
            definition_a: Agent A's definition.
            definition_b: Agent B's definition.
            context_a: Additional context for Agent A.
            context_b: Additional context for Agent B.
            
        Returns:
            NegotiationResult with the agreed definition.
        """
        self._ensure_llm()
        
        # If definitions are identical, no negotiation needed
        if definition_a.strip().lower() == definition_b.strip().lower():
            return NegotiationResult(
                term=term,
                agreed_definition=definition_a,
                original_definition_a=definition_a,
                original_definition_b=definition_b,
                negotiation_rounds=0,
                explanation="Definitions are identical",
            )
        
        # Try LLM-based negotiation
        if self._llm and self._initialized:
            try:
                return await self._negotiate_with_llm(
                    term, definition_a, definition_b,
                    context_a, context_b,
                )
            except Exception as e:
                logger.warning(f"LLM negotiation failed: {e}, using fallback")
        
        # Fallback strategy
        return self._fallback_negotiate(
            term, definition_a, definition_b,
        )
    
    async def _negotiate_with_llm(
        self,
        term: str,
        definition_a: str,
        definition_b: str,
        context_a: str,
        context_b: str,
    ) -> NegotiationResult:
        """Use LLM to negotiate a unified definition."""
        prompt = self.NEGOTIATION_PROMPT.format(
            term=term,
            definition_a=definition_a,
            definition_b=definition_b,
            context_a=context_a or "General e-commerce context",
            context_b=context_b or "General e-commerce context",
        )
        
        # Invoke LLM
        response = await self._llm.ainvoke(prompt)
        
        # Extract content
        if hasattr(response, "content"):
            unified_definition = response.content.strip()
        else:
            unified_definition = str(response).strip()
        
        # Get explanation (optional)
        explanation = ""
        try:
            explain_prompt = self.EXPLANATION_PROMPT.format(
                term=term,
                definition_a=definition_a,
                definition_b=definition_b,
                unified_definition=unified_definition,
            )
            explain_response = await self._llm.ainvoke(explain_prompt)
            if hasattr(explain_response, "content"):
                explanation = explain_response.content.strip()
            else:
                explanation = str(explain_response).strip()
        except Exception:
            pass
        
        logger.info(f"Negotiated '{term}': {unified_definition}")
        
        return NegotiationResult(
            term=term,
            agreed_definition=unified_definition,
            original_definition_a=definition_a,
            original_definition_b=definition_b,
            negotiation_rounds=1,
            success=True,
            explanation=explanation,
        )
    
    def _fallback_negotiate(
        self,
        term: str,
        definition_a: str,
        definition_b: str,
    ) -> NegotiationResult:
        """Fallback negotiation when LLM is unavailable."""
        if self.fallback_strategy == "first":
            unified = definition_a
        elif self.fallback_strategy == "second":
            unified = definition_b
        else:  # merge
            # Simple merge: combine both definitions
            if len(definition_a) > len(definition_b):
                unified = f"{definition_a}; alternatively: {definition_b}"
            else:
                unified = f"{definition_b}; alternatively: {definition_a}"
        
        return NegotiationResult(
            term=term,
            agreed_definition=unified,
            original_definition_a=definition_a,
            original_definition_b=definition_b,
            negotiation_rounds=1,
            success=True,
            explanation=f"Fallback strategy: {self.fallback_strategy}",
        )
    
    def negotiate_term_sync(
        self,
        term: str,
        definition_a: str,
        definition_b: str,
        context_a: str = "",
        context_b: str = "",
    ) -> NegotiationResult:
        """Synchronous version of negotiate_term."""
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.negotiate_term(
                term, definition_a, definition_b,
                context_a, context_b,
            )
        )
    
    async def negotiate_multiple(
        self,
        conflicts: List[Dict[str, str]],
    ) -> List[NegotiationResult]:
        """Negotiate multiple terms.
        
        Args:
            conflicts: List of dicts with 'term', 'definition_a', 'definition_b'.
            
        Returns:
            List of NegotiationResults.
        """
        results = []
        
        for conflict in conflicts:
            result = await self.negotiate_term(
                term=conflict["term"],
                definition_a=conflict.get("definition_a", ""),
                definition_b=conflict.get("definition_b", ""),
                context_a=conflict.get("context_a", ""),
                context_b=conflict.get("context_b", ""),
            )
            results.append(result)
        
        return results


# Global negotiator instance
_global_negotiator: Optional[TermNegotiator] = None


def get_negotiator() -> TermNegotiator:
    """Get or create the global term negotiator."""
    global _global_negotiator
    if _global_negotiator is None:
        _global_negotiator = TermNegotiator()
    return _global_negotiator


def reset_negotiator() -> None:
    """Reset the global negotiator."""
    global _global_negotiator
    _global_negotiator = None

