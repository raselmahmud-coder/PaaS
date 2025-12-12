"""Term registry and extraction for semantic protocol."""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import numpy as np

from src.semantic.embedder import get_embedder

logger = logging.getLogger(__name__)


@dataclass
class Term:
    """A domain term with its definition and embedding."""
    
    name: str
    definition: str
    context: str = ""
    embedding: Optional[np.ndarray] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    source_agent: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without embedding)."""
        return {
            "name": self.name,
            "definition": self.definition,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source_agent": self.source_agent,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Term":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            definition=data["definition"],
            context=data.get("context", ""),
            source_agent=data.get("source_agent", ""),
        )


class TermRegistry:
    """Registry for managing domain terms and their embeddings.
    
    Each agent maintains its own term registry with definitions
    that can be compared with other agents for alignment.
    """
    
    # Default e-commerce terms
    DEFAULT_ECOMMERCE_TERMS = {
        "product": "A physical or digital item available for sale",
        "SKU": "Stock Keeping Unit - unique identifier for a product variant",
        "listing": "A product page on an e-commerce platform",
        "inventory": "Available quantity of a product in stock",
        "vendor": "A seller or merchant on the platform",
        "order": "A customer's purchase request",
        "fulfillment": "The process of preparing and shipping an order",
        "catalog": "Collection of all products available for sale",
        "price": "The monetary cost of a product",
        "discount": "A reduction in the regular price",
        "campaign": "A marketing initiative to promote products",
        "conversion": "When a visitor completes a desired action (purchase)",
    }
    
    def __init__(
        self,
        agent_id: str = "",
        load_defaults: bool = True,
    ):
        """Initialize the term registry.
        
        Args:
            agent_id: ID of the agent owning this registry.
            load_defaults: Whether to load default e-commerce terms.
        """
        self.agent_id = agent_id
        self._terms: Dict[str, Term] = {}
        self._embedder = get_embedder()
        
        if load_defaults:
            self._load_default_terms()
    
    def _load_default_terms(self) -> None:
        """Load default e-commerce terms."""
        for name, definition in self.DEFAULT_ECOMMERCE_TERMS.items():
            self.register_term(name, definition, source_agent="system")
    
    def register_term(
        self,
        name: str,
        definition: str,
        context: str = "",
        source_agent: str = "",
    ) -> Term:
        """Register a new term or update existing one.
        
        Args:
            name: Term name.
            definition: Term definition.
            context: Additional context for the term.
            source_agent: Agent that defined this term.
            
        Returns:
            The registered Term object.
        """
        source = source_agent or self.agent_id
        
        # Generate embedding for term + definition
        text_to_embed = f"{name}: {definition}"
        embedding = self._embedder.embed(text_to_embed)
        
        if name in self._terms:
            # Update existing term
            term = self._terms[name]
            term.definition = definition
            term.context = context
            term.embedding = embedding
            term.updated_at = datetime.utcnow()
            logger.debug(f"Updated term: {name}")
        else:
            # Create new term
            term = Term(
                name=name,
                definition=definition,
                context=context,
                embedding=embedding,
                source_agent=source,
            )
            self._terms[name] = term
            logger.debug(f"Registered term: {name}")
        
        return term
    
    def get_term(self, name: str) -> Optional[Term]:
        """Get a term by name."""
        return self._terms.get(name)
    
    def get_all_terms(self) -> List[Term]:
        """Get all registered terms."""
        return list(self._terms.values())
    
    def get_term_names(self) -> List[str]:
        """Get all term names."""
        return list(self._terms.keys())
    
    def get_definitions(self) -> Dict[str, str]:
        """Get all term definitions as a dictionary."""
        return {name: term.definition for name, term in self._terms.items()}
    
    def has_term(self, name: str) -> bool:
        """Check if a term is registered."""
        return name in self._terms
    
    def remove_term(self, name: str) -> bool:
        """Remove a term from the registry.
        
        Returns:
            True if term was removed, False if not found.
        """
        if name in self._terms:
            del self._terms[name]
            return True
        return False
    
    def get_embeddings_matrix(self) -> np.ndarray:
        """Get embeddings for all terms as a matrix.
        
        Returns:
            numpy array of shape (n_terms, embedding_dim).
        """
        embeddings = []
        for term in self._terms.values():
            if term.embedding is not None:
                embeddings.append(term.embedding)
        
        if not embeddings:
            return np.array([])
        
        return np.array(embeddings)
    
    def export_terms(self) -> List[Dict[str, Any]]:
        """Export all terms as dictionaries."""
        return [term.to_dict() for term in self._terms.values()]
    
    def import_terms(self, terms: List[Dict[str, Any]]) -> int:
        """Import terms from dictionaries.
        
        Args:
            terms: List of term dictionaries.
            
        Returns:
            Number of terms imported.
        """
        count = 0
        for term_data in terms:
            self.register_term(
                name=term_data["name"],
                definition=term_data["definition"],
                context=term_data.get("context", ""),
                source_agent=term_data.get("source_agent", ""),
            )
            count += 1
        return count
    
    def __len__(self) -> int:
        """Get number of terms."""
        return len(self._terms)
    
    def __contains__(self, name: str) -> bool:
        """Check if term exists."""
        return name in self._terms


# Global term registry
_global_registry: Optional[TermRegistry] = None


def get_term_registry(agent_id: str = "") -> TermRegistry:
    """Get or create the global term registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = TermRegistry(agent_id=agent_id)
    return _global_registry


def reset_term_registry() -> None:
    """Reset the global term registry."""
    global _global_registry
    _global_registry = None


def extract_terms_from_message(message: Dict[str, Any]) -> List[str]:
    """Extract domain terms from a protocol message.
    
    Looks for terms in:
    - Explicit 'terms' field
    - Keys in payload
    - Common e-commerce terms in text fields
    
    Args:
        message: Protocol message dictionary.
        
    Returns:
        List of extracted term names.
    """
    terms: Set[str] = set()
    
    # Check for explicit terms field
    if "terms" in message:
        terms.update(message["terms"])
    
    # Check payload for terms
    payload = message.get("payload", {})
    if isinstance(payload, dict):
        # Add relevant keys as potential terms
        for key in payload.keys():
            if key not in ["state", "timestamp", "message_id"]:
                terms.add(key)
    
    # Extract from body text if present
    body = message.get("body", {})
    if isinstance(body, dict):
        for key, value in body.items():
            if isinstance(value, str):
                # Look for common e-commerce terms
                for term in TermRegistry.DEFAULT_ECOMMERCE_TERMS.keys():
                    if term.lower() in value.lower():
                        terms.add(term)
    
    return list(terms)

