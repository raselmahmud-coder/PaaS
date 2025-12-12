"""Semantic protocol layer for embedding-based inter-agent communication."""

from src.semantic.embedder import SemanticEmbedder, get_embedder
from src.semantic.similarity import (
    SemanticSimilarity,
    cosine_similarity,
    check_term_alignment,
)
from src.semantic.terms import (
    TermRegistry,
    Term,
    extract_terms_from_message,
    get_term_registry,
)
from src.semantic.handshake import (
    HandshakeState,
    HandshakeSession,
    HandshakeManager,
)
from src.semantic.negotiator import (
    TermNegotiator,
    NegotiationResult,
)

__all__ = [
    # Embedder
    "SemanticEmbedder",
    "get_embedder",
    # Similarity
    "SemanticSimilarity",
    "cosine_similarity",
    "check_term_alignment",
    # Terms
    "TermRegistry",
    "Term",
    "extract_terms_from_message",
    "get_term_registry",
    # Handshake
    "HandshakeState",
    "HandshakeSession",
    "HandshakeManager",
    # Negotiator
    "TermNegotiator",
    "NegotiationResult",
]

