"""Semantic similarity computation for term alignment checking."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.semantic.embedder import SemanticEmbedder, get_embedder

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.
    
    Args:
        a: First vector.
        b: Second vector.
        
    Returns:
        Cosine similarity value between -1 and 1.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between two sets of vectors.
    
    Args:
        a: First set of vectors, shape (n, d).
        b: Second set of vectors, shape (m, d).
        
    Returns:
        Similarity matrix of shape (n, m).
    """
    # Normalize vectors
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    
    # Compute similarity matrix
    return np.dot(a_norm, b_norm.T)


@dataclass
class AlignmentResult:
    """Result of a term alignment check."""
    
    aligned: bool
    similarity_scores: Dict[str, float] = field(default_factory=dict)
    conflicts: List[str] = field(default_factory=list)
    best_matches: Dict[str, Tuple[str, float]] = field(default_factory=dict)
    threshold: float = 0.85
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "aligned": self.aligned,
            "similarity_scores": self.similarity_scores,
            "conflicts": self.conflicts,
            "best_matches": {
                k: {"term": v[0], "similarity": v[1]} 
                for k, v in self.best_matches.items()
            },
            "threshold": self.threshold,
        }


class SemanticSimilarity:
    """Semantic similarity checker for inter-agent term alignment.
    
    Checks whether two agents share a common understanding of domain terms
    by comparing their semantic embeddings.
    """
    
    DEFAULT_THRESHOLD = 0.85
    
    def __init__(
        self,
        embedder: Optional[SemanticEmbedder] = None,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        """Initialize the similarity checker.
        
        Args:
            embedder: SemanticEmbedder instance. Uses global if None.
            threshold: Similarity threshold for alignment (0.0 to 1.0).
        """
        self.embedder = embedder or get_embedder()
        self.threshold = threshold
    
    def check_alignment(
        self,
        terms_a: List[str],
        terms_b: List[str],
        threshold: Optional[float] = None,
    ) -> AlignmentResult:
        """Check semantic alignment between two sets of terms.
        
        Compares terms from agent A with terms from agent B to find
        potential semantic conflicts (terms with different meanings).
        
        Args:
            terms_a: Terms from agent A.
            terms_b: Terms from agent B.
            threshold: Custom threshold (uses default if None).
            
        Returns:
            AlignmentResult with conflict details.
        """
        threshold = threshold or self.threshold
        
        if not terms_a or not terms_b:
            return AlignmentResult(
                aligned=True,
                threshold=threshold,
            )
        
        # Get embeddings
        emb_a = self.embedder.embed(terms_a)
        emb_b = self.embedder.embed(terms_b)
        
        # Compute similarity matrix
        sim_matrix = cosine_similarity_matrix(emb_a, emb_b)
        
        # Find conflicts and best matches
        conflicts = []
        similarity_scores = {}
        best_matches = {}
        
        for i, term_a in enumerate(terms_a):
            # Find best match in terms_b
            best_idx = np.argmax(sim_matrix[i])
            best_sim = sim_matrix[i][best_idx]
            best_term = terms_b[best_idx]
            
            similarity_scores[term_a] = float(best_sim)
            best_matches[term_a] = (best_term, float(best_sim))
            
            # Check if below threshold
            if best_sim < threshold:
                conflicts.append(term_a)
                logger.debug(
                    f"Semantic conflict: '{term_a}' best match '{best_term}' "
                    f"(similarity: {best_sim:.3f} < {threshold})"
                )
        
        return AlignmentResult(
            aligned=len(conflicts) == 0,
            similarity_scores=similarity_scores,
            conflicts=conflicts,
            best_matches=best_matches,
            threshold=threshold,
        )
    
    def compute_similarity(self, term_a: str, term_b: str) -> float:
        """Compute similarity between two terms.
        
        Args:
            term_a: First term.
            term_b: Second term.
            
        Returns:
            Cosine similarity between the terms.
        """
        emb_a = self.embedder.embed(term_a)
        emb_b = self.embedder.embed(term_b)
        return cosine_similarity(emb_a, emb_b)
    
    def find_similar_terms(
        self,
        term: str,
        candidates: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Find the most similar terms from a list of candidates.
        
        Args:
            term: Term to find matches for.
            candidates: List of candidate terms.
            top_k: Number of top matches to return.
            
        Returns:
            List of (term, similarity) tuples, sorted by similarity.
        """
        if not candidates:
            return []
        
        term_emb = self.embedder.embed(term)
        candidate_embs = self.embedder.embed(candidates)
        
        # Compute similarities
        similarities = [
            cosine_similarity(term_emb, cand_emb)
            for cand_emb in candidate_embs
        ]
        
        # Sort by similarity
        ranked = sorted(
            zip(candidates, similarities),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return ranked[:top_k]


def check_term_alignment(
    terms_a: List[str],
    terms_b: List[str],
    threshold: float = SemanticSimilarity.DEFAULT_THRESHOLD,
) -> AlignmentResult:
    """Convenience function to check term alignment.
    
    Args:
        terms_a: Terms from agent A.
        terms_b: Terms from agent B.
        threshold: Similarity threshold.
        
    Returns:
        AlignmentResult with conflict details.
    """
    checker = SemanticSimilarity(threshold=threshold)
    return checker.check_alignment(terms_a, terms_b)

