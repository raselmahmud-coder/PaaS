"""Semantic embedder using Sentence-Transformers for term embedding."""

import logging
from typing import Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

# Global embedder instance
_global_embedder: Optional["SemanticEmbedder"] = None


class SemanticEmbedder:
    """Wrapper for Sentence-Transformers model for semantic embeddings.
    
    Provides caching and batch processing for efficient embedding generation.
    """
    
    # Default model - small but effective for semantic similarity
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_embeddings: bool = True,
        device: Optional[str] = None,
    ):
        """Initialize the semantic embedder.
        
        Args:
            model_name: Name of the Sentence-Transformers model to use.
            cache_embeddings: Whether to cache embeddings for repeated terms.
            device: Device to run the model on ('cpu', 'cuda', or None for auto).
        """
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings
        self._cache: Dict[str, np.ndarray] = {}
        self._model = None
        self._device = device
        
        # Lazy load the model
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        """Lazy initialization of the model."""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading Sentence-Transformers model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                device=self._device,
            )
            self._initialized = True
            logger.info(f"Model loaded successfully on device: {self._model.device}")
            
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: "
                "poetry add sentence-transformers"
            )
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embedding(s) for text.
        
        Args:
            text: Single string or list of strings to embed.
            
        Returns:
            numpy array of shape (embedding_dim,) for single text,
            or (n_texts, embedding_dim) for list of texts.
        """
        self._ensure_initialized()
        
        # Handle single string
        if isinstance(text, str):
            return self._embed_single(text)
        
        # Handle list of strings
        return self._embed_batch(text)
    
    def _embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        # Check cache
        if self.cache_embeddings and text in self._cache:
            return self._cache[text]
        
        # Generate embedding
        embedding = self._model.encode(text, convert_to_numpy=True)
        
        # Cache if enabled
        if self.cache_embeddings:
            self._cache[text] = embedding
        
        return embedding
    
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts efficiently."""
        if not texts:
            return np.array([])
        
        # Separate cached and uncached texts
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if self.cache_embeddings and text in self._cache:
                cached_embeddings[i] = self._cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = self._model.encode(uncached_texts, convert_to_numpy=True)
            
            # Cache new embeddings
            if self.cache_embeddings:
                for text, emb in zip(uncached_texts, new_embeddings):
                    self._cache[text] = emb
            
            # Add to cached_embeddings dict
            for idx, emb in zip(uncached_indices, new_embeddings):
                cached_embeddings[idx] = emb
        
        # Reconstruct in order
        result = np.array([cached_embeddings[i] for i in range(len(texts))])
        return result
    
    def embed_terms(self, terms: List[str]) -> np.ndarray:
        """Embed a list of domain terms.
        
        Args:
            terms: List of terms to embed.
            
        Returns:
            numpy array of shape (n_terms, embedding_dim).
        """
        return self.embed(terms)
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        self._ensure_initialized()
        return self._model.get_sentence_embedding_dimension()
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
    
    @property
    def cache_size(self) -> int:
        """Get the number of cached embeddings."""
        return len(self._cache)


def get_embedder(
    model_name: str = SemanticEmbedder.DEFAULT_MODEL,
    force_new: bool = False,
) -> SemanticEmbedder:
    """Get or create the global semantic embedder instance.
    
    Args:
        model_name: Model name to use.
        force_new: If True, create a new instance even if one exists.
        
    Returns:
        SemanticEmbedder instance.
    """
    global _global_embedder
    
    if force_new or _global_embedder is None:
        _global_embedder = SemanticEmbedder(model_name=model_name)
    
    return _global_embedder


def reset_embedder() -> None:
    """Reset the global embedder instance."""
    global _global_embedder
    _global_embedder = None

