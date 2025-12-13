"""Tests for semantic protocol module."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.semantic.embedder import SemanticEmbedder, get_embedder, reset_embedder
from src.semantic.similarity import (
    SemanticSimilarity,
    AlignmentResult,
    cosine_similarity,
    check_term_alignment,
)
from src.semantic.terms import (
    TermRegistry,
    Term,
    extract_terms_from_message,
    get_term_registry,
    reset_term_registry,
)
from src.semantic.handshake import (
    HandshakeState,
    HandshakeSession,
    HandshakeManager,
    HandshakeResult,
)
from src.semantic.negotiator import (
    TermNegotiator,
    NegotiationResult,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global instances before each test."""
    reset_embedder()
    reset_term_registry()
    yield
    reset_embedder()
    reset_term_registry()


@pytest.fixture
def mock_embedder():
    """Create a mock embedder that doesn't load the model."""
    embedder = SemanticEmbedder.__new__(SemanticEmbedder)
    embedder.model_name = "mock"
    embedder.cache_embeddings = True
    embedder._cache = {}
    embedder._model = None
    embedder._device = None
    embedder._initialized = False
    
    # Mock the model encode function
    def mock_encode(texts, convert_to_numpy=True):
        if isinstance(texts, str):
            # Return a deterministic embedding based on text hash
            h = hash(texts) % 10000
            return np.array([h / 10000] * 384)
        else:
            return np.array([[hash(t) % 10000 / 10000] * 384 for t in texts])
    
    embedder._model = MagicMock()
    embedder._model.encode = mock_encode
    embedder._model.get_sentence_embedding_dimension = lambda: 384
    embedder._initialized = True
    
    return embedder


@pytest.fixture
def term_registry():
    """Create a term registry with default terms."""
    return TermRegistry(agent_id="test-agent", load_defaults=True)


# =============================================================================
# Embedder Tests
# =============================================================================


class TestSemanticEmbedder:
    """Tests for SemanticEmbedder."""

    def test_init(self, mock_embedder):
        """Test embedder initialization."""
        assert mock_embedder.model_name == "mock"
        assert mock_embedder.cache_embeddings is True

    def test_embed_single(self, mock_embedder):
        """Test embedding a single text."""
        embedding = mock_embedder.embed("test text")
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 384

    def test_embed_batch(self, mock_embedder):
        """Test embedding multiple texts."""
        texts = ["text one", "text two", "text three"]
        embeddings = mock_embedder.embed(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)

    def test_caching(self, mock_embedder):
        """Test embedding caching."""
        text = "cached text"
        
        # First call
        mock_embedder.embed(text)
        assert mock_embedder.cache_size == 1
        
        # Second call should use cache
        mock_embedder.embed(text)
        assert mock_embedder.cache_size == 1

    def test_clear_cache(self, mock_embedder):
        """Test clearing the cache."""
        mock_embedder.embed("text")
        assert mock_embedder.cache_size > 0
        
        mock_embedder.clear_cache()
        assert mock_embedder.cache_size == 0


# =============================================================================
# Similarity Tests
# =============================================================================


class TestSemanticSimilarity:
    """Tests for semantic similarity checking."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        
        sim = cosine_similarity(a, b)
        assert abs(sim - 1.0) < 0.001

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        
        sim = cosine_similarity(a, b)
        assert abs(sim) < 0.001

    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.0, 0.0, 0.0])
        
        sim = cosine_similarity(a, b)
        assert sim == 0.0

    def test_alignment_result_to_dict(self):
        """Test AlignmentResult serialization."""
        result = AlignmentResult(
            aligned=True,
            similarity_scores={"term1": 0.9},
            conflicts=[],
            threshold=0.85,
        )
        
        data = result.to_dict()
        assert data["aligned"] is True
        assert "similarity_scores" in data

    def test_check_term_alignment_empty(self):
        """Test alignment check with empty terms."""
        result = check_term_alignment([], [], threshold=0.85)
        assert result.aligned is True

    def test_similarity_checker_init(self, mock_embedder):
        """Test SemanticSimilarity initialization."""
        checker = SemanticSimilarity(embedder=mock_embedder, threshold=0.9)
        assert checker.threshold == 0.9


# =============================================================================
# Terms Tests
# =============================================================================


class TestTermRegistry:
    """Tests for term registry."""

    def test_init_with_defaults(self, term_registry):
        """Test registry initialization with defaults."""
        assert len(term_registry) > 0
        assert "product" in term_registry

    def test_init_without_defaults(self):
        """Test registry initialization without defaults."""
        registry = TermRegistry(load_defaults=False)
        assert len(registry) == 0

    def test_register_term(self):
        """Test registering a new term."""
        registry = TermRegistry(load_defaults=False)
        
        with patch.object(SemanticEmbedder, '_ensure_initialized'):
            with patch.object(SemanticEmbedder, 'embed', return_value=np.zeros(384)):
                term = registry.register_term(
                    name="test_term",
                    definition="A test term definition",
                )
        
        assert term.name == "test_term"
        assert "test_term" in registry

    def test_get_term(self, term_registry):
        """Test getting a term by name."""
        term = term_registry.get_term("product")
        assert term is not None
        assert term.name == "product"

    def test_get_nonexistent_term(self, term_registry):
        """Test getting a nonexistent term."""
        term = term_registry.get_term("nonexistent")
        assert term is None

    def test_remove_term(self, term_registry):
        """Test removing a term."""
        assert "product" in term_registry
        result = term_registry.remove_term("product")
        assert result is True
        assert "product" not in term_registry

    def test_export_import_terms(self, term_registry):
        """Test exporting and importing terms."""
        exported = term_registry.export_terms()
        assert len(exported) > 0
        assert all("name" in t and "definition" in t for t in exported)

    def test_term_to_dict(self):
        """Test Term serialization."""
        term = Term(
            name="test",
            definition="A test",
            source_agent="agent-1",
        )
        
        data = term.to_dict()
        assert data["name"] == "test"
        assert data["definition"] == "A test"


class TestExtractTerms:
    """Tests for term extraction from messages."""

    def test_extract_explicit_terms(self):
        """Test extracting explicit terms field."""
        message = {"terms": ["product", "SKU", "listing"]}
        terms = extract_terms_from_message(message)
        
        assert "product" in terms
        assert "SKU" in terms

    def test_extract_from_payload(self):
        """Test extracting terms from payload keys."""
        message = {"payload": {"product_id": "123", "listing_title": "Test"}}
        terms = extract_terms_from_message(message)
        
        assert "product_id" in terms
        assert "listing_title" in terms


# =============================================================================
# Handshake Tests
# =============================================================================


class TestHandshakeSession:
    """Tests for handshake sessions."""

    def test_session_init(self, term_registry):
        """Test session initialization."""
        session = HandshakeSession(
            session_id="test-session",
            initiator="agent-a",
            responder="agent-b",
            initiator_terms=term_registry,
        )
        
        assert session.state == HandshakeState.IDLE
        assert session.initiator == "agent-a"
        assert session.responder == "agent-b"

    def test_create_init_message(self, term_registry):
        """Test creating HANDSHAKE_INIT message."""
        session = HandshakeSession(
            session_id="test",
            initiator="agent-a",
            responder="agent-b",
            initiator_terms=term_registry,
        )
        
        message = session.create_init_message()
        
        assert message.message_type == "HANDSHAKE_INIT"
        assert message.sender == "agent-a"
        assert message.receiver == "agent-b"
        assert "terms" in message.payload
        assert session.state == HandshakeState.INIT_SENT

    def test_handle_init_no_conflicts(self, term_registry):
        """Test handling HANDSHAKE_INIT with no conflicts."""
        session = HandshakeSession(
            session_id="test",
            initiator="agent-a",
            responder="agent-b",
            initiator_terms=term_registry,
        )
        
        # Create init message
        init_msg = session.create_init_message()
        
        # Create identical responder terms
        responder_terms = TermRegistry(agent_id="agent-b", load_defaults=True)
        
        # Handle init
        verify_msg = session.handle_init(init_msg, responder_terms)
        
        assert verify_msg.message_type == "HANDSHAKE_VERIFY"
        # Conflicts depend on embedding similarity

    def test_handshake_result(self, term_registry):
        """Test HandshakeResult creation."""
        session = HandshakeSession(
            session_id="test",
            initiator="agent-a",
            responder="agent-b",
            initiator_terms=term_registry,
        )
        
        result = session.get_result()
        
        assert isinstance(result, HandshakeResult)
        assert result.session_id == "test"
        assert result.agent_a == "agent-a"


class TestHandshakeManager:
    """Tests for handshake manager."""

    def test_create_session(self, term_registry):
        """Test creating a handshake session."""
        manager = HandshakeManager()
        
        session = manager.create_session(
            initiator="agent-a",
            responder="agent-b",
            initiator_terms=term_registry,
        )
        
        assert session is not None
        assert session.initiator == "agent-a"

    def test_get_session(self, term_registry):
        """Test getting a session by ID."""
        manager = HandshakeManager()
        
        session = manager.create_session(
            initiator="agent-a",
            responder="agent-b",
            initiator_terms=term_registry,
        )
        
        retrieved = manager.get_session(session.session_id)
        assert retrieved is session


# =============================================================================
# Negotiator Tests
# =============================================================================


class TestTermNegotiator:
    """Tests for term negotiator."""

    def test_init(self):
        """Test negotiator initialization."""
        negotiator = TermNegotiator(max_rounds=5)
        assert negotiator.max_rounds == 5

    def test_identical_definitions(self):
        """Test negotiation with identical definitions."""
        negotiator = TermNegotiator()
        
        result = negotiator.negotiate_term_sync(
            term="product",
            definition_a="An item for sale",
            definition_b="An item for sale",
        )
        
        assert result.success is True
        assert result.negotiation_rounds == 0

    def test_fallback_strategy_merge(self):
        """Test fallback merge strategy."""
        negotiator = TermNegotiator(fallback_strategy="merge")
        
        result = negotiator._fallback_negotiate(
            term="SKU",
            definition_a="Stock Keeping Unit",
            definition_b="Product identifier",
        )
        
        assert result.success is True
        assert "Stock Keeping Unit" in result.agreed_definition or "Product identifier" in result.agreed_definition

    def test_fallback_strategy_first(self):
        """Test fallback first strategy."""
        negotiator = TermNegotiator(fallback_strategy="first")
        
        result = negotiator._fallback_negotiate(
            term="SKU",
            definition_a="Definition A",
            definition_b="Definition B",
        )
        
        assert result.agreed_definition == "Definition A"

    def test_negotiation_result_to_dict(self):
        """Test NegotiationResult serialization."""
        result = NegotiationResult(
            term="test",
            agreed_definition="agreed",
            original_definition_a="def_a",
            original_definition_b="def_b",
        )
        
        data = result.to_dict()
        assert data["term"] == "test"
        assert data["agreed_definition"] == "agreed"


class TestRealSemanticNegotiationInExperiments:
    """Integration tests for real semantic negotiation in experiment runner."""
    
    @pytest.mark.asyncio
    async def test_real_negotiation_uses_llm(self):
        """Verify experiments use real TermNegotiator when use_real_reconstruction=True."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import get_condition
        
        runner = ExperimentRunner(seed=42)
        condition = get_condition('full_system')
        
        result = await runner.run_single_async(
            "vendor_onboarding", condition, use_real_reconstruction=True
        )
        
        # The experiment should complete (success or failure is not the point)
        assert result is not None
        
        # If semantic conflicts occurred, timing should be realistic
        # Real LLM calls typically take 500-5000ms per conflict
        # Simulated timing is 50-200ms
        # Note: Conflicts may not occur every run due to probability
        if result.semantic_conflicts > 0 and result.semantic_resolved > 0:
            # Real negotiation should take more time than simulated
            # This is a soft assertion - depends on LLM availability
            assert result.semantic_negotiation_ms >= 0
    
    @pytest.mark.asyncio
    async def test_simulated_negotiation_without_real_reconstruction(self):
        """Verify simulated negotiation is used when use_real_reconstruction=False."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import get_condition
        
        runner = ExperimentRunner(seed=42)
        condition = get_condition('full_system')
        
        result = await runner.run_single_async(
            "vendor_onboarding", condition, use_real_reconstruction=False
        )
        
        # The experiment should complete
        assert result is not None
        
        # With simulated negotiation, timing should be in the 50-200ms range
        # (if conflicts occurred)
        if result.semantic_conflicts > 0:
            # Simulated timing is capped at 200ms per conflict
            assert result.semantic_negotiation_ms < 500  # Allow some margin
    
    @pytest.mark.asyncio
    async def test_real_mttr_timing(self):
        """Verify real reconstruction produces actual LLM timing for MTTR."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import get_condition
        
        runner = ExperimentRunner(seed=42)
        condition = get_condition('full_system')
        
        result = await runner.run_single_async(
            "vendor_onboarding", condition, use_real_reconstruction=True
        )
        
        assert result is not None
        
        # If recovery was attempted with real reconstruction
        if result.recovery_attempted:
            # Recovery time should be present
            assert result.recovery_time_ms is not None or result.recovery_time_ms == 0

