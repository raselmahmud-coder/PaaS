"""Integration tests for Shopify API integration.

These tests verify the Shopify client, agent steps, workflow, and
experiment runner functionality. Tests marked with `real_api` require
actual Shopify credentials and will skip if not configured.
"""

import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.config import settings


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_shopify_client():
    """Create a mock Shopify client."""
    from src.integrations.shopify.client import ShopifyClient
    
    client = MagicMock(spec=ShopifyClient)
    client.create_product = AsyncMock(return_value={
        "id": 12345678,
        "title": "PAAS_TEST_Test Product",
        "variants": [{"id": 87654321, "price": "29.99"}],
    })
    client.get_product = AsyncMock(return_value={
        "id": 12345678,
        "title": "PAAS_TEST_Test Product",
    })
    client.update_product = AsyncMock(return_value={
        "id": 12345678,
        "title": "PAAS_TEST_Updated Product",
    })
    client.delete_product = AsyncMock(return_value=True)
    client.health_check = AsyncMock(return_value=True)
    client.list_products = AsyncMock(return_value=[])
    
    return client


@pytest.fixture
def sample_product_data():
    """Sample product data for testing."""
    return {
        "name": "Test Widget",
        "price": 29.99,
        "category": "Test",
        "description": "A test product for PaaS experiments",
        "sku": "TEST-001",
    }


@pytest.fixture
def sample_agent_state(sample_product_data):
    """Sample agent state for testing."""
    return {
        "task_id": "test-123",
        "agent_id": "shopify-agent",
        "thread_id": "thread-123",
        "current_step": 0,
        "status": "pending",
        "messages": [],
        "product_data": sample_product_data,
        "generated_listing": None,
        "error": None,
        "metadata": {},
    }


# =============================================================================
# Shopify Client Tests
# =============================================================================


class TestShopifyClient:
    """Tests for the ShopifyClient class."""
    
    def test_client_initialization(self):
        """Test client initializes with settings."""
        from src.integrations.shopify.client import ShopifyClient
        
        client = ShopifyClient(
            store_url="test.myshopify.com",
            access_token="test_token",
            api_version="2024-01",
        )
        
        assert client.store_url == "test.myshopify.com"
        assert client.access_token == "test_token"
        assert "test.myshopify.com" in client.base_url
        assert "2024-01" in client.base_url
    
    def test_client_headers(self):
        """Test client sets correct headers."""
        from src.integrations.shopify.client import ShopifyClient
        
        client = ShopifyClient(
            store_url="test.myshopify.com",
            access_token="secret_token",
        )
        
        assert client.headers["X-Shopify-Access-Token"] == "secret_token"
        assert "Content-Type" in client.headers
    
    def test_get_shopify_client_singleton(self):
        """Test get_shopify_client returns singleton."""
        from src.integrations.shopify.client import (
            get_shopify_client,
            _client_instance,
        )
        
        # Reset singleton
        import src.integrations.shopify.client as client_module
        client_module._client_instance = None
        
        client1 = get_shopify_client()
        client2 = get_shopify_client()
        
        assert client1 is client2


class TestRateLimiter:
    """Tests for the RateLimiter class."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_initial_requests(self):
        """Test rate limiter allows initial requests."""
        from src.integrations.shopify.utils import RateLimiter
        
        limiter = RateLimiter(requests_per_second=5)
        
        # Should allow 5 immediate requests
        for _ in range(5):
            await limiter.acquire()
        
        assert len(limiter.timestamps) == 5
    
    @pytest.mark.asyncio
    async def test_rate_limiter_reset(self):
        """Test rate limiter reset clears state."""
        from src.integrations.shopify.utils import RateLimiter
        
        limiter = RateLimiter(requests_per_second=2)
        await limiter.acquire()
        await limiter.acquire()
        
        assert len(limiter.timestamps) == 2
        
        limiter.reset()
        
        assert len(limiter.timestamps) == 0


# =============================================================================
# Shopify Agent Tests
# =============================================================================


class TestShopifyAgentSteps:
    """Tests for Shopify agent step functions."""
    
    @pytest.mark.asyncio
    async def test_validate_and_create_missing_fields(self, sample_agent_state):
        """Test validation fails with missing required fields."""
        from src.agents.shopify_product import validate_and_create_shopify
        
        # Remove required field
        sample_agent_state["product_data"] = {"category": "Test"}
        
        # Disable chaos for this test
        with patch("src.chaos.config.get_chaos_config") as mock_config:
            mock_config.return_value.enabled = False
            
            # Mock the Shopify client
            with patch("src.agents.shopify_product.get_shopify_client"):
                result = await validate_and_create_shopify(sample_agent_state)
        
        assert result["status"] == "failed"
        assert "Missing required fields" in result["error"]
    
    @pytest.mark.asyncio
    async def test_validate_and_create_success(
        self, sample_agent_state, mock_shopify_client
    ):
        """Test successful product creation."""
        from src.agents.shopify_product import validate_and_create_shopify
        
        with patch("src.chaos.config.get_chaos_config") as mock_config:
            mock_config.return_value.enabled = False
            
            with patch(
                "src.agents.shopify_product.get_shopify_client",
                return_value=mock_shopify_client,
            ):
                result = await validate_and_create_shopify(sample_agent_state)
        
        assert result["status"] == "in_progress"
        assert result["shopify_product_id"] == 12345678
        assert result["shopify_variant_id"] == 87654321
    
    @pytest.mark.asyncio
    async def test_cleanup_with_product_id(
        self, sample_agent_state, mock_shopify_client
    ):
        """Test cleanup deletes product."""
        from src.agents.shopify_product import cleanup_test_product
        
        sample_agent_state["shopify_product_id"] = 12345678
        
        with patch("src.chaos.config.get_chaos_config") as mock_config:
            mock_config.return_value.enabled = False
            
            with patch(
                "src.agents.shopify_product.get_shopify_client",
                return_value=mock_shopify_client,
            ):
                result = await cleanup_test_product(sample_agent_state)
        
        assert result["status"] == "completed"
        assert result["shopify_product_id"] is None
        mock_shopify_client.delete_product.assert_called_once_with(12345678)
    
    @pytest.mark.asyncio
    async def test_cleanup_without_product_id(self, sample_agent_state):
        """Test cleanup succeeds even without product ID."""
        from src.agents.shopify_product import cleanup_test_product
        
        sample_agent_state["shopify_product_id"] = None
        
        with patch("src.chaos.config.get_chaos_config") as mock_config:
            mock_config.return_value.enabled = False
            
            result = await cleanup_test_product(sample_agent_state)
        
        assert result["status"] == "completed"


class TestAsyncChaosDecorators:
    """Tests for async chaos decorators."""
    
    @pytest.mark.asyncio
    async def test_async_inject_crash_triggers(self):
        """Test crash injection triggers with probability 1.0."""
        from src.agents.shopify_product import async_inject_crash
        from src.chaos.exceptions import AgentCrashException
        from src.chaos.config import ChaosConfig
        
        @async_inject_crash(probability=1.0)
        async def test_func():
            return "success"
        
        mock_config = MagicMock(spec=ChaosConfig)
        mock_config.enabled = True
        
        with patch("src.agents.shopify_product.get_chaos_config", return_value=mock_config):
            with pytest.raises(AgentCrashException):
                await test_func()
    
    @pytest.mark.asyncio
    async def test_async_inject_crash_disabled(self):
        """Test crash injection is skipped when chaos disabled."""
        from src.agents.shopify_product import async_inject_crash
        from src.chaos.config import ChaosConfig
        
        @async_inject_crash(probability=1.0)
        async def test_func():
            return "success"
        
        mock_config = MagicMock(spec=ChaosConfig)
        mock_config.enabled = False
        
        with patch("src.agents.shopify_product.get_chaos_config", return_value=mock_config):
            result = await test_func()
            assert result == "success"


# =============================================================================
# Network Chaos Tests
# =============================================================================


class TestNetworkChaos:
    """Tests for network chaos utilities."""
    
    @pytest.mark.asyncio
    async def test_network_delay_injection(self):
        """Test network delay is injected."""
        from src.chaos.network import inject_network_delay
        from src.chaos.config import ChaosConfig
        import time
        
        mock_config = MagicMock(spec=ChaosConfig)
        mock_config.enabled = True
        
        with patch("src.chaos.network.get_chaos_config", return_value=mock_config):
            start = time.time()
            async with inject_network_delay(delay_ms=100, jitter_ms=0, probability=1.0):
                pass
            elapsed = (time.time() - start) * 1000
            
            assert elapsed >= 90  # Allow some variance
    
    @pytest.mark.asyncio
    async def test_network_conditioner(self):
        """Test NetworkConditioner applies conditions."""
        from src.chaos.network import NetworkConditioner
        
        conditioner = NetworkConditioner(
            latency_ms=50,
            jitter_ms=0,
            packet_loss_probability=0.0,
        )
        
        import time
        start = time.time()
        async with conditioner.condition():
            pass
        elapsed = (time.time() - start) * 1000
        
        assert elapsed >= 40  # Allow some variance
    
    @pytest.mark.asyncio
    async def test_network_conditioner_disabled(self):
        """Test disabled NetworkConditioner has no effect."""
        from src.chaos.network import NetworkConditioner
        
        conditioner = NetworkConditioner(latency_ms=1000)
        conditioner.disable()
        
        import time
        start = time.time()
        async with conditioner.condition():
            pass
        elapsed = (time.time() - start) * 1000
        
        assert elapsed < 100  # Should be nearly instant


# =============================================================================
# Workflow Tests
# =============================================================================


class TestShopifyWorkflow:
    """Tests for Shopify workflow."""
    
    def test_create_workflow(self):
        """Test workflow can be created."""
        from src.workflows.shopify_workflow import create_shopify_product_workflow
        
        workflow = create_shopify_product_workflow()
        assert workflow is not None
    
    def test_create_simple_workflow(self):
        """Test simple workflow can be created."""
        from src.workflows.shopify_workflow import create_simple_shopify_workflow
        
        workflow = create_simple_shopify_workflow()
        assert workflow is not None


# =============================================================================
# Experiment Condition Tests
# =============================================================================


class TestRealAPICondition:
    """Tests for RealAPICondition."""
    
    def test_real_api_condition_config(self):
        """Test RealAPICondition has correct config."""
        from src.experiments.conditions import RealAPICondition
        
        condition = RealAPICondition()
        
        assert condition.name == "real_api"
        assert condition.config.is_real_api is True
        assert condition.should_attempt_recovery() is True
        assert condition.should_use_semantic_protocol() is True
        assert condition.should_use_automata() is True
        assert condition.should_query_peers() is True
    
    def test_real_api_condition_in_registry(self):
        """Test RealAPICondition is in registry."""
        from src.experiments.conditions import get_condition, list_conditions
        
        assert "real_api" in list_conditions()
        
        condition = get_condition("real_api")
        assert condition.name == "real_api"


# =============================================================================
# Real API Tests (require Shopify credentials)
# =============================================================================


def has_shopify_credentials():
    """Check if Shopify credentials are configured."""
    return bool(settings.shopify_store_url and settings.shopify_access_token)


@pytest.mark.skipif(
    not has_shopify_credentials(),
    reason="Shopify credentials not configured"
)
class TestRealShopifyAPI:
    """Real API tests (require credentials)."""
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test real Shopify health check."""
        from src.integrations.shopify import get_shopify_client
        
        client = get_shopify_client()
        result = await client.health_check()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_create_and_delete_product(self):
        """Test real product creation and deletion."""
        from src.integrations.shopify import get_shopify_client
        from src.integrations.shopify.utils import generate_test_product_data
        
        client = get_shopify_client()
        
        # Create product
        product_data = generate_test_product_data(
            name="Integration Test",
            price=9.99,
            category="Test",
        )
        
        product = await client.create_product(product_data)
        
        assert product["id"] is not None
        assert "PAAS_TEST_" in product["title"]
        
        # Delete product
        deleted = await client.delete_product(product["id"])
        assert deleted is True


# =============================================================================
# Utility Tests
# =============================================================================


class TestShopifyUtils:
    """Tests for Shopify utility functions."""
    
    def test_generate_test_product_data(self):
        """Test generate_test_product_data creates valid data."""
        from src.integrations.shopify.utils import generate_test_product_data
        
        data = generate_test_product_data(
            name="Test Product",
            price=29.99,
            category="Test",
        )
        
        assert data["title"].startswith("PAAS_TEST_")
        assert data["product_type"] == "Test"
        assert data["variants"][0]["price"] == "29.99"
        assert "paas-test" in data["tags"]
    
    @pytest.mark.asyncio
    async def test_cleanup_test_products_dry_run(self, mock_shopify_client):
        """Test cleanup_test_products in dry run mode."""
        from src.integrations.shopify.utils import cleanup_test_products
        
        # Mock products to return
        mock_shopify_client.list_products = AsyncMock(return_value=[
            {"id": 1, "title": "PAAS_TEST_Product1"},
            {"id": 2, "title": "PAAS_TEST_Product2"},
        ])
        
        deleted = await cleanup_test_products(mock_shopify_client, dry_run=True)
        
        assert deleted == 2
        # Should not actually delete in dry run
        mock_shopify_client.delete_product.assert_not_called()

