"""Tests for Marketing Agent."""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from src.agents.marketing import generate_marketing_copy, schedule_campaign
from src.agents.base import MarketingState


class TestMarketingAgent:
    """Test Marketing Agent functions."""
    
    @patch('src.agents.marketing.llm')
    def test_generate_marketing_copy_constructs_prompt(self, mock_llm):
        """Test that generate_marketing_copy constructs correct prompt."""
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "Amazing product! Buy now!"
        mock_llm.invoke.return_value = mock_response
        
        state: MarketingState = {
            "product_description": "A revolutionary new gadget",
            "product_id": "prod-123",
            "marketing_channel": "email",
            "messages": [],
        }
        
        result = generate_marketing_copy(state)
        
        # Verify LLM was called
        assert mock_llm.invoke.called
        call_args = mock_llm.invoke.call_args[0][0]
        assert isinstance(call_args[0], HumanMessage)
        
        # Verify prompt contains key information
        prompt_content = call_args[0].content
        assert "revolutionary new gadget" in prompt_content
        assert "prod-123" in prompt_content
        assert "email" in prompt_content
        assert "marketing copy" in prompt_content.lower()
        
        # Verify result
        assert result["status"] == "in_progress"
        assert result["current_step"] == 1
        assert result["marketing_copy"] == "Amazing product! Buy now!"
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
    
    @patch('src.agents.marketing.llm')
    def test_generate_marketing_copy_handles_missing_fields(self, mock_llm):
        """Test that generate_marketing_copy handles missing optional fields."""
        mock_response = MagicMock()
        mock_response.content = "Generic marketing copy"
        mock_llm.invoke.return_value = mock_response
        
        state: MarketingState = {
            "product_description": "Product",
            "messages": [],
        }
        
        result = generate_marketing_copy(state)
        
        # Should use defaults for missing fields when reading
        assert result["marketing_copy"] == "Generic marketing copy"
        # Note: defaults are used when reading, not stored in result
        assert result.get("product_id") is None or result.get("product_id") == "unknown"
        assert result.get("marketing_channel") is None or result.get("marketing_channel") == "email"
    
    @patch('src.agents.marketing.llm')
    def test_generate_marketing_copy_appends_to_messages(self, mock_llm):
        """Test that generate_marketing_copy appends to existing messages."""
        mock_response = MagicMock()
        mock_response.content = "New copy"
        mock_llm.invoke.return_value = mock_response
        
        existing_message = AIMessage(content="Previous message")
        state: MarketingState = {
            "product_description": "Product",
            "messages": [existing_message],
        }
        
        result = generate_marketing_copy(state)
        
        # Should have 2 messages now
        assert len(result["messages"]) == 2
        assert result["messages"][0] == existing_message
        assert isinstance(result["messages"][1], AIMessage)
        assert "New copy" in result["messages"][1].content
    
    def test_schedule_campaign_creates_schedule_message(self):
        """Test that schedule_campaign creates appropriate schedule message."""
        state: MarketingState = {
            "marketing_copy": "Buy now! Limited time offer!",
            "marketing_channel": "social_media",
            "product_id": "prod-456",
            "messages": [],
        }
        
        result = schedule_campaign(state)
        
        # Verify result
        assert result["status"] == "completed"
        assert result["current_step"] == 2
        assert result["campaign_scheduled"] is True
        assert len(result["messages"]) == 1
        
        # Verify message content
        message = result["messages"][0]
        assert isinstance(message, AIMessage)
        assert "social_media" in message.content
        assert "prod-456" in message.content
        assert "Buy now!" in message.content
        assert "24 hours" in message.content
    
    def test_schedule_campaign_handles_missing_fields(self):
        """Test that schedule_campaign handles missing optional fields."""
        state: MarketingState = {
            "marketing_copy": "Copy",
            "messages": [],
        }
        
        result = schedule_campaign(state)
        
        # Should use defaults when reading
        assert result["campaign_scheduled"] is True
        # Note: defaults are used when reading, not stored in result
        assert result.get("marketing_channel") is None or result.get("marketing_channel") == "email"
        assert result.get("product_id") is None or result.get("product_id") == "unknown"
    
    def test_schedule_campaign_appends_to_messages(self):
        """Test that schedule_campaign appends to existing messages."""
        existing_message = AIMessage(content="Previous")
        state: MarketingState = {
            "marketing_copy": "Copy",
            "messages": [existing_message],
        }
        
        result = schedule_campaign(state)
        
        # Should have 2 messages
        assert len(result["messages"]) == 2
        assert result["messages"][0] == existing_message
        assert isinstance(result["messages"][1], AIMessage)
    
    def test_schedule_campaign_truncates_long_copy(self):
        """Test that schedule_campaign truncates very long marketing copy."""
        long_copy = "A" * 500  # Very long copy
        state: MarketingState = {
            "marketing_copy": long_copy,
            "messages": [],
        }
        
        result = schedule_campaign(state)
        
        # Preview should be truncated
        message = result["messages"][0]
        assert len(message.content) < len(long_copy) + 200  # Some overhead for other text
        assert "..." in message.content

