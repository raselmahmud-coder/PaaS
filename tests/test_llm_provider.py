"""Tests for LLM provider factory functions."""

import pytest
from unittest.mock import patch, MagicMock
from langchain_openai import ChatOpenAI

from src.llm.provider import get_llm, get_llm_with_params
from src.config import settings


class TestLLMProvider:
    """Test LLM provider factory functions."""
    
    @patch('src.llm.provider.ChatOpenAI')
    def test_get_llm_uses_default_settings(self, mock_chat_openai):
        """Test that get_llm uses default settings when no overrides provided."""
        get_llm()
        
        mock_chat_openai.assert_called_once()
        call_kwargs = mock_chat_openai.call_args[1]
        
        assert call_kwargs["model"] == settings.llm_model
        assert call_kwargs["temperature"] == settings.llm_temperature
        assert call_kwargs["api_key"] == settings.llm_api_key
        
        # base_url should be included if provider is not OpenAI
        if settings.llm_provider != "openai" and settings.llm_base_url:
            assert "base_url" in call_kwargs
            assert call_kwargs["base_url"] == settings.llm_base_url
    
    @patch('src.llm.provider.ChatOpenAI')
    def test_get_llm_with_model_override(self, mock_chat_openai):
        """Test that get_llm accepts model override."""
        custom_model = "gpt-4-turbo"
        get_llm(model=custom_model)
        
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs["model"] == custom_model
        assert call_kwargs["temperature"] == settings.llm_temperature  # Still uses default
    
    @patch('src.llm.provider.ChatOpenAI')
    def test_get_llm_with_temperature_override(self, mock_chat_openai):
        """Test that get_llm accepts temperature override."""
        custom_temp = 0.9
        get_llm(temperature=custom_temp)
        
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs["temperature"] == custom_temp
        assert call_kwargs["model"] == settings.llm_model  # Still uses default
    
    @patch('src.llm.provider.ChatOpenAI')
    def test_get_llm_with_additional_kwargs(self, mock_chat_openai):
        """Test that get_llm passes through additional kwargs."""
        get_llm(max_tokens=100, top_p=0.95)
        
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["top_p"] == 0.95
    
    @patch('src.llm.provider.ChatOpenAI')
    def test_get_llm_with_params_uses_defaults(self, mock_chat_openai):
        """Test that get_llm_with_params uses default settings."""
        get_llm_with_params()
        
        mock_chat_openai.assert_called_once()
        call_kwargs = mock_chat_openai.call_args[1]
        
        assert call_kwargs["model"] == settings.llm_model
        assert call_kwargs["temperature"] == settings.llm_temperature
        assert call_kwargs["api_key"] == settings.llm_api_key
    
    @patch('src.llm.provider.ChatOpenAI')
    def test_get_llm_with_params_allows_full_override(self, mock_chat_openai):
        """Test that get_llm_with_params allows full parameter override."""
        custom_params = {
            "model": "custom-model",
            "temperature": 0.5,
            "max_tokens": 200,
        }
        get_llm_with_params(**custom_params)
        
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs["model"] == "custom-model"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 200
    
    @patch('src.llm.provider.ChatOpenAI')
    def test_get_llm_base_url_for_non_openai(self, mock_chat_openai):
        """Test that base_url is added for non-OpenAI providers."""
        with patch.object(settings, 'llm_provider', 'anthropic'):
            with patch.object(settings, 'llm_base_url', 'https://api.anthropic.com'):
                get_llm()
                
                call_kwargs = mock_chat_openai.call_args[1]
                assert "base_url" in call_kwargs
                assert call_kwargs["base_url"] == "https://api.anthropic.com"
    
    @patch('src.llm.provider.ChatOpenAI')
    def test_get_llm_no_base_url_for_openai(self, mock_chat_openai):
        """Test that base_url is not added for OpenAI provider."""
        with patch.object(settings, 'llm_provider', 'openai'):
            get_llm()
            
            call_kwargs = mock_chat_openai.call_args[1]
            # base_url should not be in kwargs for OpenAI
            assert "base_url" not in call_kwargs or call_kwargs.get("base_url") is None
