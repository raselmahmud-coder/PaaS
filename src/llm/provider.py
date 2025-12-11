"""LLM provider factory for creating configured ChatOpenAI instances."""

from typing import Optional
from langchain_openai import ChatOpenAI
from src.config import settings


def get_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    **kwargs
) -> ChatOpenAI:
    """
    Get a configured ChatOpenAI instance based on settings.
    
    Args:
        model: Override default model name
        temperature: Override default temperature
        **kwargs: Additional parameters to pass to ChatOpenAI
    
    Returns:
        Configured ChatOpenAI instance
    """
    # Use provided values or fall back to settings
    llm_model = model or settings.llm_model
    llm_temperature = temperature if temperature is not None else settings.llm_temperature
    llm_api_key = settings.llm_api_key
    llm_base_url = settings.llm_base_url
    
    # Build ChatOpenAI parameters
    llm_params = {
        "model": llm_model,
        "temperature": llm_temperature,
        "api_key": llm_api_key,
    }
    
    # Add base_url only if provider is not OpenAI (OpenAI doesn't need base_url)
    if settings.llm_provider != "openai" and llm_base_url:
        llm_params["base_url"] = llm_base_url
    
    # Add any additional kwargs
    llm_params.update(kwargs)
    
    return ChatOpenAI(**llm_params)


def get_llm_with_params(**kwargs) -> ChatOpenAI:
    """
    Get a ChatOpenAI instance with custom parameters.
    
    This function allows full control over LLM parameters while still
    using the configured provider settings as defaults.
    
    Args:
        **kwargs: Parameters to pass directly to ChatOpenAI
    
    Returns:
        Configured ChatOpenAI instance
    """
    # Start with default settings
    default_params = {
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "api_key": settings.llm_api_key,
    }
    
    # Add base_url if not OpenAI
    if settings.llm_provider != "openai" and settings.llm_base_url:
        default_params["base_url"] = settings.llm_base_url
    
    # Override with provided kwargs
    default_params.update(kwargs)
    
    return ChatOpenAI(**default_params)

