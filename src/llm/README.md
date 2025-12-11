# LLM Provider Module

This module provides centralized LLM configuration and initialization for the entire PaaS system.

## Usage

### Basic Usage

```python
from src.llm import get_llm

# Get LLM with default settings from config
llm = get_llm()

# Get LLM with custom temperature
llm = get_llm(temperature=0.5)

# Get LLM with custom model
llm = get_llm(model="kimi-k2-thinking-turbo")
```

### Advanced Usage

```python
from src.llm import get_llm_with_params

# Full control over parameters
llm = get_llm_with_params(
    model="custom-model",
    temperature=0.3,
    max_tokens=1000
)
```

## Configuration

LLM provider is configured via environment variables in `.env`:

### Moonshot/Kimi API (Default)

```bash
LLM_PROVIDER=moonshot
LLM_API_KEY=your_moonshot_api_key
# Or use: KIMI_MOONSHOT_API_KEY=your_moonshot_api_key
LLM_BASE_URL=https://api.moonshot.ai/v1
LLM_MODEL=kimi-k2-turbo-preview
LLM_TEMPERATURE=0.7
```

### OpenAI API

```bash
LLM_PROVIDER=openai
LLM_API_KEY=your_openai_api_key
# Or use: OPENAI_API_KEY=your_openai_api_key
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7
```

## Supported Models

### Moonshot/Kimi Models
- `kimi-k2-turbo-preview` - Fast, high-speed (60-100 tokens/s)
- `kimi-k2-0905-preview` - Latest version, 256K context
- `kimi-k2-thinking` - Long-term thinking, multi-step reasoning
- `kimi-k2-thinking-turbo` - Fast thinking version

### OpenAI Models
- `gpt-4o-mini` - Fast and cost-effective
- `gpt-4o` - More capable
- `gpt-4-turbo` - High performance

## Benefits

- **Single source of truth**: All LLM configuration in one place
- **Easy provider switching**: Change one environment variable
- **Consistent usage**: All agents use the same LLM configuration
- **Future-proof**: Easy to add new providers

