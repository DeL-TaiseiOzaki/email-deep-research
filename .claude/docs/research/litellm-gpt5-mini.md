# litellm + gpt-5-mini Research Summary

**Date**: 2026-02-26
**Topic**: Using OpenAI gpt-5-mini with litellm

## Key Facts

### Model Classification
- gpt-5-mini **IS a reasoning model** (same family as gpt-5, gpt-5-nano)
- 400K context window, 128K max output tokens
- Faster, cost-efficient version of gpt-5
- Supports reasoning_effort parameter

### Model String Format in litellm

| API | Model String |
|-----|-------------|
| Chat Completions | `"gpt-5-mini"` |
| Responses API (recommended) | `"openai/responses/gpt-5-mini"` |

litellm recommends using the Responses API for latest OpenAI models (prefix `openai/responses/`).

### Unsupported Parameters (CRITICAL)

These parameters are **NOT supported** by gpt-5-mini and will cause 400 errors:

| Parameter | Status |
|-----------|--------|
| `temperature` | NOT supported (only default value 1 accepted) |
| `top_p` | NOT supported |
| `presence_penalty` | NOT supported |
| `frequency_penalty` | NOT supported |
| `logprobs` | NOT supported |

### Supported Parameters

| Parameter | Details |
|-----------|---------|
| `reasoning_effort` | `"minimal"`, `"low"`, `"medium"` (default), `"high"` |
| `max_completion_tokens` | Use instead of `max_tokens` for Chat Completions API |
| `max_tokens` | litellm auto-translates to `max_completion_tokens` (since PR #13390) |
| `reasoning.summary` | `"auto"`, `"concise"`, `"detailed"` (Responses API) |

### max_tokens Handling

- OpenAI's gpt-5 series requires `max_completion_tokens` (not `max_tokens`) at the API level
- litellm **automatically translates** `max_tokens` -> `max_completion_tokens` for gpt-5 models (fixed in PR #13390)
- You can safely pass `max_tokens` to litellm; it handles the translation
- For the Responses API, OpenAI uses `max_output_tokens`

### drop_params for Safety

To prevent errors from unsupported parameters:

```python
# Global setting
import litellm
litellm.drop_params = True

# Per-request setting
response = await litellm.acompletion(
    model="gpt-5-mini",
    messages=messages,
    drop_params=True,
)
```

### Check Reasoning Support Programmatically

```python
import litellm
is_reasoning = litellm.supports_reasoning(model="gpt-5-mini")  # Returns True
```

## Example Code

### Basic acompletion (Chat Completions API)

```python
import litellm

response = await litellm.acompletion(
    model="gpt-5-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing."},
    ],
    max_tokens=4096,           # litellm auto-translates to max_completion_tokens
    reasoning_effort="medium", # "minimal", "low", "medium", "high"
    drop_params=True,          # Safety: drops any unsupported params
    # DO NOT pass: temperature, top_p, presence_penalty, frequency_penalty
)

print(response.choices[0].message.content)
```

### Using Responses API (Recommended by litellm)

```python
import litellm

response = await litellm.acompletion(
    model="openai/responses/gpt-5-mini",
    messages=[
        {"role": "user", "content": "Explain quantum computing."},
    ],
    max_tokens=4096,
    reasoning_effort="medium",
)

# Access reasoning content if available
message = response.choices[0].message
if hasattr(message, "reasoning_content") and message.reasoning_content:
    print("Reasoning:", message.reasoning_content)
print("Answer:", message.content)
```

## Gotchas and Known Issues

1. **Temperature error**: Passing `temperature` (even `temperature=1.0`) may cause a 400 error. Omit it entirely.
2. **max_tokens translation**: Older litellm versions (<= ~1.56) don't auto-translate; upgrade or use `max_completion_tokens` directly.
3. **reasoning_effort on Chat Completions**: Works, but Responses API gives richer reasoning output.
4. **reasoning_effort="none"**: Supported on gpt-5.1+ but NOT on gpt-5-mini (use "minimal" instead).
5. **Caching**: litellm's standard caching mechanisms work with gpt-5-mini (no special handling needed).
6. **Cost**: gpt-5-mini is cheaper than gpt-5 but still a reasoning model; reasoning tokens count toward output cost.

## Sources

- https://docs.litellm.ai/docs/providers/openai
- https://docs.litellm.ai/docs/reasoning_content
- https://docs.litellm.ai/docs/completion/input
- https://developers.openai.com/api/docs/models/gpt-5-mini
- https://developers.openai.com/api/docs/guides/reasoning/
- https://github.com/BerriAI/litellm/issues/13381
- https://community.openai.com/t/gpt-5-models-temperature/1337957
