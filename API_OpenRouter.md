# OpenRouter API Documentation

---

## Using the OpenRouter API Directly

### Python Example

```python
import requests
import json
response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": "Bearer <OPENROUTER_API_KEY>",
        # Optional. Site URL for rankings on openrouter.ai.
        "HTTP-Referer": "<YOUR_SITE_URL>",
        # Optional. Site title for rankings on openrouter.ai.
        "X-Title": "<YOUR_SITE_NAME>"
    },
    data=json.dumps({
        "model": "openai/gpt-4o",  # Optional
        "messages": [
            {"role": "user", "content": "What is the meaning of life?"}
        ]
    })
)
```

---

## Models

### NVIDIA: Llama 3.1 Nemotron Ultra 253B v1 (free)

**Model ID:** `nvidia/llama-3.1-nemotron-ultra-253b-v1:free`

- **Created:** Apr 8, 2025
- **Context:** 131,072 tokens
- **Input Cost:** $0/M tokens
- **Output Cost:** $0/M tokens

Llama-3.1-Nemotron-Ultra-253B-v1 is a large language model (LLM) optimized for advanced reasoning, human-interactive chat, retrieval-augmented generation (RAG), and tool-calling tasks. Derived from Meta’s Llama-3.1-405B-Instruct, it has been significantly customized using Neural Architecture Search (NAS), resulting in enhanced efficiency, reduced memory usage, and improved inference latency. The model supports a context length of up to 128K tokens and can operate efficiently on an 8x NVIDIA H100 node.

**Note:** You must include detailed thinking in the system prompt to enable reasoning. Please see Usage Recommendations for more.

---

### Providers for Llama 3.1 Nemotron Ultra 253B v1 (free)

OpenRouter routes requests to the best providers that are able to handle your prompt size and parameters, with fallbacks to maximize uptime.

- **Provider**: Chutes
- **Precision**: bf16
- **Context**: 131K
- **Max Output**: 131K
- **Input Cost**: $0
- **Output Cost**: $0
- **Latency**: 1.88s
- **Throughput**: 36.77t/s
- **Uptime**: High

---

### meta-llama/llama-4-maverick:free

- **Chat**
- **Compare**
- **Created**: Apr 5, 2025
- **Context**: 256,000 tokens
- **Input Cost**: $0/M tokens
- **Output Cost**: $0/M tokens

---

---

# OpenRouter API Documentation

---

## Llama 4 Maverick 17B Instruct (128E)

Llama 4 Maverick 17B Instruct (128E) is a high-capacity multimodal language model from Meta, built on a mixture-of-experts (MoE) architecture with 128 experts and 17 billion active parameters per forward pass (400B total).

- **Multimodal**: Supports multilingual text and image input, produces multilingual text and code output (12 languages)
- **Optimized for**: Vision-language tasks, instruction-following, image reasoning, and general-purpose multimodal interaction
- **Context Window**: 1 million tokens
- **Training Data**: ~22 trillion tokens, knowledge cutoff August 2024
- **License**: Llama 4 Community License (April 5, 2025)

---

## Providers and API Details

| Model Name | Provider | Context | Max Output | Input Cost | Output Cost | Latency | Throughput | Uptime |
|------------|----------|---------|------------|------------|-------------|---------|------------|--------|
| Llama 4 Maverick | OpenRouter (Chutes/fp8) | 256K | 256K | $0 | $0 | 1.20s | 68.53t/s | High |
| MAI DS R1        | Microsoft (fp8)         | 164K | 164K | $0 | $0 | 12.32s| 109.4t/s | High |

---

## Llama 4 Maverick (free)

OpenRouter routes requests to the best providers that can handle your prompt size and parameters, with fallbacks for uptime.

- **Provider**: Chutes/fp8
- **Context**: 256K
- **Max Output**: 256K
- **Input/Output Cost**: $0
- **Latency**: 1.20s
- **Throughput**: 68.53t/s
- **Uptime**: High

---

## Microsoft: MAI DS R1 (free)

- **Model**: microsoft/mai-ds-r1:free
- **Context**: 164K
- **Max Output**: 164K
- **Input/Output Cost**: $0/M tokens
- **Latency**: 12.32s
- **Throughput**: 109.4t/s
- **Uptime**: High

MAI-DS-R1 is a post-trained variant of DeepSeek-R1 developed by Microsoft AI to improve responsiveness and safety. It integrates 110k examples from the Tulu-3 SFT dataset and 350k curated multilingual safety-alignment samples.

- **Improved**: Harm mitigation, general reasoning, coding, problem-solving
- **Not for**: Legal, medical, or autonomous systems
- **Architecture**: Transformer MoE

---

## API Usage

```python
# Example: Using OpenRouter API with Llama 4 Maverick
import requests

url = "https://openrouter.ai/api/v1/chat"
headers = {"Authorization": "Bearer <OPENROUTER_API_KEY>"}
payload = {
    "model": "meta-llama/llama-4-maverick:free",
    "messages": [{"role": "user", "content": "Solve: x^2 + 2x + 1 = 0"}],
    "max_tokens": 256
}
response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

---
164K
Input
$0
Output
$0
Latency
12,32s
Throughput
109,4t/s
Uptime

