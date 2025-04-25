# API_OpenRouter.md


Using the OpenRouter API directly

Python

TypeScript

Shell

import requests
import json
response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer <OPENROUTER_API_KEY>",
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  data=json.dumps({
    "model": "openai/gpt-4o", # Optional
    "messages": [
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ]
  })
)

MODELS:


NVIDIA: Llama 3.1 Nemotron Ultra 253B v1 (free)
nvidia/llama-3.1-nemotron-ultra-253b-v1:free

Chat
Compare
Created Apr 8, 2025
131,072 context
$0/M input tokens
$0/M output tokens
Llama-3.1-Nemotron-Ultra-253B-v1 is a large language model (LLM) optimized for advanced reasoning, human-interactive chat, retrieval-augmented generation (RAG), and tool-calling tasks. Derived from Meta’s Llama-3.1-405B-Instruct, it has been significantly customized using Neural Architecture Search (NAS), resulting in enhanced efficiency, reduced memory usage, and improved inference latency. The model supports a context length of up to 128K tokens and can operate efficiently on an 8x NVIDIA H100 node.

Note: you must include detailed thinking on in the system prompt to enable reasoning. Please see Usage Recommendations for more.


Model weights
Overview
Providers
Apps
Activity
Uptime
API
Providers for Llama 3.1 Nemotron Ultra 253B v1 (free)

OpenRouter routes requests to the best providers that are able to handle your prompt size and parameters, with fallbacks to maximize uptime. 
Chutes
bf16
Context
131K
Max Output
131K
Input
$0
Output
$0
Latency
1,88s
Throughput
36,77t/s
Uptime



meta-llama/llama-4-maverick:free

Chat
Compare
Created Apr 5, 2025
256,000 context
$0/M input tokens
$0/M output tokens
Llama 4 Maverick 17B Instruct (128E) is a high-capacity multimodal language model from Meta, built on a mixture-of-experts (MoE) architecture with 128 experts and 17 billion active parameters per forward pass (400B total). It supports multilingual text and image input, and produces multilingual text and code output across 12 supported languages. Optimized for vision-language tasks, Maverick is instruction-tuned for assistant-like behavior, image reasoning, and general-purpose multimodal interaction.

Maverick features early fusion for native multimodality and a 1 million token context window. It was trained on a curated mixture of public, licensed, and Meta-platform data, covering ~22 trillion tokens, with a knowledge cutoff in August 2024. Released on April 5, 2025 under the Llama 4 Community License, Maverick is suited for research and commercial applications requiring advanced multimodal understanding and high model throughput.



Free
Model weights
Overview
Providers
Apps
Activity
Uptime
API
Providers for Llama 4 Maverick (free)

OpenRouter routes requests to the best providers that are able to handle your prompt size and parameters, with fallbacks to maximize uptime. 
Chutes
fp8
Context
256K
Max Output
256K
Input
$0
Output
$0
Latency
1,20s
Throughput
68,53t/s
Uptime



Microsoft: MAI DS R1 (free)
microsoft/mai-ds-r1:free

Chat
Compare
Created Apr 21, 2025
163,840 context
$0/M input tokens
$0/M output tokens
MAI-DS-R1 is a post-trained variant of DeepSeek-R1 developed by the Microsoft AI team to improve the model’s responsiveness on previously blocked topics while enhancing its safety profile. Built on top of DeepSeek-R1’s reasoning foundation, it integrates 110k examples from the Tulu-3 SFT dataset and 350k internally curated multilingual safety-alignment samples. The model retains strong reasoning, coding, and problem-solving capabilities, while unblocking a wide range of prompts previously restricted in R1.

MAI-DS-R1 demonstrates improved performance on harm mitigation benchmarks and maintains competitive results across general reasoning tasks. It surpasses R1-1776 in satisfaction metrics for blocked queries and reduces leakage in harmful content categories. The model is based on a transformer MoE architecture and is suitable for general-purpose use cases, excluding high-stakes domains such as legal, medical, or autonomous systems.


Model weights
Overview
Providers
Apps
Activity
Uptime
API
Providers for MAI DS R1 (free)
OpenRouter routes requests to the best providers that are able to handle your prompt size and parameters, with fallbacks to maximize uptime. 
Chutes
fp8
Context
164K
Max Output
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

