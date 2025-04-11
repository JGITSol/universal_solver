<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# generate docs based on spec sheets - I need data on how to prompt the models properly using ollama, as well as license info, short description - for list of models: ollama run cogito:3b

ollama run llama3.2
ollama run gemma3
ollama run exaone-deep:2.4b

To effectively use the listed models with Ollama, here's a structured guide covering prompting techniques, licensing details, and brief descriptions for each model:

---

## **Cogito:3b**[^12]

**Description**: A 3.61B-parameter hybrid reasoning model optimized for coding, STEM, multilingual tasks, and tool calling. Outperforms comparable models like LLaMA and DeepSeek in benchmarks.
**License**: Llama 3.2 Community License (commercial use permitted up to 700M monthly users; restricts competition and trademark use)[^6][^12].
**Prompting**:

- Use the system prompt `Enable deep thinking subroutine.` to activate extended reasoning[^12].
- Supports 128k context and 30+ languages.
- Example:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "cogito:3b",
  "messages": [
    {"role": "system", "content": "Enable deep thinking subroutine."},
    {"role": "user", "content": "How many Rs are in 'Strawberry'?"}
  ]
}'
```


---

## **Llama3.2**[^6][^9]

**Description**: Meta's lightweight 3B-parameter model for edge/mobile use, featuring text-only processing and efficient on-device performance.
**License**: Llama 3 Community License (commercial use allowed with attribution; restricts scaling beyond 700M users and competitive model training)[^6].
**Prompting**:

- Follow standard Llama 3 instruction templates (e.g., `[INST]...[/INST]`).
- Optimized for tasks like summarization and text rewriting[^9].
- Example:

```bash
ollama run llama3.2 "Summarize this article in three sentences: [text]"
```


---

## **Gemma3**[^3][^7][^10]

**Description**: Google's 3B-parameter multimodal model supporting text and images, 128k context, and 140+ languages.
**License**: Custom restrictive license (prohibits large-scale commercial use; allows Google to remotely restrict usage)[^7][^10].
**Prompting**:

- For multimodal input, structure prompts with text and image references.
- Example (text-only):

```bash
ollama run gemma3:3b "Explain quantum computing in simple terms."
```


---

## **EXAONE-Deep:2.4b**[^4][^11][^13]

**Description**: LG AI Research's 2.4B-parameter model specialized in math, coding, and scientific reasoning.
**License**: Research-only availability (exact terms unspecified; check Hugging Face for updates)[^13].
**Prompting**:

- Provide structured problem statements (e.g., `Solve 2x + 3 = 7`).
- Example:

```bash
ollama run exaone-deep:2.4b "Write a Python function to calculate Fibonacci sequence up to n terms."
```


---

## Key Considerations

- **Licensing**: Verify commercial use limits for Meta/Google models[^6][^7].
- **Hardware**: Ensure sufficient RAM/GPU for larger models (e.g., 16GB+ for Cogito 3B)[^1][^4].
- **API Testing**: Use tools like Apidog to validate model-generated endpoints[^1][^4].

For updates, refer to official Ollama libraries and Hugging Face repositories[^12][^13].

<div>⁂</div>

[^1]: https://apidog.com/blog/run-deep-cogito-locally-with-ollama/

[^2]: https://www.reddit.com/r/ollama/comments/1czw7mj/how_to_set_system_prompt_in_ollama/

[^3]: https://www.linkedin.com/pulse/practical-guide-running-gemma-3-locally-ollama-hiram-reis-neto-1mt8f

[^4]: https://apidog.com/blog/run-exaone-deep-with-ollama/

[^5]: https://www.reddit.com/r/LocalLLaMA/comments/1jum5s1/cogito_releases_strongest_llms_of_sizes_3b_8b_14b/

[^6]: https://dev.to/llm_explorer/llama3-license-explained-2915

[^7]: https://techcrunch.com/2025/03/14/open-ai-model-licenses-often-carry-concerning-restrictions/

[^8]: https://venturebeat.com/ai/new-open-source-ai-company-deep-cogito-releases-first-models-and-theyre-already-topping-the-charts/

[^9]: https://www.datacamp.com/blog/llama-3-2

[^10]: https://huggingface.co/blog/gemma3

[^11]: https://www.linkedin.com/posts/kadir-nar_exaone-deep-is-lg-ai-researchs-cutting-edge-activity-7309600650332401664-z_lm

[^12]: https://ollama.com/library/cogito:3b

[^13]: https://arxiv.org/html/2503.12524v2

[^14]: https://github.com/ivanfioravanti/prompt-eng-ollama-interactive-tutorial

[^15]: https://ollama.com/library/llama3.2

[^16]: https://ai.google.dev/gemma/docs/integrations/ollama

[^17]: https://ollama.com/MHKetbi/LGAI_EXAONE-Deep-32B

[^18]: https://apidog.com/blog/run-deepcoder-locally-with-ollama/

[^19]: https://github.com/ollama/ollama/blob/main/README.md

[^20]: https://apidog.com/blog/run-gemma-3-locally-with-ollama/

[^21]: https://modelscope.cn/models/LGAI-EXAONE/EXAONE-Deep-2.4B

[^22]: https://ollama.com/library

[^23]: https://ollama.com/library/llama3:instruct

[^24]: https://docs.unsloth.ai/basics/tutorial-how-to-run-and-fine-tune-gemma-3

[^25]: https://ollama.com/library/exaone-deep:2.4b

[^26]: https://github.com/meta-llama/llama3/blob/main/LICENSE

[^27]: https://ollama.com/library/gemma3/blobs/dd084c7d92a3

[^28]: https://github.com/LG-AI-EXAONE/EXAONE-Deep

[^29]: https://www.together.ai/models/cogito-v1-preview-llama-3b

[^30]: https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE

[^31]: https://ai.google.dev/gemma/docs/core

[^32]: https://aimresearch.co/market-industry/from-stealth-to-superintelligence-meet-deep-cogito

[^33]: https://www.reddit.com/r/LocalLLaMA/comments/1fqhjs9/is_llama_32_banned_to_use_in_eu/

[^34]: https://blog.google/technology/developers/gemma-3/

[^35]: https://builds.modular.com/models/EXAONE-Deep/2.4B

[^36]: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision

[^37]: https://huggingface.co/google/gemma-3-27b-it

[^38]: https://wtai.cc/item/exaone-deep

[^39]: https://modelscope.cn/models/deepcogito/cogito-v1-preview-llama-3B

[^40]: https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/

[^41]: https://developers.googleblog.com/en/introducing-gemma3/

[^42]: https://ai.google.dev/gemma

[^43]: https://www.youtube.com/watch?v=1nVqwqszHCY

[^44]: https://dev.to/nodeshiftcloud/a-step-by-step-guide-to-install-gemma-3-locally-with-ollama-or-transformers-12g6

[^45]: https://github.com/LG-AI-EXAONE/EXAONE-Deep/blob/main/README.md

[^46]: https://llm.extractum.io/model/LGAI-EXAONE%2FEXAONE-Deep-2.4B,7vvgtcr8CZaLsftDr6xztT

[^47]: https://workmind.ai/new-ai-startup-deep-cogito-open-source-ai-models/

[^48]: https://www.llama.com/faq/

[^49]: https://gradientflow.com/gemma-3-what-you-need-to-know/

[^50]: https://huggingface.co/meta-llama/Llama-3.2-3B

[^51]: https://habr.com/en/articles/896290/

