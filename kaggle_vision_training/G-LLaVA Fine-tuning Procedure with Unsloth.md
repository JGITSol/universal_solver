<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# G-LLaVA Fine-tuning Procedure with Unsloth

This guide outlines a structured approach to fine-tune G-LLaVA using Unsloth for geometric problem-solving.

## Environment Setup

```python
# Install Unsloth
pip install --upgrade --no-cache unsloth loth_zoo

# Clone G-LLaVA repository for dataset access
git clone https://github.com/pipilurj/G-LLaVA
cd G-LLaVA
```


## Model Selection

For geometric problem-solving, use LLaVA as the base model, which G-LLaVA is built upon:

```python
from unsloth import FastVisionModel
import torch

# Start with LLaVA since G-LLaVA uses this architecture
model_id = "llava-hf/llava-1.5-7b-hf"  # Or 13B version depending on your GPU
max_seq_length = 2048  # Set appropriate context length

# Initialize model with Unsloth's optimized loader
model, tokenizer = FastVisionModel.from_pretrained(
    model_name = model_id,
    max_seq_length = max_seq_length,
    load_in_4bit = True,  # Enable 4-bit quantization for memory efficiency
)
```


## Fine-tuning Configuration

Configure the model with G-LLaVA-specific parameters:

```python
# Configure PEFT for G-LLaVA
# Higher rank and alpha values based on LLaVA recommendations
model = FastVisionModel.get_peft_model(
    model,
    # Enable fine-tuning for both vision and language components
    finetune_vision_layers = True,     # Critical for geometric understanding
    finetune_language_layers = True,   # Important for mathematical reasoning
    finetune_attention_modules = True, # Key for cross-modal alignment
    finetune_mlp_modules = True,       # Important for computation
    # Higher rank following LLaVA recommendations for geometric understanding 
    r = 128,                           # Higher rank for complex geometric tasks
    lora_alpha = 256,                  # Higher alpha for stability in geometric reasoning
    lora_dropout = 0,                  # Optimized value
    bias = "none",                     # Optimized setting
    use_gradient_checkpointing = "unsloth",  # For memory efficiency
    random_state = 3407,               # For reproducibility
    max_seq_length = max_seq_length
)
```


## Dataset Preparation

Prepare the Geo170K dataset for fine-tuning:

```python
import json
import pandas as pd
from datasets import Dataset

# Load G-LLaVA dataset
# Modify paths based on your setup
alignment_data = json.load(open("playground/data/alignment.json"))
qa_data = json.load(open("playground/data/qa_tuning.json"))

# Prepare dataset for Unsloth
def prepare_training_data(data):
    prepared_data = []
    for item in data:
        # Format based on whether it's alignment or QA data
        if "caption" in item:  # Alignment data
            prepared_data.append({
                "question": "Describe this geometric figure in detail.",
                "answer": item["caption"],
                "image_path": item["image"]
            })
        else:  # QA data
            prepared_data.append({
                "question": item["question"],
                "answer": item["answer"],
                "image_path": item["image"]
            })
    return prepared_data

# Combine and prepare data
combined_data = prepare_training_data(alignment_data) + prepare_training_data(qa_data)
df = pd.DataFrame(combined_data)
dataset = Dataset.from_pandas(df)

# Process dataset for model
def process_dataset(examples):
    # Process according to model's requirements
    processed = tokenizer.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": q}, {"type": "image_url", "image_url": {"url": img}}]}, 
         {"role": "assistant", "content": a}]
        for q, a, img in zip(examples["question"], examples["answer"], examples["image_path"])
    )
    return {"text": processed}

tokenized_dataset = dataset.map(process_dataset, batched=True)
```


## Training Configuration

Set training parameters optimized for geometric fine-tuning:

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="g-llava-finetuned",
    num_train_epochs=1,                # Start with 1 epoch to avoid overfitting
    per_device_train_batch_size=2,     # Adjust based on VRAM
    gradient_accumulation_steps=4,     # Simulate larger batch sizes
    learning_rate=2e-4,                # Start higher, adjust as needed
    weight_decay=0.01,                 # Regularization
    warmup_ratio=0.1,                  # Gradual warm-up
    logging_steps=10,
    save_strategy="epoch",
    fp16=True                          # Mixed precision training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)
```


## Training Process

Implement the two-phase training approach of G-LLaVA:

```python
# Phase 1: Geometric Visual-Language Alignment
# Focus on aligning visual features with language
alignment_dataset = tokenized_dataset.filter(lambda x: "describe" in x["text"].lower())
trainer.train_dataset = alignment_dataset
trainer.train()

# Phase 2: Geometric Instruction Tuning
# Focus on solving geometric problems
qa_dataset = tokenized_dataset.filter(lambda x: "describe" not in x["text"].lower())
trainer.train_dataset = qa_dataset
trainer.args.learning_rate = 5e-5  # Lower learning rate for fine-tuning
trainer.train()
```


## Evaluation

Evaluate the model on geometric problems:

```python
# Load test dataset from G-LLaVA
test_data = json.load(open("playground/data/test_question.jsonl"))
test_dataset = prepare_training_data(test_data)

# Function to evaluate model on geometric problems
def evaluate_geometric_understanding(model, test_dataset):
    correct = 0
    total = len(test_dataset)
    
    for item in test_dataset:
        # Process test example and get model prediction
        # Compare with expected answer
        # Increment correct if prediction matches
        pass
    
    return correct / total

accuracy = evaluate_geometric_understanding(model, test_dataset)
print(f"Geometric problem-solving accuracy: {accuracy:.2%}")
```


## Export Model

Save the fine-tuned model for deployment:

```python
# Save the adapter weights
model.save_pretrained("g-llava-adapter")

# For deployment with Ollama, create a Modelfile
with open("g-llava.Modelfile", "w") as f:
    f.write("""
FROM llava:latest

# Custom configurations for G-LLaVA
PARAMETER temperature 0.1
PARAMETER top_p 0.9

# Specify system prompt for geometric reasoning
SYSTEM """
You are G-LLaVA, a multimodal AI assistant specialized in solving geometric problems. 
You can analyze geometric figures, understand spatial relationships, and solve problems related to angles, lengths, areas, and other geometric concepts.
"""
""")
```


## Best Practices for Success

1. **Dataset Quality**: Focus on the Geo170K dataset's quality, ensuring images are properly aligned with text descriptions and problem statements.
2. **Training Parameters**:
    - Use higher rank (128) and alpha (256) values for geometric reasoning tasks
    - Start with higher learning rates (2e-4) and decrease for fine-tuning phases
    - Use 4-bit quantization for memory efficiency unless using H100 GPUs
3. **Selective Fine-tuning**:
    - Fine-tune both vision and language components for geometric tasks
    - Pay special attention to cross-modal projection layers
4. **Optimization Strategy**:
    - Implement the two-phase training approach (alignment, then instruction tuning)
    - Monitor loss during training - aim for values around 0.5
    - Avoid overfitting by limiting epochs (1-3 recommended)
5. **Evaluation**:
    - Test on problems of varying difficulty (OP=1 through OP≥4)
    - Evaluate on different types of geometric questions (angles, lengths, areas)
6. **Hardware Considerations**:
    - 16GB GPU VRAM is sufficient for LLaVA-7B with 4-bit quantization
    - For 13B models, consider 24GB+ VRAM or gradient checkpointing

This procedure combines G-LLaVA's architectural strengths with Unsloth's optimization capabilities to create an efficient fine-tuning pipeline specifically tailored for geometric problem-solving.

<div style="text-align: center">⁂</div>

[^1]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/57219224/cbfba08c-3177-4dee-8e48-10bfe788c5b1/README.md

[^2]: https://openreview.net/forum?id=px1674Wp3C\&noteId=H99kD23um8

[^3]: https://github.com/haotian-liu/LLaVA

[^4]: https://docs.unsloth.ai/get-started/fine-tuning-guide

[^5]: https://www.firecrawl.dev/blog/gemma-3-fine-tuning-firecrawl-unsloth

[^6]: https://github.com/unslothai/unsloth

[^7]: https://arxiv.org/html/2312.11370

[^8]: https://www.reddit.com/r/LocalLLaMA/comments/1gwoqm9/llama_32_vision_finetuning_now_in_unsloth_16gb/

[^9]: https://blog.futuresmart.ai/fine-tune-llama-32-vision-language-model-on-custom-datasets

[^10]: https://arxiv.org/abs/2312.11370

[^11]: https://dblp.org/rec/journals/corr/abs-2312-11370

[^12]: https://llava-vl.github.io

[^13]: https://www.youtube.com/watch?v=eIziN2QUt8U

[^14]: https://x.com/arankomatsuzaki/status/1736982740036141445

[^15]: https://openreview.net/pdf/ce070b8e6451fc25a6264f6f6383eec4b2bc65d9.pdf

[^16]: https://github.com/haotian-liu/LLaVA/issues/1018

[^17]: https://github.com/pipilurj/G-LLaVA

[^18]: https://www.youtube.com/watch?v=wW93ygFGhQQ

[^19]: https://dl.acm.org/doi/10.1145/3688866.3689124

[^20]: https://www.linkedin.com/posts/ilyes-ben-khalifa-112045221_top-article-on-llava-fine-tuning-in-2024-activity-7165067353074974720-c1Uj

[^21]: https://github.com/haotian-liu/LLaVA/issues/219

[^22]: https://docs.unsloth.ai/basics/vision-fine-tuning

[^23]: https://docs.unsloth.ai/get-started/beginner-start-here/faq-+-is-fine-tuning-right-for-me

[^24]: https://www.kaggle.com/code/danielhanchen/qwen2-vision-finetuning-unsloth-kaggle

[^25]: https://github.com/unslothai/unsloth/issues/1559

[^26]: https://unfoldai.com/vision-fine-tuning-unslothai/

[^27]: https://github.com/unslothai/unsloth

[^28]: https://github.com/unslothai/unsloth/issues/2170

[^29]: https://docs.unsloth.ai/basics/tutorials-how-to-fine-tune-and-run-llms

[^30]: https://unsloth.ai

[^31]: https://www.youtube.com/watch?v=dMY3dBLojTk

[^32]: https://github.com/hiyouga/LLaMA-Factory

[^33]: https://colab.research.google.com/drive/1j0N4XTY1zXXy7mPAhOC1_gMYZ2F2EBlk?usp=sharing

[^34]: https://openreview.net/forum?id=px1674Wp3C\&noteId=H99kD23um8

[^35]: https://www.youtube.com/watch?v=MQwryfkydc0

[^36]: https://www.reddit.com/r/LocalLLaMA/comments/1bd18y8/gemma_finetuning_should_be_much_better_now/

[^37]: https://www.reddit.com/r/unsloth/comments/1k12ryj/new_datasets_guide_for_finetuning_best_practices/

[^38]: https://www.kaggle.com/code/danielhanchen/llama-3-2-vision-finetuning-unsloth-kaggle/notebook

[^39]: https://multiplatform.ai/introducing-g-llava-revolutionizing-geometric-problem-solving-and-outshining-gpt-4-v-with-the-innovative-geo170k-dataset/

[^40]: https://www.marktechpost.com/2023/12/21/meet-g-llava-the-game-changer-in-geometric-problem-solving-and-surpasses-gpt-4-v-with-the-innovative-geo170k-dataset/

[^41]: https://arxiv.org/html/2408.13296v1

[^42]: https://huggingface.co/docs/transformers/en/model_doc/llava

[^43]: https://huggingface.co/docs/transformers/v4.37.1/en/model_doc/llava

[^44]: https://www.datacamp.com/tutorial/unsloth-guide-optimize-and-speed-up-llm-fine-tuning

[^45]: https://www.youtube.com/watch?v=Lt7KrFMcCis

[^46]: https://www.kdnuggets.com/fine-tuning-llama-using-unsloth

[^47]: https://www.linkedin.com/posts/itsomarsalah26_ai-machinelearning-llm-activity-7251592613290508291-CmgN

[^48]: https://www.youtube.com/watch?v=fUoVp7dcCqc

[^49]: https://arxiv.org/html/2312.11370

[^50]: https://unsloth.ai/blog/vision

