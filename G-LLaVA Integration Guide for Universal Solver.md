<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# G-LLaVA Integration Guide for Universal Solver

This guide provides detailed instructions for integrating G-LLaVA with your Universal Solver project using Ollama or LM Studio as local providers.

## G-LLaVA Installation and Setup

G-LLaVA is an open-source multimodal model specifically designed for geometric problem solving. Here's how to set it up:

### Option 1: Direct Installation from GitHub

```bash
# Clone the repository
git clone https://github.com/pipilurj/G-LLaVA
cd G-LLaVA

# Create and activate conda environment
conda create -n gllava python=3.10 -y
conda activate gllava

# Install the package
pip install -e .

# Enable DeepSpeed (optional but recommended for performance)
pip install deepspeed
```


### Data Preparation

```bash
# Create data directory structure
mkdir -p playground/data/images/geo3k
mkdir -p playground/data/images/geoqa_plus
mkdir -p playground/data/images/test

# Download the dataset from the G-LLaVA repository
# Place files in the following structure:
# playground/data/
# ├── images/
# │   ├── geo3k/
# │   ├── geoqa_plus/
# │   ├── test/
# ├── alignment.json
# ├── qa_tuning.json
# ├── test_question.jsonl
# ├── test_answers.jsonl
```


### Model Training (if needed)

First stage (alignment):

```bash
bash scripts/run_alignment.sh
```

Second stage (instruction tuning):

```bash
bash scripts/run_qa.sh
```


## Option 2: Running G-LLaVA via Ollama

For a simpler setup, you can use Ollama to run LLaVA models locally:

1. Download and install Ollama from https://ollama.com/[^3]
2. Run the LLaVA model with Ollama:
```bash
ollama run llava
```

3. Create a custom model definition for G-LLaVA (create a file named `gllava.Modelfile`):
```
FROM llava:latest

# Custom configurations for G-LLaVA
# You may need to modify based on the specific G-LLaVA implementation
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER seed 42

# Specify system prompt for geometric reasoning
SYSTEM """
You are G-LLaVA, a multimodal AI assistant specialized in solving geometric problems. 
You can analyze geometric figures, understand spatial relationships, and solve problems related to angles, lengths, areas, and other geometric concepts.
"""
```

4. Create the custom model:
```bash
ollama create gllava -f gllava.Modelfile
```

5. Run the custom G-LLaVA model:
```bash
ollama run gllava
```


## Option 3: Using LM Studio as Provider

1. Download and install LM Studio from their website
2. Set up LM Studio as a local inference server:
```bash
# In your Python environment
pip install lmstudio-client
```

3. Configure LM Studio to serve the G-LLaVA model by loading the model weights from the G-LLaVA repository[^5]

## Integration with Universal Solver Project

Based on the attached README.md, here's how to integrate G-LLaVA with the Universal Solver project:

### 1. Add G-LLaVA to the Advanced Math Resolver

Create a new module in the `adv_resolver_math` directory:

```python
# adv_resolver_math/gllava_solver.py

import os
from typing import Dict, Any, Optional

class GLLaVASolver:
    """Solver that uses G-LLaVA model for geometric problems with diagrams."""
    
    def __init__(self, use_ollama: bool = True, api_url: Optional[str] = None):
        """
        Initialize the G-LLaVA solver.
        
        Args:
            use_ollama: Whether to use Ollama (True) or LM Studio (False)
            api_url: API URL for the model server (default: http://localhost:11434/api/generate for Ollama)
        """
        self.use_ollama = use_ollama
        self.api_url = api_url or "http://localhost:11434/api/generate"
        
    def solve(self, problem: Dict[str, Any]) -&gt; Dict[str, Any]:
        """
        Solve a geometric problem using G-LLaVA.
        
        Args:
            problem: Dictionary containing 'text' and optionally 'image_path'
            
        Returns:
            Dictionary with solution and metadata
        """
        # Implementation depends on whether using Ollama or LM Studio
        if self.use_ollama:
            return self._solve_with_ollama(problem)
        else:
            return self._solve_with_lmstudio(problem)
    
    def _solve_with_ollama(self, problem: Dict[str, Any]) -&gt; Dict[str, Any]:
        """Solve problem using Ollama API"""
        import requests
        import json
        
        prompt = problem['text']
        image_path = problem.get('image_path')
        
        payload = {
            "model": "gllava",
            "prompt": prompt,
            "stream": False
        }
        
        # Add image if provided
        if image_path and os.path.exists(image_path):
            import base64
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                payload["images"] = [encoded_image]
        
        response = requests.post(self.api_url, json=payload)
        result = response.json()
        
        return {
            "solution": result.get("response", ""),
            "model_used": "G-LLaVA (Ollama)",
            "confidence": 0.9,  # Placeholder
            "metadata": result
        }
    
    def _solve_with_lmstudio(self, problem: Dict[str, Any]) -&gt; Dict[str, Any]:
        """Solve problem using LM Studio"""
        # Implementation for LM Studio
        # This would depend on the LM Studio API
        pass
```


### 2. Update Requirements File

Add the following to your `requirements.txt`:

```
# G-LLaVA dependencies
requests&gt;=2.28.0
pillow&gt;=9.0.0
```


### 3. Register G-LLaVA Solver in the Math Ensemble

Modify your solver registry to include G-LLaVA:

```python
# adv_resolver_math/solver_registry.py

from .gllava_solver import GLLaVASolver

def register_solvers():
    solvers = {
        # Existing solvers
        "gllava_ollama": GLLaVASolver(use_ollama=True),
        "gllava_lmstudio": GLLaVASolver(use_ollama=False, 
                                        api_url="http://localhost:8080/v1/completions")
    }
    return solvers
```


### 4. Create a Benchmarking Workflow for Geometric Problems

```python
# benchmark_geometry.py

from benchmark_datasets import load_benchmark_dataset, get_problem_and_answer
from adv_resolver_math.solver_registry import register_solvers
import pandas as pd
import os

def run_geometry_benchmark(dataset_name="geoqa", sample_size=20, out_dir="geometry_results"):
    """Run benchmark on geometry datasets using G-LLaVA"""
    
    # Load dataset
    ds = load_benchmark_dataset(dataset_name, sample_size=sample_size)
    
    # Get solvers
    solvers = register_solvers()
    gllava_solver = solvers["gllava_ollama"]  # or gllava_lmstudio
    
    results = []
    
    for ex in ds:
        problem, answer = get_problem_and_answer(ex, dataset_name)
        
        # Prepare problem with image path if available
        problem_input = {
            "text": problem,
            "image_path": ex.get("image_path")  # Assuming dataset provides image paths
        }
        
        # Solve with G-LLaVA
        solution = gllava_solver.solve(problem_input)
        
        # Evaluate result
        is_correct = evaluate_solution(solution["solution"], answer)
        
        results.append({
            "problem": problem,
            "expected_answer": answer,
            "solution": solution["solution"],
            "is_correct": is_correct,
            "model_used": solution["model_used"]
        })
    
    # Save results
    os.makedirs(out_dir, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{out_dir}/{dataset_name}_gllava_results.csv", index=False)
    
    # Print summary
    correct = sum(r["is_correct"] for r in results)
    print(f"Accuracy: {correct/len(results):.2%} ({correct}/{len(results)})")
    
    return results

def evaluate_solution(solution, answer):
    """Basic evaluation logic - would need to be customized"""
    # Implement appropriate evaluation logic
    return answer.strip() in solution
```


## Running the Integration

1. Ensure Ollama or LM Studio is running with the G-LLaVA model loaded
2. Run a benchmark:
```bash
python benchmark_geometry.py --dataset geoqa --sample-size 10 --out-dir showcase_results
```

3. For CLI usage:
```bash
python benchmark_cli.py --solver gllava_ollama --dataset geoqa --sample-size 5 --out-dir results
```


## Model Availability and Performance

- G-LLaVA-7B and G-LLaVA-13B models are available from the original repository[^2]
- The model significantly outperforms GPT4-V on geometry tasks in the MathVista benchmark despite having only 7B parameters[^6]
- G-LLaVA leverages the Geo170K dataset, which contains more than 170,000 geometric image-caption and question-answer pairs[^6]


## Limitations and Considerations

1. G-LLaVA is specifically designed for geometric problems and may not perform well on other mathematical domains
2. Running locally requires sufficient hardware resources (at least 8GB RAM, preferably with a GPU)[^4]
3. For optimal performance on complex geometric problems, consider fine-tuning the model on your specific dataset

This integration setup provides a solid foundation for incorporating G-LLaVA's geometric problem-solving capabilities into your Universal Solver project.

<div style="text-align: center">⁂</div>

[^1]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/57219224/cbfba08c-3177-4dee-8e48-10bfe788c5b1/README.md

[^2]: https://github.com/pipilurj/G-LLaVA

[^3]: https://ollama.com/library/llava

[^4]: https://merlio.app/blog/run-llava-locally-guide

[^5]: https://pyimagesearch.com/2024/06/24/integrating-local-llm-frameworks-a-deep-dive-into-lm-studio-and-anythingllm/

[^6]: https://openreview.net/forum?id=px1674Wp3C\&noteId=H99kD23um8

[^7]: https://www.gravio.com/en-blog/tutorial-using-ollama-llava-and-gravio-to-build-a-local-visual-question-and-answer-ai-assistant

[^8]: https://www.toolify.ai/ai-news/experience-the-power-of-local-multimodal-ai-with-ollama-and-llava-2038676

[^9]: https://llava-vl.github.io

[^10]: https://huggingface.co/renjiepi/G-LLaVA-13B

[^11]: https://github.com/LLaVA-VL/LLaVA-NeXT

[^12]: https://huggingface.co/docs/transformers/en/model_doc/llava

[^13]: https://huggingface.co/renjiepi/G-LLaVA-7B

[^14]: https://github.com/MapEval/MapEval-Visual

[^15]: https://arxiv.org/abs/2312.11370

[^16]: https://dl.acm.org/doi/10.1145/3688866.3689124

[^17]: https://llava-vl.github.io/llava-grounding/

[^18]: https://towardsdatascience.com/llava-an-open-source-alternative-to-gpt-4v-ision-b06f88ce8efa/

[^19]: https://llava-vl.github.io/llava-plus/

[^20]: https://www.youtube.com/watch?v=CNGwscEsl0E

[^21]: https://python.langchain.com/docs/integrations/llms/ollama/

[^22]: https://github.com/ollama/ollama

[^23]: https://llama-2.ai/how-to-install-llava-llm-locally/

[^24]: https://www.byteplus.com/en/topic/516166

[^25]: https://lmstudio.ai

[^26]: https://ollama.com/blog/vision-models?__from__=talkingdev

[^27]: https://dev.to/auden/how-to-run-ai-models-locally-with-ollama-deploy-llms-and-debug-apis-in-minutes-59pc

[^28]: https://ollama.com

[^29]: https://github.com/microsoft/autogen/issues/1234

[^30]: https://www.kaggle.com/code/alfathterry/llava-via-ollama-image-annotation

[^31]: https://www.youtube.com/watch?v=2Tv5ZfPabGM

[^32]: https://github.com/haotian-liu/LLaVA

[^33]: https://multiplatform.ai/introducing-g-llava-revolutionizing-geometric-problem-solving-and-outshining-gpt-4-v-with-the-innovative-geo170k-dataset/

[^34]: https://www.marktechpost.com/2023/12/21/meet-g-llava-the-game-changer-in-geometric-problem-solving-and-surpasses-gpt-4-v-with-the-innovative-geo170k-dataset/

[^35]: https://github.com/AIDC-AI/TG-LLaVA

[^36]: https://arxiv.org/html/2312.11370

[^37]: https://paperswithcode.com/paper/g-llava-solving-geometric-problem-with-multi

[^38]: https://docs.vultr.com/how-to-install-lmstudio-a-graphical-application-for-running-large-language-models-llms

[^39]: https://www.youtube.com/watch?v=KEkWeeYAztA

[^40]: https://www.youtube.com/watch?v=smvSivZApdI

