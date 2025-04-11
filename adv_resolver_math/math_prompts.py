# math_prompts.py

# Define the math prompt template for different models

BASE_MATH_PROMPT = """
You are an expert mathematical problem solver. Your task is to solve the following math problem step-by-step, showing all your work clearly.

Problem: {problem}

Please follow these guidelines:
1. Break down the problem into clear, logical steps
2. Explain your reasoning at each step
3. Use mathematical notation where appropriate
4. Verify your answer
5. Present your final answer in a \\boxed{{answer}} format or as **Final Answer**: [your answer]

Solution:
"""

# Model-specific prompts can be defined here if needed
LLAMA_MATH_PROMPT = BASE_MATH_PROMPT
MISTRAL_MATH_PROMPT = BASE_MATH_PROMPT
GEMMA_MATH_PROMPT = BASE_MATH_PROMPT

# Dictionary mapping model names to their specialized prompts
MODEL_PROMPTS = {
    "llama3": LLAMA_MATH_PROMPT,
    "mistral": MISTRAL_MATH_PROMPT,
    "gemma": GEMMA_MATH_PROMPT,
    # Add more models as needed
}

# Default prompt to use if model not found in the dictionary
DEFAULT_MATH_PROMPT = BASE_MATH_PROMPT

def get_math_prompt(model_name):
    """Get the appropriate math prompt for a given model."""
    return MODEL_PROMPTS.get(model_name.lower(), DEFAULT_MATH_PROMPT)