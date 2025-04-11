# math_ensemble_langchain_ollama.py
import os
import re
import time
import json
import asyncio
import sympy
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

# LangChain components
from langchain.llms import Ollama
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_core.runnables import RunnableConfig
from langchain.schema.runnable import RunnablePassthrough

# For benchmarking
import datasets
from datasets import load_dataset

# Define the math prompt template
MATH_PROMPT_TEMPLATE = """
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


class MathSolvingCallbackHandler(BaseCallbackHandler):
    """Callback handler for monitoring math solving progress."""
    
    def __init__(self, console: Console, model_name: str):
        self.console = console
        self.model_name = model_name
        self.start_time = None
        self.tokens = 0
        
    def on_llm_start(self, *args, **kwargs):
        self.start_time = time.time()
        self.console.print(f"[dim]{self.model_name} is thinking...[/dim]")
        
    def on_llm_new_token(self, token: str, **kwargs):
        self.tokens += 1
        if self.tokens % 50 == 0:
            self.console.print(f"[dim]{self.model_name}: {self.tokens} tokens generated[/dim]", end="\r")
            
    def on_llm_end(self, *args, **kwargs):
        elapsed = time.time() - self.start_time
        self.console.print(f"[dim]{self.model_name} completed in {elapsed:.2f} seconds, generated {self.tokens} tokens[/dim]")
        
    def on_llm_error(self, error: Exception, **kwargs):
        self.console.print(f"[red]Error with {self.model_name}: {error}[/red]")


class MathEnsembleSolver:
    """
    Advanced Math Problem Solving Ensemble using LangChain with Ollama
    for local LLM execution in both sequential and parallel modes.
    """
    
    def __init__(
        self, 
        models: List[str] = None, 
        mode: str = "parallel",
        temperature: float = 0.1,
        max_tokens: int = 512,
        ollama_base_url: str = "http://localhost:11434",
        use_cache: bool = True,
        cache_dir: str = "./math_cache",
        verbose: bool = True
    ):
        self.console = Console()
        self.models = models or ["cogito:3b", "gemma3", "phi4-mini:latest"]
        self.mode = mode.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ollama_base_url = ollama_base_url
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.verbose = verbose
        
        # Setup environment and initialize models
        self._setup_environment()
        self.model_chains = self._initialize_models()
        
        # Set up answer extraction and validation
        self.answer_pattern = re.compile(r"\\boxed{([^}]+)}|\*\*Final Answer\*\*:\s*([^\n]+)")
        
        # Cache setup
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache_file = os.path.join(self.cache_dir, "math_solutions_cache.json")
            self._load_cache()
    
    def _setup_environment(self):
        """Setup the environment for math solving."""
        # Create cache directory if needed
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        # Check if Ollama service is running
        try:
            import requests
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2)
            if response.status_code != 200:
                self.console.print("[yellow]Warning: Ollama service may not be running correctly[/yellow]")
                self.console.print(f"[yellow]Response code: {response.status_code}[/yellow]")
        except Exception as e:
            self.console.print("[red]Error: Cannot connect to Ollama service[/red]")
            self.console.print(f"[red]Make sure Ollama is running at {self.ollama_base_url}[/red]")
            self.console.print(f"[red]Error details: {e}[/red]")
            self.console.print("[yellow]You can download Ollama from https://ollama.ai/[/yellow]")
    
    def _initialize_models(self):
        """Initialize all LLM chains for math problem solving."""
        model_chains = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            for model_name in self.models:
                task = progress.add_task(f"[cyan]Initializing {model_name}...[/cyan]", total=1)
                
                try:
                    callback_handler = MathSolvingCallbackHandler(
                        console=self.console, 
                        model_name=model_name
                    ) if self.verbose else None
                    
                    callbacks = [callback_handler] if callback_handler else []
                    
                    # Initialize the Ollama LLM with the current model
                    llm = Ollama(
                        model=model_name,
                        base_url=self.ollama_base_url,
                        temperature=self.temperature,
                        callbacks=callbacks,
                        num_ctx=4096,
                        num_predict=self.max_tokens,
                    )
                    
                    # Create prompt template
                    prompt = PromptTemplate(
                        template=MATH_PROMPT_TEMPLATE,
                        input_variables=["problem"]
                    )
                    
                    # Create chain
                    chain = prompt | llm
                    model_chains[model_name] = chain
                    
                    progress.update(task, completed=1, description=f"[green]Initialized {model_name}[/green]")
                except Exception as e:
                    progress.update(task, completed=1, description=f"[red]Failed to initialize {model_name}: {e}[/red]")
        
        return model_chains
    
    def _load_cache(self):
        """Load the solution cache from disk."""
        self.cache = {}
        if self.use_cache and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                self.console.print(f"[dim]Loaded {len(self.cache)} cached solutions[/dim]")
            except Exception as e:
                self.console.print(f"[yellow]Error loading cache: {e}[/yellow]")
                self.cache = {}
    
    def _save_cache(self):
        """Save the solution cache to disk."""
        if self.use_cache:
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.cache, f)
            except Exception as e:
                self.console.print(f"[yellow]Error saving cache: {e}[/yellow]")
    
    def _extract_answer(self, solution):
        """Extract the final answer from a solution text."""
        if not solution:
            return None
            
        # Look for boxed answer or final answer format
        match = self.answer_pattern.search(solution)
        if match:
            # Return the first non-None group
            return next((g for g in match.groups() if g is not None), None)
            
        # If no standard format found, try to find the last line with an equals sign
        lines = solution.strip().split('\n')
        for line in reversed(lines):
            if '=' in line:
                # Extract what's after the last equals sign
                return line.split('=')[-1].strip()
                
        return None
    
    def _validate_expression(self, expr):
        """
        Symbolically validate a mathematical expression using SymPy.
        
        Args:
            expr: The mathematical expression to validate
            
        Returns:
            Simplified expression or None if invalid
        """
        if not expr:
            return None
            
        try:
            # Handle common answer formats
            # Convert fractions like "3/4" to sympy Rational
            if '/' in expr and all(c.isdigit() or c == '/' for c in expr):
                num, denom = expr.split('/')
                return str(sympy.Rational(int(num), int(denom)))
                
            # Handle decimal answers
            expr = expr.replace(',', '')  # Remove commas in numbers
            
            return str(sympy.simplify(expr))
        except (sympy.SympifyError, TypeError, ValueError):
            try:
                # Try alternative parsing for complex expressions
                return str(eval(expr))
            except:
                return None
    
    def _score_solution(self, problem, solution):
        """
        Score the quality of a mathematical solution.
        
        Args:
            problem: The original math problem
            solution: The model's solution text
            
        Returns:
            A score between 0 and 1
        """
        # Extract and validate the answer
        raw_answer = self._extract_answer(solution)
        validated_answer = self._validate_expression(raw_answer)
        
        # Answer correctness (50% weight)
        answer_score = 0.5 if validated_answer else 0.0
        
        # Step completeness and correctness (30% weight)
        steps = re.split(r"(?:Step \d+|\\item|\n\d+\.)", solution)
        if len(steps) <= 1:  # No clear steps
            step_score = 0.0
        else:
            valid_steps = 0
            for step in steps[1:]:  # Skip first element which is usually empty
                step = step.strip()
                if not step:
                    continue
                    
                # Check if step has some mathematical expression
                if re.search(r'[=<>+\-*/]', step):
                    valid_steps += 1
            
            step_score = 0.3 * (valid_steps / max(1, len(steps) - 1))
        
        # Solution coherence and relevance (20% weight)
        # Count problem keywords in solution
        problem_words = set(re.findall(r'\b\w+\b', problem.lower()))
        solution_words = set(re.findall(r'\b\w+\b', solution.lower()))
        relevance_score = 0.2 * (len(problem_words.intersection(solution_words)) / max(1, len(problem_words)))
        
        return answer_score + step_score + relevance_score
    
    def _solve_sequential(self, problem):
        """
        Solve a math problem using all models sequentially.
        
        Args:
            problem: The math problem to solve
            
        Returns:
            Dictionary of model solutions and scores
        """
        # Check cache first
        cache_key = problem.strip()
        if self.use_cache and cache_key in self.cache:
            self.console.print("[dim]Using cached solution[/dim]")
            return self.cache[cache_key]
            
        solutions = {}
        scores = {}
        
        for model_name, chain in self.model_chains.items():
            try:
                self.console.print(f"[cyan]Solving with {model_name}...[/cyan]")
                start_time = time.time()
                
                # Run the chain
                solution = chain.invoke({"problem": problem})
                
                elapsed = time.time() - start_time
                self.console.print(f"[dim]{model_name} completed in {elapsed:.2f} seconds[/dim]")
                
                # Score the solution
                score = self._score_solution(problem, solution)
                solutions[model_name] = solution
                scores[model_name] = score
                
                self.console.print(f"[green]{model_name} score: {score:.2f}[/green]")
            except Exception as e:
                self.console.print(f"[red]Error with {model_name}: {e}[/red]")
                solutions[model_name] = f"Error: {e}"
                scores[model_name] = 0.0
        
        # Create and cache the result
        result = {
            "problem": problem,
            "solutions": solutions,
            "scores": scores,
            "best_model": max(scores.items(), key=lambda x: x[1])[0] if scores else None
        }
        
        if self.use_cache:
            self.cache[cache_key] = result
            self._save_cache()
            
        return result
    
    async def _solve_async(self, problem):
        """
        Solve a math problem asynchronously using all models in parallel.
        
        Args:
            problem: The math problem to solve
            
        Returns:
            Dictionary of model solutions and scores
        """
        # Check cache first
        cache_key = problem.strip()
        if self.use_cache and cache_key in self.cache:
            self.console.print("[dim]Using cached solution[/dim]")
            return self.cache[cache_key]
        
        solutions = {}
        scores = {}
        
        async def solve_with_model(model_name, chain):
            try:
                # Run the chain
                solution = await chain.ainvoke({"problem": problem})
                # Score the solution
                score = self._score_solution(problem, solution)
                return model_name, solution, score
            except Exception as e:
                self.console.print(f"[red]Error with {model_name}: {e}[/red]")
                return model_name, f"Error: {e}", 0.0
                
        # Create tasks for all models
        tasks = [solve_with_model(model_name, chain) 
                for model_name, chain in self.model_chains.items()]
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Process results
        for model_name, solution, score in results:
            solutions[model_name] = solution
            scores[model_name] = score
        
        # Create and cache the result
        result = {
            "problem": problem,
            "solutions": solutions,
            "scores": scores,
            "best_model": max(scores.items(), key=lambda x: x[1])[0] if scores else None
        }
        
        if self.use_cache:
            self.cache[cache_key] = result
            self._save_cache()
            
        return result
    
    def solve(self, problem):
        """Solve a math problem using the configured mode."""
        if self.mode == "sequential":
            return self._solve_sequential(problem)
        elif self.mode == "parallel":
            return asyncio.run(self._solve_async(problem))
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'sequential' or 'parallel'.")
    
    def display_results(self, result):
        """Display the results of a math problem solution."""
        if not result:
            self.console.print("[red]No results to display[/red]")
            return
            
        problem = result.get("problem", "Unknown problem")
        solutions = result.get("solutions", {})
        scores = result.get("scores", {})
        best_model = result.get("best_model")
        
        # Create a table for the results
        table = Table(title=f"Solutions for: {problem}")
        table.add_column("Model", style="cyan")
        table.add_column("Score", style="magenta")
        table.add_column("Answer", style="green")
        
        for model_name, solution in solutions.items():
            score = scores.get(model_name, 0.0)
            answer = self._extract_answer(solution) or "No answer found"
            
            # Highlight the best model
            if model_name == best_model:
                model_display = f"[bold]{model_name} (Best)[/bold]"
            else:
                model_display = model_name
                
            table.add_row(model_display, f"{score:.2f}", answer)
            
        self.console.print(table)
        
        # Display the best solution in detail
        if best_model and best_model in solutions:
            self.console.print(f"\n[bold]Best Solution ({best_model}):[/bold]")
            self.console.print(Panel(solutions[best_model], expand=False))
    
    def benchmark(self, dataset_name="gsm8k", split="test", num_samples=10):
        """
        Benchmark the math solver on a standard dataset.
        
        Args:
            dataset_name: Name of the dataset (default: gsm8k)
            split: Dataset split to use
            num_samples: Number of samples to evaluate (use -1 for all)
            
        Returns:
            Dictionary with benchmark results
        """
        self.console.print(f"[bold]Benchmarking on {dataset_name} dataset ({split} split)[/bold]")
        
        # Load the dataset
        try:
            dataset = load_dataset(dataset_name, 'main', split=split)
        except Exception as e:
            self.console.print(f"[red]Error loading dataset: {e}[/red]")
            return {"error": str(e)}
            
        # Limit to num_samples if specified
        if num_samples > 0:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            
        self.console.print(f"Running benchmark on {len(dataset)} problems...")
        
        results = []
        correct_counts = {model: 0 for model in self.models}
        total_scores = {model: 0.0 for model in self.models}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Benchmarking...[/cyan]", total=len(dataset))
            
            for i, item in enumerate(dataset):
                # Extract problem and answer based on dataset format
                if dataset_name == "gsm8k":
                    problem = item.get("question", "")
                    reference_answer = item.get("answer", "")
                elif dataset_name == "math":
                    problem = item.get("problem", "")
                    reference_answer = item.get("solution", "")
                else:
                    # Generic fallback
                    problem = str(item)
                    reference_answer = ""
                
                # Solve the problem
                result = self.solve(problem)
                results.append(result)
                
                # Update statistics
                for model, score in result.get("scores", {}).items():
                    total_scores[model] += score
                    
                    # Check if answer is correct (simplified comparison)
                    solution = result.get("solutions", {}).get(model, "")
                    extracted_answer = self._extract_answer(solution)
                    if extracted_answer and reference_answer and \
                       (extracted_answer in reference_answer or reference_answer in extracted_answer):
                        correct_counts[model] += 1
                
                progress.update(task, advance=1, description=f"[cyan]Problem {i+1}/{len(dataset)}[/cyan]")
        
        # Calculate statistics
        model_accuracy = {model: count / len(dataset) for model, count in correct_counts.items()}
        avg_model_scores = {model: score / len(dataset) for model, score in total_scores.items()}
        
        # Determine best model
        best_model = max(model_accuracy.items(), key=lambda x: x[1])[0] if model_accuracy else None
        
        # Create benchmark result
        benchmark_result = {
            "dataset": dataset_name,
            "split": split,
            "num_samples": len(dataset),
            "model_accuracy": model_accuracy,
            "avg_model_scores": avg_model_scores,
            "best_model": best_model,
            "results": results
        }
        
        # Display summary
        self._display_benchmark_summary(benchmark_result)
        
        # Generate plots
        self._plot_benchmark_results(benchmark_result)
        
        return benchmark_result
    
    def _display_benchmark_summary(self, benchmark_result):
        """Display a summary of benchmark results."""
        self.console.print(f"\n[bold]Benchmark Summary: {benchmark_result['dataset']}[/bold]")
        
        # Create a table for the results
        table = Table(title=f"Benchmark Results ({benchmark_result['num_samples']} problems)")
        table.add_column("Model", style="cyan")
        table.add_column("Accuracy", style="green")
        table.add_column("Avg Score", style="magenta")
        
        for model in self.models:
            accuracy = benchmark_result["model_accuracy"].get(model, 0.0)
            avg_score = benchmark_result["avg_model_scores"].get(model, 0.0)
            
            # Highlight the best model
            if model == benchmark_result["best_model"]:
                model_display = f"[bold]{model} (Best)[/bold]"
            else:
                model_display = model
                
            table.add_row(model_display, f"{accuracy:.2%}", f"{avg_score:.2f}")
            
        self.console.print(table)
    
    def _plot_benchmark_results(self, benchmark_result):
        """Generate plots for benchmark results."""
        try:
            # Create bar chart for accuracy
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            # Accuracy chart
            models = list(benchmark_result["model_accuracy"].keys())
            accuracy = [benchmark_result["model_accuracy"][model] for model in models]
            
            ax[0].bar(models, accuracy, color='skyblue')
            ax[0].set_ylim(0, 1.0)
            ax[0].set_title('Model Accuracy')
            ax[0].set_ylabel('Accuracy')
            ax[0].set_xticks(range(len(models)))
            ax[0].set_xticklabels(models, rotation=45)
            
            # Average score chart
            avg_scores = [benchmark_result["avg_model_scores"][model] for model in models]
            
            ax[1].bar(models, avg_scores, color='lightgreen')
            ax[1].set_ylim(0, 1.0)
            ax[1].set_title('Average Solution Quality Score')
            ax[1].set_ylabel('Avg Score')
            ax[1].set_xticks(range(len(models)))
            ax[1].set_xticklabels(models, rotation=45)
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(self.cache_dir, f"benchmark_{benchmark_result['dataset']}.png")
            plt.savefig(plot_path)
            
            self.console.print(f"[dim]Benchmark plot saved to {plot_path}[/dim]")
        except Exception as e:
            self.console.print(f"[yellow]Error generating plots: {e}[/yellow]")
    
    def export_results(self, benchmark_result, format="json"):
        """Export benchmark results to a file."""
        if not benchmark_result:
            self.console.print("[red]No results to export[/red]")
            return
            
        # Create export directory
        export_dir = os.path.join(self.cache_dir, "exports")
        os.makedirs(export_dir, exist_ok=True)
        
        # Generate filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dataset_name = benchmark_result.get("dataset", "unknown")
        filename = f"{dataset_name}_benchmark_{timestamp}"
        
        if format.lower() == "json":
            # Export as JSON
            file_path = os.path.join(export_dir, f"{filename}.json")
            try:
                # Create a copy without the full results to save space
                export_data = benchmark_result.copy()
                export_data.pop("results", None)  # Remove detailed results
                
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                    
                self.console.print(f"[green]Results exported to {file_path}[/green]")
            except Exception as e:
                self.console.print(f"[red]Error exporting results: {e}[/red]")
        
        elif format.lower() == "csv":
            # Export as CSV
            file_path = os.path.join(export_dir, f"{filename}.csv")
            try:
                # Create a DataFrame with the results
                data = {
                    "Model": list(benchmark_result["model_accuracy"].keys()),
                    "Accuracy": list(benchmark_result["model_accuracy"].values()),
                    "Avg_Score": list(benchmark_result["avg_model_scores"].values())
                }
                
                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False)
                
                self.console.print(f"[green]Results exported to {file_path}[/green]")
            except Exception as e:
                self.console.print(f"[red]Error exporting results: {e}[/red]")
        
        else:
            self.console.print(f"[red]Unsupported export format: {format}[/red]")


class MetaMathEnsemble:
    """
    A meta-ensemble that uses both sequential and parallel approaches
    and determines the best strategy for different problem types.
    """
    
    def __init__(self, models=None, ollama_base_url="http://localhost:11434",
                 use_cache=True, cache_dir="./math_cache", verbose=True):
        self.console = Console()
        self.console.print("[bold]Initializing Meta Math Ensemble[/bold]")
        
        # Create both types of solvers
        self.sequential_solver = MathEnsembleSolver(
            models=models,
            mode="sequential",
            ollama_base_url=ollama_base_url,
            use_cache=use_cache,
            cache_dir=os.path.join(cache_dir, "sequential"),
            verbose=verbose
        )
        
        self.parallel_solver = MathEnsembleSolver(
            models=models,
            mode="parallel",
            ollama_base_url=ollama_base_url,
            use_cache=use_cache,
            cache_dir=os.path.join(cache_dir, "parallel"),
            verbose=verbose
        )
        
        # Problem classifier
        self.problem_classifier = self._initialize_problem_classifier()
    
    def _initialize_problem_classifier(self):
        """Initialize a problem classifier to determine which strategy to use."""
        def classify_problem(problem):
            problem_lower = problem.lower()
            
            # Problem categories
            categories = {
                "algebra": ["solve", "equation", "x =", "y =", "variable"],
                "arithmetic": ["add", "sum", "subtract", "multiply", "divide", "percentage"],
                "geometry": ["angle", "triangle", "circle", "square", "rectangle", "area", "volume"],
                "probability": ["probability", "chance", "likelihood", "random", "dice", "coin"],
                "combination": ["how many ways", "combination", "permutation"]
            }
            
            # Check for keywords
            detected_categories = []
            for category, keywords in categories.items():
                if any(keyword in problem_lower for keyword in keywords):
                    detected_categories.append(category)
            
            # Determine best approach based on heuristics
            if not detected_categories:
                return "unknown", "parallel"  # Default to parallel for unknown
                
            if "algebra" in detected_categories or "arithmetic" in detected_categories:
                return detected_categories[0], "sequential"  # Sequential for algebra/arithmetic
                
            # For more complex problems, use parallel
            return detected_categories[0], "parallel"
        
        return classify_problem
    
    def solve(self, problem):
        """Solve a math problem using the optimal strategy."""
        # Classify the problem and determine the best strategy
        category, strategy = self.problem_classifier(problem)
        
        self.console.print(f"[dim]Problem classified as {category}, using {strategy} strategy[/dim]")
        
        # Use the appropriate solver
        if strategy == "sequential":
            return self.sequential_solver.solve(problem)
        else:
            return self.parallel_solver.solve(problem)
    
    def display_results(self, result):
        """Display the results of a math problem solution."""
        # Use the sequential solver's display method
        self.sequential_solver.display_results(result)
    
    def benchmark(self, dataset_name="gsm8k", split="test", num_samples=10):
        """Run benchmarks using both strategies and compare results."""
        self.console.print(f"[bold]Meta-Ensemble Benchmark on {dataset_name}[/bold]")
        
        # Load the dataset
        try:
            dataset = load_dataset(dataset_name, 'main', split=split)
        except Exception as e:
            self.console.print(f"[red]Error loading dataset: {e}[/red]")
            return {"error": str(e)}
            
        # Limit to num_samples if specified
        if num_samples > 0:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            
        self.console.print(f"Running benchmark on {len(dataset)} problems...")
        
        results = []
        strategy_counts = {"sequential": 0, "parallel": 0}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Benchmarking...[/cyan]", total=len(dataset))
            
            for i, item in enumerate(dataset):
                # Extract problem based on dataset format
                if dataset_name == "gsm8k":
                    problem = item.get("question", "")
                elif dataset_name == "math":
                    problem = item.get("problem", "")
                else:
                    # Generic fallback
                    problem = str(item)
                
                # Classify the problem
                category, strategy = self.problem_classifier(problem)
                strategy_counts[strategy] += 1
                
                # Solve using meta-ensemble
                result = self.solve(problem)
                results.append({
                    "problem": problem,
                    "category": category,
                    "strategy": strategy,
                    "result": result
                })
                
                progress.update(task, advance=1, description=f"[cyan]Problem {i+1}/{len(dataset)}[/cyan]")
        
        # Calculate statistics
        sequential_pct = strategy_counts["sequential"] / len(dataset) * 100
        parallel_pct = strategy_counts["parallel"] / len(dataset) * 100
        
        # Create benchmark result
        benchmark_result = {
            "dataset": dataset_name,
            "split": split,
            "num_samples": len(dataset),
            "strategy_counts": strategy_counts,
            "sequential_percentage": sequential_pct,
            "parallel_percentage": parallel_pct,
            "results": results
        }
        
        # Display summary
        self._display_benchmark_summary(benchmark_result)
        
        return benchmark_result
    
    def _display_benchmark_summary(self, benchmark_result):
        """Display a summary of meta-ensemble benchmark results."""
        self.console.print(f"\n[bold]Meta-Ensemble Benchmark Summary: {benchmark_result['dataset']}[/bold]")
        
        # Create a table for the strategy distribution
        table = Table(title=f"Strategy Distribution ({benchmark_result['num_samples']} problems)")
        table.add_column("Strategy", style="cyan")
        table.add_column("Count", style="magenta")
        table.add_column("Percentage", style="green")
        
        sequential_count = benchmark_result["strategy_counts"]["sequential"]
        parallel_count = benchmark_result["strategy_counts"]["parallel"]
        
        table.add_row(
            "Sequential", 
            str(sequential_count),
            f"{benchmark_result['sequential_percentage']:.1f}%"
        )
        
        table.add_row(
            "Parallel", 
            str(parallel_count),
            f"{benchmark_result['parallel_percentage']:.1f}%"
        )
        
        self.console.print(table)


if __name__ == "__main__":
    console = Console()
    
    console.print(Panel.fit(
        "[bold magenta]Math Ensemble Solver[/bold magenta]\n"
        "State-of-the-Art Math Problem Solving using Local LLMs via Ollama and LangChain",
        border_style="green"
    ))
    
    # Sample math problems for testing
    problems = [
        "If 2x + 5 = 15, what is the value of x?",
        "A rectangle has a length of 10 cm and a width of 5 cm. What is its area?",
        "If the probability of an event is 0.3, what is the probability that it does not occur?",
        "Solve the quadratic equation: x^2 - 5x + 6 = 0",
        "A train travels at 60 km/h. How far will it travel in 2.5 hours?"
    ]
    
    # Ask user which demo to run
    console.print("[bold]Choose a demo mode:[/bold]")
    console.print("1. Quick Demo (solve sample problems)")
    console.print("2. Sequential Mode Benchmark")
    console.print("3. Parallel Mode Benchmark")
    console.print("4. Meta-Ensemble Benchmark")
    console.print("5. Run All Benchmarks (comprehensive)")
    
    choice = input("\nEnter choice (1-5): ")
    
    # Execute the chosen demo
    if choice == "1":
        # Quick demo with sample problems
        solver = MathEnsembleSolver(mode="parallel")
        for i, problem in enumerate(problems):
            console.print(f"\n[bold]Problem {i+1}[/bold]: {problem}")
            result = solver.solve(problem)
            solver.display_results(result)
    
    elif choice == "2":
        # Sequential benchmark
        solver = MathEnsembleSolver(mode="sequential")
        benchmark_result = solver.benchmark(num_samples=5)
        solver.export_results(benchmark_result)
    
    elif choice == "3":
        # Parallel benchmark
        solver = MathEnsembleSolver(mode="parallel")
        benchmark_result = solver.benchmark(num_samples=5)
        solver.export_results(benchmark_result)
    
    elif choice == "4":
        # Meta-ensemble benchmark
        solver = MetaMathEnsemble()
        benchmark_result = solver.benchmark(num_samples=5)
    
    elif choice == "5":
        # Comprehensive benchmarks
        console.print("\n[bold]Running Sequential Benchmark[/bold]")
        seq_solver = MathEnsembleSolver(mode="sequential")
        seq_result = seq_solver.benchmark(num_samples=3)
        
        console.print("\n[bold]Running Parallel Benchmark[/bold]")
        par_solver = MathEnsembleSolver(mode="parallel")
        par_result = par_solver.benchmark(num_samples=3)
        
        console.print("\n[bold]Running Meta-Ensemble Benchmark[/bold]")
        meta_solver = MetaMathEnsemble()
        meta_result = meta_solver.benchmark(num_samples=3)
    
    else:
        console.print("[red]Invalid choice. Exiting.[/red]")