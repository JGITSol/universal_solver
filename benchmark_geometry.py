"""
benchmark_geometry.py

Benchmark script for evaluating G-LLaVA-based solvers on geometry datasets.
Provides functions to run benchmarks, evaluate solutions, and output results.
"""
from adv_resolver_math.solver_registry import register_solvers
from benchmark_datasets import load_benchmark_dataset, get_problem_and_answer
import pandas as pd
import os

def evaluate_solution(solution, answer):
    """
    Evaluate if the solution is correct for the given answer.

    Args:
        solution (str): The solution string returned by the solver.
        answer (str): The expected answer string.
    Returns:
        bool: True if the answer is found in the solution, False otherwise.
    """
    return answer.strip() in solution

def run_geometry_benchmark(dataset_name="geoqa", sample_size=20, out_dir="geometry_results"):
    """
    Run a benchmark on geometry datasets using the G-LLaVA solver.

    Args:
        dataset_name (str): Name of the dataset to use.
        sample_size (int): Number of samples to evaluate.
        out_dir (str): Output directory for results CSV.
    Returns:
        list: List of result dictionaries for each problem.
    """
    ds = load_benchmark_dataset(dataset_name, sample_size=sample_size)
    solvers = register_solvers()
    gllava_solver = solvers["gllava_ollama"]  # or gllava_lmstudio
    results = []
    for ex in ds:
        problem, answer = get_problem_and_answer(ex, dataset_name)
        problem_input = {
            "text": problem,
            "image_path": ex.get("image_path")
        }
        solution = gllava_solver.solve(problem_input)
        is_correct = evaluate_solution(solution["solution"], answer)
        results.append({
            "problem": problem,
            "expected_answer": answer,
            "solution": solution["solution"],
            "is_correct": is_correct,
            "model_used": solution["model_used"]
        })
    os.makedirs(out_dir, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{out_dir}/{dataset_name}_gllava_results.csv", index=False)
    correct = sum(r["is_correct"] for r in results)
    print(f"Accuracy: {correct/len(results):.2%} ({correct}/{len(results)})")
    return results
