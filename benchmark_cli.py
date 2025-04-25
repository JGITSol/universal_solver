"""
CLI for benchmarking solvers on industry-standard math datasets.
Supports integration with HuggingFace, Kaggle, Google Cloud, Azure, and Colab-ready workflows.
"""
import argparse
from benchmark_datasets import list_benchmark_datasets, load_benchmark_dataset, get_problem_and_answer
from showcase_advanced_math import agents, solvers
from adv_resolver_math.math_ensemble_adv_ms_hackaton import MathProblemSolver
import pandas as pd
from pathlib import Path
from datetime import datetime

def run_benchmark(dataset_name, sample_size, solver_names=None, out_dir="showcase_results"):
    ds = load_benchmark_dataset(dataset_name, sample_size=sample_size)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    rows = []
    for i, ex in enumerate(ds):
        problem, answer = get_problem_and_answer(ex, dataset_name)
        for solver_name, solver in solvers:
            if solver_names and solver_name not in solver_names:
                continue
            try:
                agent_solutions = [solver.get_solution(agent, problem) for agent in agents]
                result = solver.vote_on_solutions(agent_solutions)
                for sol in agent_solutions:
                    rows.append({
                        'solver': solver_name,
                        'problem': problem,
                        'dataset': dataset_name,
                        'agent': sol.agent_name,
                        'answer': sol.answer,
                        'reference': answer,
                        'confidence': sol.confidence,
                        'explanation': sol.explanation,
                        'i': i,
                        'timestamp': timestamp
                    })
                # Add consensus row
                rows.append({
                    'solver': solver_name,
                    'problem': problem,
                    'dataset': dataset_name,
                    'agent': 'CONSENSUS',
                    'answer': result.answer,
                    'reference': answer,
                    'confidence': result.confidence,
                    'explanation': '',
                    'i': i,
                    'timestamp': timestamp
                })
            except Exception as e:
                rows.append({'solver': solver_name, 'problem': problem, 'dataset': dataset_name, 'agent': 'ERROR', 'answer': str(e), 'reference': answer, 'confidence': 0.0, 'explanation': '', 'i': i, 'timestamp': timestamp})
    df = pd.DataFrame(rows)
    Path(out_dir).mkdir(exist_ok=True)
    out_path = Path(out_dir) / f"benchmark_{dataset_name}_{timestamp}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Benchmark complete. Results saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark solvers on math datasets.")
    parser.add_argument('--dataset', type=str, choices=list_benchmark_datasets(), required=True)
    parser.add_argument('--sample-size', type=int, default=10)
    parser.add_argument('--solvers', type=str, nargs='*', help='List of solvers to run (optional)')
    parser.add_argument('--out-dir', type=str, default="showcase_results")
    args = parser.parse_args()
    run_benchmark(args.dataset, args.sample_size, args.solvers, args.out_dir)

if __name__ == "__main__":
    main()
