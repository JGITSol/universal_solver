import argparse
import json
from adv_resolver_math.ensemble_iterations.rstar_math_solver import RStarMathSolver
from adv_resolver_math.math_ensemble_adv_ms_hackaton import Agent
from adv_resolver_math.memory import MemoryManager


def main():
    parser = argparse.ArgumentParser(prog="usolve", description="Universal Solver CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # RStarMathSolver command
    parser_rstar = subparsers.add_parser("rstar", help="Run RStarMathSolver")
    parser_rstar.add_argument("problem", type=str, help="Mathematical problem to solve")
    # Configuration options
    parser_rstar.add_argument("--mcts-rounds", type=int, default=3, help="Number of MCTS rollout rounds")
    parser_rstar.add_argument("--evolution-rounds", type=int, default=2, help="Number of evolutionary iteration rounds")
    parser_rstar.add_argument("--weight-coherence", type=float, default=0.4, help="Weight for step coherence in reward")
    parser_rstar.add_argument("--weight-conceptual", type=float, default=0.3, help="Weight for conceptual consistency in reward")
    parser_rstar.add_argument("--weight-efficiency", type=float, default=0.3, help="Weight for computational efficiency in reward")

    args = parser.parse_args()

    if args.command == "rstar":
        # Initialize cache
        mem_mgr = MemoryManager()
        cached = mem_mgr.get(args.problem)
        if cached:
            print(json.dumps(cached, indent=2))
            return
        # Default agents; customize prompts and parameters as needed
        agents = [
            Agent("A", "phi4-mini:latest", "Default prompt A", 0.2, 1000),
            Agent("B", "phi4-mini:latest", "Default prompt B", 0.3, 1000)
        ]
        # Build custom reward weights
        reward_weights = {
            'step_coherence': args.weight_coherence,
            'conceptual_consistency': args.weight_conceptual,
            'computational_efficiency': args.weight_efficiency
        }
        solver = RStarMathSolver(
            agents=agents,
            mcts_rounds=args.mcts_rounds,
            evolution_rounds=args.evolution_rounds,
            reward_weights=reward_weights
        )
        result = solver.solve(args.problem)
        # Cache result
        mem_mgr.add(args.problem, result)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
