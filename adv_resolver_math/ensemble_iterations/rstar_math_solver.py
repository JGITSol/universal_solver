import sympy
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from adv_resolver_math.ensemble_iterations.latent_space_solver import LatentSpaceMathSolver, Solution, VotingResult
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from adv_resolver_math.math_ensemble_adv_ms_hackaton import Agent, logger

class RStarMathSolver(LatentSpaceMathSolver):
    """
    Solver with r*-math: symbolic/code verification, process reward, and MCTS-inspired exploration.
    """
    def __init__(self, *args, mcts_rounds: int = 3, evolution_rounds: int = 2, reward_weights: Dict[str, float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.verification_cache = {}
        # dynamic configuration
        self.mcts_rounds = mcts_rounds
        self.evolution_rounds = evolution_rounds
        # allow custom reward weights or defaults
        self.process_reward_model = reward_weights or self.init_reward_model()
        self.mcts_tree = {}

    def init_reward_model(self):
        return {
            'step_coherence': 0.4,
            'conceptual_consistency': 0.3,
            'computational_efficiency': 0.3
        }

    def symbolic_verification(self, step: str) -> bool:
        try:
            expr = sympy.parse_expr(step.split('=')[-1].split('→')[-1].strip())
            simplified = sympy.simplify(expr)
            return simplified.equals(sympy.true)
        except:
            return False

    def code_verification(self, problem: str, solution: Solution) -> float:
        verification_score = 0.0
        steps = solution.explanation.split('\n')
        with ThreadPoolExecutor() as executor:
            futures = []
            for step in steps:
                if '=' in step or '→' in step:
                    futures.append(executor.submit(self.verify_single_step, problem, step.strip()))
            results = [f.result() for f in futures]
            verification_score = np.mean(results) if results else 0.0
        return verification_score

    def verify_single_step(self, problem: str, step: str) -> float:
        try:
            symbolic_valid = self.symbolic_verification(step)
            # Optionally execute code for more robust check (not implemented for safety)
            return 1.0 if symbolic_valid else 0.5
        except:
            return 0.0

    def calculate_process_reward(self, solution: Solution) -> float:
        # Dummy: reward based on explanation length and confidence
        reward = (len(solution.explanation.split()) / 100) * self.process_reward_model['step_coherence']
        reward += solution.confidence * self.process_reward_model['conceptual_consistency']
        return min(1.0, reward)

    def analyze_step_coherence(self, steps: List[str]) -> float:
        """Evaluate logical flow between steps."""
        embeddings = [self.embedder.encode(step) for step in steps]
        scores = []
        for i in range(1, len(embeddings)):
            scores.append(cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0])
        return float(np.mean(scores)) if scores else 0.0

    def analyze_conceptual_consistency(self, solution: Solution) -> float:
        """Evaluate conceptual alignment with problem type."""
        concepts = {
            'algebra': ['variable', 'equation', 'solve'],
            'calculus': ['derivative', 'integral', 'limit'],
            'geometry': ['angle', 'area', 'volume']
        }
        exp = solution.explanation.lower()
        detected = [c for c, kws in concepts.items() if any(kw in exp for kw in kws)]
        return len(set(detected)) / len(concepts)

    def analyze_computational_efficiency(self, steps: List[str]) -> float:
        """Evaluate solution path optimality."""
        if any('nested' in step.lower() for step in steps):
            return 0.6
        if any('loop' in step.lower() for step in steps):
            return 0.8
        return 1.0

    def mcts_rollout(self, agent: Agent, problem: str) -> Solution:
        """Monte Carlo Tree Search enhanced solution generation."""
        best_sol = None
        best_score = -np.inf
        for _ in range(self.mcts_rounds):
            sol = self.get_solution(agent, problem)
            vs = self.code_verification(problem, sol)
            pr = self.calculate_process_reward(sol)
            score = 0.7 * vs + 0.3 * pr + 0.1 * sol.confidence
            if score > best_score:
                best_score = score
                best_sol = sol
        return best_sol

    def refine_with_feedback(self, solution: Solution, problem: str) -> Solution:
        """Generate a refined solution using process feedback."""
        agent = next((a for a in self.agents if a.name == solution.agent_name), None)
        if agent:
            return self.get_solution(agent, problem, previous_solutions=[solution])
        return solution

    def verification_aware_vote(self, solutions: List[Solution]) -> VotingResult:
        """Vote weighting by combined verification and process scores."""
        groups: Dict[str, Dict[str, Any]] = {}
        for sol in solutions:
            key = sol.answer
            vs = getattr(sol, 'verification_score', 0.0)
            pr = getattr(sol, 'process_reward', 0.0)
            weight = vs + pr
            if key not in groups:
                groups[key] = {'solutions': [], 'weight': 0.0}
            groups[key]['solutions'].append(sol)
            groups[key]['weight'] += weight
        best_ans, best_grp = max(groups.items(), key=lambda x: x[1]['weight'])
        agents = [s.agent_name for s in best_grp['solutions']]
        confidence = best_grp['weight'] / (len(solutions) * 2)
        return VotingResult(answer=best_ans, confidence=confidence, agents_in_agreement=agents)

    def solve(self, problem: str):
        """r*-math enhanced solve with MCTS, evolution, and final vote."""
        logger.info(f"Starting r*-math solve for: {problem}")
        # Stage 1: MCTS rollouts
        sols = []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.mcts_rollout, ag, problem): ag for ag in self.agents}
            for fut in futures:
                sols.append(fut.result())
        # Stage 2: Score solutions
        for sol in sols:
            sol.verification_score = self.code_verification(problem, sol)
            sol.process_reward = self.calculate_process_reward(sol)
            sol.confidence = 0.6 * sol.verification_score + 0.4 * sol.process_reward
        # Stage 3: Evolutionary iterations (customizable)
        survivors = sols
        for _ in range(self.evolution_rounds):
            refined = [self.refine_with_feedback(sol, problem) for sol in survivors]
            for sol in refined:
                sol.verification_score = self.code_verification(problem, sol)
                sol.process_reward = self.calculate_process_reward(sol)
                sol.confidence = 0.6 * sol.verification_score + 0.4 * sol.process_reward
            survivors = sorted(refined, key=lambda s: s.confidence, reverse=True)[:max(1, len(self.agents)//2)]
        # Stage 4: Final vote
        final = self.verification_aware_vote(survivors)
        # Prepare result
        result = {'solutions': []}
        for sol in survivors:
            result['solutions'].append({
                'agent_name': sol.agent_name,
                'answer': sol.answer,
                'explanation': sol.explanation,
                'confidence': sol.confidence,
                'verification_score': sol.verification_score,
                'process_reward': sol.process_reward
            })
        result['final_answer'] = final.answer
        result['final_confidence'] = final.confidence
        result['supporting_agents'] = final.agents_in_agreement
        return result
