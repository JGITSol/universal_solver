import re
import numpy as np
from typing import List, Any
from dataclasses import dataclass, field
from adv_resolver_math.math_ensemble_adv_ms_hackaton import MathProblemSolver, Agent, Solution, VotingResult
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class EnhancedMathSolver(MathProblemSolver):
    """
    Advanced solver with semantic clustering, performance-weighted voting, and modular SOTA features.
    Inherits from MathProblemSolver as the working base.
    """
    embedding_dim: int = field(default=384)

    def __post_init__(self):
        super().__post_init__()
        # Semantic embedder for answer similarity
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        # Track agent performance (dummy init, extend as needed)
        self.performance_stats = {agent.name: {"correct": 1, "total": 1} for agent in self.agents}

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using sentence embeddings."""
        embeddings = self.embedder.encode([text1, text2])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    def vote_on_solutions(self, solutions: List[Solution]) -> VotingResult:
        """
        Enhanced voting with semantic clustering and agent performance weighting.
        """
        # Cluster solutions by semantic similarity
        solution_groups = []
        for solution in solutions:
            matched = False
            for group in solution_groups:
                avg_sim = np.mean([
                    self._calculate_semantic_similarity(solution.answer, s.answer)
                    for s in group
                ])
                if avg_sim > 0.85:
                    group.append(solution)
                    matched = True
                    break
            if not matched:
                solution_groups.append([solution])

        # Weighted scores by confidence and historical performance
        group_scores = []
        for group in solution_groups:
            score = sum(
                s.confidence * (self.performance_stats[s.agent_name]["correct"] /
                                max(1, self.performance_stats[s.agent_name]["total"]))
                for s in group
            )
            group_scores.append((score, group))

        # Select best group
        best_group = max(group_scores, key=lambda x: x[0])[1] if group_scores else []

        # Prepare result
        return VotingResult(
            answer=self._select_representative_answer(best_group) if best_group else "No consensus",
            confidence=len(best_group)/len(self.agents) if self.agents else 0.0,
            agents_in_agreement=[s.agent_name for s in best_group]
        )

    def _select_representative_answer(self, group: List[Solution]) -> str:
        """Choose the most common answer in the group (fallback: first)."""
        if not group:
            return "No consensus"
        answers = [s.answer for s in group]
        return max(set(answers), key=answers.count)
