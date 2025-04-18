import numpy as np
from adv_resolver_math.ensemble_iterations.memory_sharing_solver import MemorySharingMathSolver, Solution, VotingResult

class LatentReasoner:
    def __init__(self, base_model):
        self.base_model = base_model
        self.latent_cache = []
    def continuous_thought_step(self, problem_embedding):
        hidden_state = self.base_model.get_last_hidden_state(problem_embedding)
        self.latent_cache.append(hidden_state)
        return hidden_state
    def latent_to_text(self, hidden_state):
        return self.base_model.decode(hidden_state)

class CoconutAgent:
    def __init__(self, agent, base_model, reasoning_depth=3):
        self.agent = agent
        self.latent_reasoner = LatentReasoner(base_model)
        self.reasoning_depth = reasoning_depth
    def solve(self, problem):
        latent_state = self.embed_problem(problem)
        for _ in range(self.reasoning_depth):
            latent_state = self.latent_reasoner.continuous_thought_step(latent_state)
        return self.latent_reasoner.latent_to_text(latent_state)

class LatentSpaceMathSolver(MemorySharingMathSolver):
    """
    Solver using latent space reasoning (Coconut/Chain of Continuous Thought).
    """
    def latent_voting(self, solutions):
        # Vector-space voting using latent representations
        embeddings = [self.embedder.encode([s.answer])[0] for s in solutions]
        similarity_matrix = np.dot(embeddings, np.transpose(embeddings))
        consensus_idx = np.argmax(similarity_matrix.sum(axis=0))
        return solutions[consensus_idx]

    def vote_on_solutions(self, solutions):
        # Use latent voting for consensus
        return self.latent_voting(solutions)
