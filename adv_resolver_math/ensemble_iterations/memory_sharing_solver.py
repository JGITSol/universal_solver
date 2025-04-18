import torch
import torch.nn as nn
import numpy as np
from typing import List
from adv_resolver_math.ensemble_iterations.enhanced_solver import EnhancedMathSolver, Solution, VotingResult

class SharedMemoryLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, 8, batch_first=True)
    def forward(self, agent_memories):
        # agent_memories: [batch, seq_len, d_model]
        pooled_memory, _ = self.attention(agent_memories, agent_memories, agent_memories)
        return pooled_memory

from dataclasses import dataclass, field

@dataclass
class MemorySharingMathSolver(EnhancedMathSolver):
    embedding_dim: int = 384
    """
    Extends EnhancedMathSolver with shared memory and knowledge distillation between agents.
    """
    def __post_init__(self):
        super().__post_init__()
        # Use embedding_dim if provided, else fallback to 128 for backward compatibility
        self.memory_dim = getattr(self, 'embedding_dim', 128)
        self.shared_memory_layer = SharedMemoryLayer(self.memory_dim)
        self.agent_memories = {name: torch.zeros(1, 1, self.memory_dim) for name in self.performance_stats}
    
    def aggregate_memories(self):
        # Aggregate all agents' memories
        memories = torch.cat([mem for mem in self.agent_memories.values()], dim=1)
        pooled = self.shared_memory_layer(memories)
        return pooled

    def update_memory(self, agent_name: str, embedding: np.ndarray):
        # Update agent's memory (ensure float32 dtype for torch compatibility)
        self.agent_memories[agent_name] = torch.tensor(embedding, dtype=torch.float32).reshape(1, 1, -1)

    def vote_on_solutions(self, solutions: List[Solution]) -> VotingResult:
        # Before voting, update memories with answer embeddings
        for s in solutions:
            self.update_memory(s.agent_name, self.embedder.encode([s.answer]))
        # Optionally aggregate memories for use in voting
        _ = self.aggregate_memories()
        return super().vote_on_solutions(solutions)
