from dataclasses import dataclass
from typing import List
import numpy as np
import torch
import torch.nn as nn

from adv_resolver_math.ensemble_iterations.enhanced_solver import (
    EnhancedMathSolver,
    Solution,
    VotingResult,
)


class SharedMemoryLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, 8, batch_first=True)

    def forward(self, agent_memories):
        # agent_memories: [batch, seq_len, d_model]
        pooled_memory, _ = self.attention(
            agent_memories,
            agent_memories,
            agent_memories,
        )
        return pooled_memory


@dataclass
class MemorySharingMathSolver(EnhancedMathSolver):
    embedding_dim: int = 384
    """
    Extends EnhancedMathSolver with shared memory i knowledge distillation między agentami.
    """

    def __post_init__(self):
        super().__post_init__()
        # Użyj embedding_dim jeśli podany, w przeciwnym razie 128 dla kompatybilności wstecznej
        self.memory_dim = getattr(
            self,
            "embedding_dim",
            128,
        )
        self.shared_memory_layer = SharedMemoryLayer(self.memory_dim)
        self.agent_memories = {}
        for name in self.performance_stats:
            zeros_shape = (1, 1, self.memory_dim)
            self.agent_memories[name] = torch.zeros(
                *zeros_shape
            )

    def aggregate_memories(self):
        # Agreguj pamięci wszystkich agentów
        agent_memory_list = [mem for mem in self.agent_memories.values()]
        cat_dim = 1
        memories = torch.cat(
            agent_memory_list,
            dim=cat_dim
        )
        pooled = self.shared_memory_layer(memories)
        return pooled

    def update_memory(self, agent_name: str, embedding: np.ndarray):
        # Aktualizuj pamięć agenta (float32 dla torch)
        self.agent_memories[agent_name] = (
            torch.tensor(
                embedding,
                dtype=torch.float32,
            ).reshape(
                1,
                1,
                -1,
            )
        )

    def vote_on_solutions(self, solutions: List[Solution]) -> VotingResult:
        # Przed głosowaniem aktualizuj pamięci embeddingami odpowiedzi
        for s in solutions:
            self.update_memory(s.agent_name, self.embedder.encode([s.answer]))
        # Opcjonalnie agreguj pamięci do głosowania
        _ = self.aggregate_memories()
        return super().vote_on_solutions(solutions)
