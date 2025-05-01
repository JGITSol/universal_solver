import numpy as np

class LatentReasoningModule:
    """
    Minimal latent reasoning module for embedding mathematical objects and computing similarity.
    """
    def __init__(self, embedding_dim=8):
        self.embedding_dim = embedding_dim

    def embed_object(self, obj):
        # Dummy: use hash for reproducibility, map to fixed-size vector
        h = abs(hash(str(obj)))
        np.random.seed(h % (2**32))
        return np.random.rand(self.embedding_dim)

    def similarity(self, vec1, vec2):
        # Dummy: cosine similarity
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
