import pytest
import numpy as np
from adv_resolver_math.latent_reasoning import LatentReasoningModule

def test_embedding_shape():
    module = LatentReasoningModule(embedding_dim=8)
    vec = module.embed_object("x^2 + y^2")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (8,)

def test_similarity_self():
    module = LatentReasoningModule(embedding_dim=8)
    vec = module.embed_object("a+b")
    sim = module.similarity(vec, vec)
    assert abs(sim - 1.0) < 1e-6

def test_similarity_different():
    module = LatentReasoningModule(embedding_dim=8)
    v1 = module.embed_object("a+b")
    v2 = module.embed_object("x^2 + y^2")
    sim = module.similarity(v1, v2)
    assert 0.0 <= sim <= 1.0
