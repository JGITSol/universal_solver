import pytest
from adv_resolver_math.symbolic_engine import EnhancedSymbolicEngine, MathDomain
from adv_resolver_math.ensemble_skest_search import EnhancedSKESTSearch

def engine_factory_with_fact():
    engine = EnhancedSymbolicEngine(domain=MathDomain.ALGEBRA)
    engine.add_entity("prod1", {"type": "product", "value": 0})
    return engine

def test_parallel_search_fact_sharing():
    # Both threads should share the zero_product fact
    search = EnhancedSKESTSearch(engine_factory=engine_factory_with_fact, num_threads=2, max_iterations=3)
    search.run_search()
    shared_facts = search.get_shared_knowledge()
    assert "zero_product" in shared_facts

def engine_factory_no_fact():
    return EnhancedSymbolicEngine(domain=MathDomain.ALGEBRA)

def test_parallel_search_no_false_facts():
    # If no entity triggers a rule, no fact should be shared
    search = EnhancedSKESTSearch(engine_factory=engine_factory_no_fact, num_threads=2, max_iterations=3)
    search.run_search()
    shared_facts = search.get_shared_knowledge()
    assert "zero_product" not in shared_facts
