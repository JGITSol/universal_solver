import pytest
from adv_resolver_math.symbolic_engine import EnhancedSymbolicEngine, MathDomain

def test_algebra_zero_product_rule():
    engine = EnhancedSymbolicEngine(domain=MathDomain.ALGEBRA)
    engine.add_entity("prod1", {"type": "product", "value": 0})
    engine.infer()
    assert "zero_product" in engine.get_facts()

def test_geometry_triangle_sum_rule():
    engine = EnhancedSymbolicEngine(domain=MathDomain.GEOMETRY)
    engine.add_entity("tri1", {"type": "triangle"})
    engine.infer()
    assert "triangle_sum_180" in engine.get_facts()

def test_no_false_facts():
    engine = EnhancedSymbolicEngine(domain=MathDomain.ALGEBRA)
    engine.add_entity("prod2", {"type": "product", "value": 5})
    engine.infer()
    assert "zero_product" not in engine.get_facts()
