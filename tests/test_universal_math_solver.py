import pytest
from adv_resolver_math.universal_math_solver import UniversalMathSolver
from adv_resolver_math.symbolic_engine import MathDomain

def test_solve_algebra_zero_product():
    solver = UniversalMathSolver(domain=MathDomain.ALGEBRA)
    entity = {"type": "product", "value": 0}
    result = solver.solve("If ab=0, what can you say about a or b?", entity=entity)
    assert "zero_product" in result["facts"]
    assert "zero_product" in result["ensemble_facts"]

def test_solve_geometry_triangle_sum():
    solver = UniversalMathSolver(domain=MathDomain.GEOMETRY)
    entity = {"type": "triangle"}
    result = solver.solve("What is the sum of angles in a triangle?", entity=entity)
    assert "triangle_sum_180" in result["facts"]
    assert "triangle_sum_180" in result["ensemble_facts"]
