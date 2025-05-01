from enum import Enum
from typing import Any, Dict, Set, List

class MathDomain(Enum):
    GEOMETRY = "geometry"
    ALGEBRA = "algebra"
    NUMBER_THEORY = "number_theory"
    CALCULUS = "calculus"
    COMBINATORICS = "combinatorics"
    GENERAL = "general"

class EnhancedSymbolicEngine:
    """
    Symbolic reasoning engine supporting multiple mathematical domains.
    """
    def __init__(self, domain=MathDomain.GENERAL):
        self.domain = domain
        self.facts: Set[Any] = set()
        self.entities: Dict[str, Any] = {}
        self.rules: List[Any] = []
        self.knowledge_bases: Dict[MathDomain, Dict[str, Any]] = {
            MathDomain.GEOMETRY: self._init_geometry_kb(),
            MathDomain.ALGEBRA: self._init_algebra_kb(),
            MathDomain.NUMBER_THEORY: {},
            MathDomain.CALCULUS: {},
            MathDomain.COMBINATORICS: {},
            MathDomain.GENERAL: {}
        }
        self._configure_domain()

    def _configure_domain(self):
        if self.domain == MathDomain.GEOMETRY:
            self._register_geometry_rules()
        elif self.domain == MathDomain.ALGEBRA:
            self._register_algebra_rules()
        # Extend for other domains as needed

    def _init_geometry_kb(self):
        # Minimal geometry knowledge base
        return {"triangle_sum": "sum of angles in triangle is 180"}

    def _init_algebra_kb(self):
        # Minimal algebra knowledge base
        return {"zero_product": "if ab=0 then a=0 or b=0"}

    def _register_geometry_rules(self):
        def triangle_sum_rule(facts, entities):
            # Example: if triangle, add sum-of-angles fact
            for ent in entities.values():
                if ent.get("type") == "triangle":
                    facts.add("triangle_sum_180")
        self.rules.append(triangle_sum_rule)

    def _register_algebra_rules(self):
        def zero_product_rule(facts, entities):
            # Example: if product is zero, add zero-product fact
            for ent in entities.values():
                if ent.get("type") == "product" and ent.get("value") == 0:
                    facts.add("zero_product")
        self.rules.append(zero_product_rule)

    def add_entity(self, name: str, entity: Dict[str, Any]):
        self.entities[name] = entity

    def add_fact(self, fact: Any):
        self.facts.add(fact)

    def infer(self, max_iterations=10):
        for _ in range(max_iterations):
            facts_before = set(self.facts)
            for rule in self.rules:
                rule(self.facts, self.entities)
            if self.facts == facts_before:
                break

    def get_facts(self) -> Set[Any]:
        return set(self.facts)
