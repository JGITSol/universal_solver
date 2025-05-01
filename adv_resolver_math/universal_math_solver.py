from adv_resolver_math.symbolic_engine import EnhancedSymbolicEngine, MathDomain
from adv_resolver_math.ensemble_skest_search import EnhancedSKESTSearch
from adv_resolver_math.proof_assistant_interface import ProofAssistantInterface
from adv_resolver_math.latent_reasoning import LatentReasoningModule

class UniversalMathSolver:
    """
    Minimal orchestrator integrating symbolic engine, ensemble search, proof assistant, and latent reasoning.
    """
    def __init__(self, domain=MathDomain.ALGEBRA):
        self.domain = domain
        self.symbolic_engine = EnhancedSymbolicEngine(domain=domain)
        self.latent_module = LatentReasoningModule()
        self.proof_assistant = ProofAssistantInterface()
        self._entity = None
        def engine_factory_with_entity():
            engine = EnhancedSymbolicEngine(domain=domain)
            if self._entity is not None:
                engine.add_entity("main", self._entity)
            return engine
        self.ensemble_search = EnhancedSKESTSearch(
            engine_factory=engine_factory_with_entity,
            num_threads=2,
            max_iterations=3
        )

    def solve(self, problem_text, entity=None):
        # Store entity for use in all engines
        self._entity = entity
        if entity:
            self.symbolic_engine.add_entity("main", entity)
        self.symbolic_engine.infer()
        # Run ensemble search with entity-aware engines
        self.ensemble_search.run_search()
        return {
            "facts": self.symbolic_engine.get_facts(),
            "ensemble_facts": self.ensemble_search.get_shared_knowledge()
        }
