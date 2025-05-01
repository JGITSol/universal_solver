import threading
from typing import Callable, Any, Set, List
import time

class EnhancedSKESTSearch:
    """
    Parallel ensemble search with shared knowledge synchronization.
    Each thread runs its own symbolic engine, periodically sharing facts.
    """
    def __init__(self, engine_factory: Callable[[], Any], num_threads: int = 2, max_iterations: int = 5):
        self.engine_factory = engine_factory
        self.num_threads = num_threads
        self.max_iterations = max_iterations
        self.shared_knowledge: Set[Any] = set()
        self.shared_knowledge_lock = threading.Lock()
        self.threads: List[threading.Thread] = []
        self.finished = threading.Event()

    def _search_thread(self, thread_id: int):
        engine = self.engine_factory()
        for _ in range(self.max_iterations):
            # Synchronize shared knowledge into local engine
            with self.shared_knowledge_lock:
                for fact in self.shared_knowledge:
                    engine.add_fact(fact)
            engine.infer(1)  # One inference step per iteration
            # Share new facts with ensemble
            with self.shared_knowledge_lock:
                before = set(self.shared_knowledge)
                self.shared_knowledge.update(engine.get_facts())
                after = self.shared_knowledge
            time.sleep(0.01)  # Simulate work, avoid busy-wait
        if thread_id == 0:
            self.finished.set()

    def run_search(self):
        self.threads = [threading.Thread(target=self._search_thread, args=(i,)) for i in range(self.num_threads)]
        for t in self.threads:
            t.start()
        self.finished.wait(timeout=2)
        for t in self.threads:
            t.join(timeout=2)

    def get_shared_knowledge(self) -> Set[Any]:
        return set(self.shared_knowledge)
