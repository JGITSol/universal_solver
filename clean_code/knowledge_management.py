from typing import Any, Dict, List
import json
import os
from clean_code.logger import get_logger

logger = get_logger(__name__)

class KnowledgeManagementSystem:
    """
    Persistent knowledge management for storing problems, solutions, and heuristics.
    """
    def __init__(self, db_path="knowledge_db.json"):
        self.db_path = db_path
        self.knowledge = self._load_knowledge()

    def _load_knowledge(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    content = f.read().strip()
                    if not content:
                        return []
                    return json.loads(content)
            except (json.decoder.JSONDecodeError, OSError):
                return []
        return []

    def store_solution(self, problem: str, solution: Dict[str, Any]):
        logger.info(f"Storing solution for problem: {problem}")
        entry = {"problem": problem, "solution": solution}
        self.knowledge.append(entry)
        with open(self.db_path, "w") as f:
            json.dump(self.knowledge, f, indent=2)
        logger.info(f"Solution stored for problem: {problem}")

    def query(self, query_str: str) -> List[Dict[str, Any]]:
        logger.info(f"Querying knowledge for query string: {query_str}")
        results = [e for e in self.knowledge if query_str.lower() in e["problem"].lower()]
        logger.info(f"Found {len(results)} results for query string: {query_str}")
        return results
