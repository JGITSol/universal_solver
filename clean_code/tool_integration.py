import numpy as np
import sympy as sp
from typing import Any, Dict
from clean_code.logger import get_logger

logger = get_logger(__name__)

class ToolIntegrationManager:
    """
    Manages integration with external mathematical tools and libraries.
    """
    def __init__(self):
        self.tools = {
            "sympy": self.sympy_tool,
            "numpy": self.numpy_tool
        }
        logger.info("ToolIntegrationManager initialized.")

    def call_tool(self, tool_name: str, function_name: str, *args, **kwargs) -> Any:
        if tool_name in self.tools:
            logger.info(f"Calling tool: {tool_name}")
            return self.tools[tool_name](function_name, *args, **kwargs)
        else:
            logger.error(f"Tool {tool_name} not found.")
            raise ValueError(f"Tool {tool_name} not found.")

    def sympy_tool(self, function_name: str, *args, **kwargs) -> Any:
        logger.info(f"Calling sympy function: {function_name}")
        func = getattr(sp, function_name, None)
        if func:
            return func(*args, **kwargs)
        else:
            logger.error(f"Sympy function {function_name} not found.")
            raise ValueError(f"Sympy function {function_name} not found.")

    def numpy_tool(self, function_name: str, *args, **kwargs) -> Any:
        logger.info(f"Calling numpy function: {function_name}")
        func = getattr(np, function_name, None)
        if func:
            return func(*args, **kwargs)
        else:
            logger.error(f"NumPy function {function_name} not found.")
            raise ValueError(f"NumPy function {function_name} not found.")
