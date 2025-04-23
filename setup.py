from setuptools import setup, find_packages

setup(
    name="universal_solver",
    version="0.1.0",
    packages=find_packages(exclude=["tests*", "htmlcov*", "docs*", "project_guidelines*"]),
    install_requires=[
        "numpy",
        "sympy",
        "scikit-learn",
        "sentence-transformers",
        "requests",
        "langchain_ollama",
        "tenacity"
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "usolve=adv_resolver_math.cli:main"
        ]
    }
}
