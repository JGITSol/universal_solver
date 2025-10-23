from pathlib import Path

from setuptools import find_packages, setup


def read_version() -> str:
    version_file = Path(__file__).resolve().parent / "VERSION"
    return version_file.read_text(encoding="utf-8").strip()


setup(
    name="universal_solver",
    version=read_version(),
    description=(
        "Modular, extensible platform for advanced mathematical problem solving, "
        "symbolic regression, and AI-driven research workflows."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Universal Solver contributors",
    author_email="",
    url="https://github.com/JGITSol/universal_solver",
    project_urls={
        "Bug Tracker": "https://github.com/JGITSol/universal_solver/issues",
        "Documentation": "https://github.com/JGITSol/universal_solver/tree/main/docs",
        "Source Code": "https://github.com/JGITSol/universal_solver",
        "CI/CD": "https://github.com/JGITSol/universal_solver/actions",
    },
    packages=find_packages(
        exclude=["tests*", "htmlcov*", "docs*", "project_guidelines*"]
    ),
    install_requires=[
        "numpy==1.26.4",
        "sympy==1.13.1",
        "scikit-learn==1.6.1",
        "sentence-transformers==3.4.1",
        "requests==2.32.3",
        "langchain>=0.3.0",
        "langchain-ollama>=0.3.0",
        "ollama>=0.1.5",
        "pykan==0.2.8",
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
        "matplotlib==3.8.4",
        "pandas==2.2.2",
        "openpyxl==3.1.2",
        "pyarrow==15.0.2",
        "seaborn==0.13.2",
        "plotly==5.22.0",
        "ydata-profiling==4.16.1",
        "memory_profiler==0.61.0",
        "jupyter==1.0.0",
        "nbconvert==7.16.0",
        "datasets==3.5.0",
        "python-dotenv==1.0.1",
        "rich==13.9.4",
        "asyncio==3.4.3",
        "typing-extensions>=4.13.2",
        "Pillow>=10.2.0,<11.0.0",
        "tenacity>=8.1.0,<9.0.0",
        "transformers==4.50.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.10.0",
            "isort>=5.12.0",
            "mypy>=1.6.0",
            "flake8>=6.1.0",
        ]
    },
    python_requires=">=3.8",
    entry_points={"console_scripts": ["usolve=adv_resolver_math.cli:main"]},
    include_package_data=True,
    classifiers=[
        # Development Status
        "Development Status :: 3 - Alpha",
        # Intended Audience
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        # Topic
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        # License
        "License :: OSI Approved :: MIT License",
        # Operating System
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        # Programming Language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        # Environment
        "Environment :: Console",
        "Environment :: GPU",
        # Framework
        "Framework :: Jupyter",
        # Natural Language
        "Natural Language :: English",
        # Typing
        "Typing :: Typed",
    ],
    keywords=[
        "mathematics",
        "solver",
        "symbolic-mathematics",
        "symbolic-regression",
        "ai",
        "machine-learning",
        "deep-learning",
        "neural-networks",
        "benchmark",
        "langchain",
        "ollama",
        "pytorch",
        "sympy",
        "kan",
        "kolmogorov-arnold-networks",
        "ensemble-learning",
        "math-problem-solving",
        "gsm8k",
        "math-dataset",
    ],
)
