from setuptools import setup, find_packages

setup(
    name="universal_solver",
    version="0.1.0",
    description="Modular, extensible platform for advanced mathematical problem solving, symbolic regression, and AI-driven research workflows.",
    author="Your Name",
    packages=find_packages(exclude=["tests*", "htmlcov*", "docs*", "project_guidelines*"]),
    install_requires=[
        "numpy==1.26.4",
        "sympy==1.13.1",
        "scikit-learn==1.6.1",
        "sentence-transformers==3.4.1",
        "requests==2.32.3",
        "langchain==0.1.0",
        "langchain-core>=0.1.7,<0.2",
        "langchain-community==0.0.10",
        # "langchain-ollama==0.1.1",  # Temporarily removed due to incompatible langchain-core requirements
        # "ollama==0.1.5",
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
        "typing-extensions==4.13.1",
        "pillow==9.0.0",
        "tenacity>=8.1.0,<9.0.0",
        "transformers==4.50.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.10.0",
            "isort>=5.12.0",
            "mypy>=1.6.0",
            "flake8>=6.1.0"
        ]
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "usolve=adv_resolver_math.cli:main"
        ]
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

