# Universal Solver Roadmap

For the latest and most detailed roadmap, see [`docs/project_roadmap.md`](docs/project_roadmap.md).

---

## Vision

Universal Solver aims to be the leading open-source platform for advanced mathematical problem solving, symbolic regression, and AI-driven research workflows—empowering researchers, educators, and developers with modular, extensible, and collaborative tools.

## Roadmap Overview

### Release Cadence

| Quarter (Target) | Primary Focus | Key Deliverables |
| ---------------- | ------------- | ---------------- |
| 2025&nbsp;Q4 | Hardening the math ensemble stack | Complete CI/CD modernization, ship v0.2.0 (solver orchestration refresh, GUI stability), publish benchmarking playbook |
| 2026&nbsp;Q1 | Distributed execution & observability | Remote inference support, structured logging pipeline, telemetry dashboards, beta Docker images |
| 2026&nbsp;Q2 | Community release & extensibility | Plugin SDK, documentation revamp, first community hackathon, v0.3.0 release |
| 2026&nbsp;H2 | Productization runway | Hosted demo environment, enterprise auth roadmap, scalable storage abstractions |

Progress is reviewed at the end of every quarter. Items that miss their target are re-estimated during the roadmap retro and assigned to the next feasible milestone.

### Core Platform & Architecture

- Modular, plug-and-play solvers and agents
- CLI, Python API, and GUI improvements
- Standardized configuration and environment management
- Logging, error handling, and reproducibility
- Distributed/cloud-native execution (mid-term)
- Plugin system for third-party solvers (mid-term)
- Web-based interactive UI/dashboard (long-term)

### AI Models & Integration

- Expand supported models (Ollama, Gemini, Llama 3, etc.)
- Improve prompt engineering and agent voting
- Remote/hybrid model execution (mid-term)
- External API integration (OpenAI, Google, HuggingFace)

### Symbolic Regression & Math Engines

- Enhance KAN and symbolic regression modules
- Integrate new math engines
- Unified benchmarking for all solvers

### Benchmarks, Testing, & Quality

- 100% code coverage for core logic
- Automated regression and boundary testing
- Standardized benchmarking and reporting

### Developer Experience & Documentation

- Improved guides, API docs, and onboarding
- Automated code formatting, linting, and type-checking

### Collaboration & Community

- Collaborative notebook workflows
- Community contribution guidelines
- Public model and benchmark sharing

---

For detailed milestones, timelines, and progress, refer to [`docs/project_roadmap.md`](docs/project_roadmap.md).
