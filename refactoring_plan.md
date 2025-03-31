# Refactoring Plan for Phase Reconstruction Framework

This document outlines a plan to refactor the phase reconstruction codebase for improved long-term scalability, maintainability, and testability.

## Goals

*   Enhance modularity and separation of concerns.
*   Standardize configuration management.
*   Implement robust data handling and I/O.
*   Systematize parallelization for analysis tasks.
*   Introduce a comprehensive testing suite.
*   Decouple visualization from core computation.
*   Enforce consistent code style and documentation standards.

## Refactoring Actions

1.  **Enhance Modularity & Separation of Concerns:**
    *   **Action:** Break down large files (`holographic_phase_retrieval.py`, `perturbation_analysis.py`, `visualization.py`, etc.) into smaller, focused modules (e.g., `src/algorithms/`, `src/perturbations/`, `src/simulation_runner.py`, `src/metrics.py`). Move utilities to appropriate locations.
    *   **Benefit:** Improves organization, testability, reuse, and targeted optimization.

2.  **Standardize Configuration Management:**
    *   **Action:** Adopt a robust configuration library (e.g., Hydra, Pydantic+YAML/JSON). Centralize configuration loading and management. Define base configurations with easy overrides.
    *   **Benefit:** Simplifies experiment setup, enhances reproducibility, scales better for complex parameter spaces.

3.  **Implement Robust Data Handling & I/O:**
    *   **Action:** Create a dedicated `src/io.py` module. Standardize on efficient formats (e.g., HDF5, NetCDF) for large numerical data. Implement consistent output naming conventions.
    *   **Benefit:** Improves data management, interoperability, supports large datasets, ensures consistency.

4.  **Systematize Parallelization:**
    *   **Action:** Refactor analysis scripts (`sensitivity_analysis/`, etc.) to use `multiprocessing`, `joblib`, or `dask` for distributing independent simulation runs. Ensure core simulation functions are parallel-safe.
    *   **Benefit:** Speeds up parameter sweeps and analyses, crucial for exploring large design spaces.

5.  **Introduce Comprehensive Testing:**
    *   **Action:** Set up `pytest`. Write unit tests for core functions and integration tests for key workflows. Use small, deterministic test data. Integrate testing into the development workflow.
    *   **Benefit:** Catches regressions, validates correctness, builds confidence, essential for scalability and collaboration.

6.  **Decouple Visualization:**
    *   **Action:** Refactor visualization functions to operate on saved output data or well-defined data structures. Minimize plotting calls within core computational loops. Consider separate visualization scripts/tools.
    *   **Benefit:** Enables headless runs, makes visualization code independent, prevents visualization bottlenecks.

7.  **Enforce Code Style and Documentation Standards:**
    *   **Action:** Adopt and enforce code style (`black`, `flake8`/`ruff`). Add comprehensive docstrings (NumPy/Google style). Consider Sphinx for HTML documentation generation. Keep documentation updated.
    *   **Benefit:** Improves readability, maintainability, and developer onboarding.

## Suggested Execution Order

1.  Setup Testing Framework & Linters (Actions 5 & 7 foundations).
2.  Start Modularization (Action 1), applying tests incrementally.
3.  Implement Centralized Configuration (Action 2).
4.  Refactor Data I/O (Action 3).
5.  Decouple Visualization (Action 6).
6.  Integrate Systematic Parallelization (Action 4).
7.  Continuously improve Documentation and Test Coverage (Actions 5 & 7).