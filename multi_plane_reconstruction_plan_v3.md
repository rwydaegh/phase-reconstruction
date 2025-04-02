# Plan: Multi-Plane Phase Reconstruction Refactor (v3 - Animated Visualizations)

This document outlines the plan to refactor the phase reconstruction framework to support training and testing on distinct planes, with enhanced animated visualizations.

**Goal:** Enable defining multiple measurement planes, designating roles (train/test), run reconstruction, evaluate generalization, and visualize the process dynamically.

## Phase 1: Design & Configuration (Completed)

1.  [x] **Define New Configuration Structure (`conf/multi_plane_config.yaml`)**
    *   [x] Subtasks 1.1 - 1.8 (Includes `legacy_mode` flag)
    *   [x] **Subtask 1.9 (New):** Add optional `visualization` sub-section to each plane definition in `measurement_planes` list:
        *   `animate_comparison` (boolean, default: value of `use_test` for that plane).

2.  [x] **Design Data Flow & Structures**
    *   [x] Subtasks 2.1 - 2.3 (Mermaid diagram created)

## Phase 2: Core Implementation (Partially Completed)

1.  [x] **Unified Main Script (`multi_plane_reconstruction.py`)**
    *   [x] Subtasks 1.1, 1.3, 1.4, 1.5
    *   [x] Subtask 1.2 (Legacy mode delegation implemented based on config structure)

2.  [x] **Plane Processing Module (`src/plane_processing.py`)**
    *   [x] Subtasks 2.1, 2.3, 2.4
    *   [x] Subtask 2.2 (`load_real_plane` coordinate generation logic cleaned up).

3.  [x] **Source Point Cloud Handling (`src/source_handling.py`)**
    *   [x] Subtasks 3.1 - 3.4 (Includes loading, generation, tangents, perturbation).
    *   [x] Added current generation (`generate_original_currents`).
    *   [x] Added configurable source translation.

4.  [x] **Channel Matrix Handling (in `multi_plane_reconstruction.py`)**
    *   [x] Subtask 4.1 (Filtering train/test planes).
    *   [x] Subtask 4.2 (Training matrix creation logic implemented).
    *   [x] Subtask 4.3 (Model type handling).

5.  [x] **Gerchberg-Saxton Execution & History (`src/algorithms/gerchberg_saxton.py`, `multi_plane_reconstruction.py`)**
    *   [x] **Subtask 5.1 (Modify):** Update `holographic_phase_retrieval` signature to accept test plane info (coordinates, ground truth magnitudes). (Done - accepts `test_planes_data` dict with precomputed H)
    *   [x] **Subtask 5.2 (Modify):** Inside the GS loop (respecting `frame_skip`):
        *   [x] Calculate overall training RMSE.
        *   [x] Calculate per-training-plane RMSE.
        *   [x] Calculate reconstructed field on each test plane.
    *   [x] **Subtask 5.3 (Modify):** Change return value to be a comprehensive history structure (e.g., list of dicts per iteration) containing coefficients, train field segments, test fields, overall train RMSE, per-train-plane RMSEs. (Done - returns `full_history`)
    *   [x] *(Main Script)* Call `holographic_phase_retrieval`.
    *   [x] *(Main Script)* Update unpacking of the new history structure.

## Phase 3: Evaluation & Visualization (Revised)

1.  [x] **Evaluation Module (`src/evaluation.py`)**
    *   [x] Subtasks 1.1 - 1.8 (Calculates final metrics for test planes using final coefficients).

2.  [x] **Reporting (in `multi_plane_reconstruction.py`)**
    *   [x] Subtasks 2.1, 2.2 (Printing stats/metrics).
    *   [x] Subtask 2.3 (Saving results to pickle).

3.  [x] **New Visualization Module (`src/visualization/multi_plane_plots.py`)**
    *   [x] **Subtask 3.1:** Create `animate_convergence_errors` function (takes history, plots overall + per-train-plane RMSE vs iteration).
    *   [x] **Subtask 3.2:** Create `animate_layout_and_currents` function (takes history, points, planes; shows animated 3D rotating view of currents + plane boundaries).
    *   [x] **Subtask 3.3:** Create `animate_plane_comparison` function (takes history, list of planes to animate; generates one animation per plane showing True Mag vs Recon Mag vs Error over iterations).

4.  [x] **Visualization Integration (in `multi_plane_reconstruction.py`)**
    *   [x] **Subtask 4.1:** Import new visualization functions.
    *   [x] **Subtask 4.2:** Replace placeholder calls with calls to the new animation functions, passing the comprehensive history structure and relevant data. (`animate_convergence_errors`, `animate_layout_and_currents`, `animate_plane_comparison` integrated).
    *   [x] **Subtask 4.3:** Filter which planes get comparison animations based on the new `plane_cfg.visualization.animate_comparison` flag. (Logic added and confirmed).

## Phase 4: Testing & Refinement (Partially Completed)

1.  [x] **Unit Tests:** Basic tests created for new modules. Need updates for new functions/changes. (Reviewed existing tests; no immediate updates required for refactoring, new tests for visualization could be added).
2.  [~] **Integration Tests:** Create test configurations covering multi-plane scenarios. (Basic sim train/test config and test added).
3.  [x] **Legacy Mode Test:** Implement and test delegation. (Delegation logic implemented and tested successfully for both legacy config types).
4.  [x] **Refinement:** Addressed initial test failures and comparison discrepancies.

**Backward Compatibility Notes:** (Unchanged)
*   `legacy_mode` flag.
*   Core function reuse.
*   Helper function refactoring.
*   Avoid deleting legacy scripts immediately.
