# Plan: Multi-Plane Phase Reconstruction Refactor

This document outlines the plan to refactor the phase reconstruction framework to support training and testing on distinct, user-defined measurement planes.

**Goal:** Enable defining multiple measurement planes (real or simulated) in the configuration, designating each for training (optimization) and/or testing (evaluation), allowing for more robust validation and analysis (e.g., testing generalization across different planes).

## Phase 1: Design & Configuration

1.  [x] **Define New Configuration Structure (`conf/multi_plane_config.yaml`)**
    *   [x] **Subtask 1.1:** Define the top-level structure (e.g., `global_params`, `source_pointcloud`, `measurement_planes`).
    *   [x] **Subtask 1.2:** Define `global_params` section (wavelength, GS settings, model type, etc.).
    *   [x] **Subtask 1.3:** Define `source_pointcloud` section (path, perturbation settings).
    *   [x] **Subtask 1.4:** Define the `measurement_planes` list structure. Each element should be a dictionary representing a plane.
    *   [x] **Subtask 1.5:** Define common plane parameters: `name` (string, unique), `is_real_plane` (bool), `use_train` (bool), `use_test` (bool), `translation` (list `[x, y, z]`).
    *   [x] **Subtask 1.6:** Define parameters specific to `is_real_plane: true`: `measured_data_path` (string), `target_resolution` (int).
    *   [x] **Subtask 1.7:** Define parameters specific to `is_real_plane: false`: `plane_type` (e.g., 'xy', 'yz', 'xz'), `center` (list `[x, y, z]`), `size` (list `[w, h]`), `resolution` (int).
    *   [x] **Subtask 1.8:** Add a `legacy_mode` flag (boolean, default `false`) at the top level. If `true`, the system should attempt to run using the old single-plane config structure (`measured_data.yaml` or `simulated_data.yaml`).

2.  [x] **Design Data Flow & Structures**
    *   [x] **Subtask 2.1:** Define internal data structures to hold processed plane information (coordinates, magnitudes, flags, original data reference).
    *   [x] **Subtask 2.2:** Map the flow: Config -> Plane Processing -> Channel Matrix Creation (Train) -> GS Algo -> Channel Matrix Creation (Test) -> Field Calculation (Test) -> Evaluation.
    *   [x] **Subtask 2.3:** Create a Mermaid diagram illustrating the data flow (included below).

```mermaid
graph TD
    A[Load Config] --> A1{Legacy Mode?};
    A1 -- Yes --> B_Legacy[Load Legacy Config];
    A1 -- No --> B_Multi[Load Multi-Plane Config];

    B_Multi --> C[Load/Gen Source Points];
    B_Multi --> D[Process Measurement Planes List];
    subgraph Process Planes
        direction TB
        D1{For Each Plane Def} --> D2{Load/Gen Coords};
        D2 --> D3[Sample/Translate];
        D3 --> D4[Store Processed Plane Data];
    end
    C --> E;
    D --> E{Separate Train/Test Planes};

    subgraph Training Path
        direction LR
        E -- Train --> F(Create H_plane);
        F --> G(Stack H_train);
        E -- Train --> H(Get/Gen Mag_plane);
        H --> I(Stack Mag_train);
        G & I --> J[Run Gerchberg-Saxton];
        J --> K(Final Coefficients);
    end

    subgraph Testing Path
        direction LR
        E -- Test --> L(Store Test Plane Info);
        K --> M{For Each Test Plane};
        L --> M;
        M --> N(Get Ground Truth Mag);
        M --> O(Create H_test_plane);
        O --> P(Calc Recon Field);
        N & P --> Q(Calculate Metrics);
        Q --> R(Collect Test Results);
    end

    R --> S[Report & Visualize];

    B_Legacy --> T[Run Legacy Workflow (Existing Scripts)];

    style J fill:#f9f,stroke:#333,stroke-width:2px
    style P fill:#ccf,stroke:#333,stroke-width:2px
    style Q fill:#ccf,stroke:#333,stroke-width:2px
```

## Phase 2: Core Implementation

1.  [x] **Unified Main Script (`multi_plane_reconstruction.py`)**
    *   [x] **Subtask 1.1:** Set up Hydra integration to load the new config (`multi_plane_config.yaml`).
    *   [~] **Subtask 1.2:** Implement logic to check the `legacy_mode` flag. If true, delegate to existing `measured_data_reconstruction.py` or `simulated_data_reconstruction.py` (requires minor modifications to them to be callable). (Placeholder exists)
    *   [x] **Subtask 1.3:** If `legacy_mode` is false, orchestrate the new multi-plane workflow by calling subsequent modules/functions.
    *   [x] **Subtask 1.4:** Handle random seed setup.
    *   [x] **Subtask 1.5:** Manage output directory creation.

2.  [x] **Plane Processing Module (`src/plane_processing.py`)**
    *   [x] **Subtask 2.1:** Create `process_plane_definitions(plane_configs)` function: Iterates through plane definitions from the config.
    *   [x] **Subtask 2.2:** Create `load_real_plane(config)`: Loads data using `src/io.load_measurement_data`, handles sampling via `sample_measurement_data` (refactor/move from `measured_data_reconstruction.py`), applies translation. Returns processed data structure. (Dummy plane creation logic remains)
    *   [x] **Subtask 2.3:** Create `generate_simulated_plane(config)`: Generates coordinates based on `plane_type`, `center`, `size`, `resolution`. Applies translation. Returns processed data structure.
    *   [x] **Subtask 2.4:** Ensure `src/io.py` functions (`load_measurement_data`, `sample_measurement_data`) are robust and reusable.

3.  [x] **Source Point Cloud Handling (`src/source_handling.py` or similar)**
    *   [x] **Subtask 3.1:** Create `get_source_pointcloud(config)` function.
    *   [x] **Subtask 3.2:** Encapsulate logic for loading from file (`config.source_pointcloud.path`) or generating a test cloud (using `create_test_pointcloud`).
    *   [x] **Subtask 3.3:** Encapsulate tangent calculation/loading logic (using `preprocess_pointcloud.get_tangent_vectors`).
    *   [x] **Subtask 3.4:** Encapsulate point perturbation logic based on `config.source_pointcloud.perturb_points` and `perturbation_factor`. Return both `points_true` and `points_perturbed` (and corresponding tangents).
    *   [x] *(Implicit)* Generate original currents.

4.  [x] **Channel Matrix Handling (in `multi_plane_reconstruction.py` or dedicated module)**
    *   [x] **Subtask 4.1:** Filter processed planes into `train_planes` and `test_planes` lists based on `use_train`/`use_test` flags.
    *   [x] **Subtask 4.2:** Implement training matrix creation logic (integrated into `run_multi_plane_workflow`):
        *   [x] Iterate through `train_planes`.
        *   [x] Call `create_channel_matrix` for each plane using `points_perturbed`.
        *   [x] Stack the resulting `H_plane` matrices vertically into `H_train`.
        *   [x] Extract/generate and stack corresponding magnitudes into `Mag_train`.
        *   [x] *(Implicit)* Return `H_train`, `Mag_train`.
    *   [x] **Subtask 4.3:** Ensure `create_channel_matrix` handles vector/scalar model based on global config.

5.  [x] **Gerchberg-Saxton Execution (in `multi_plane_reconstruction.py`)**
    *   [x] **Subtask 5.1:** Call `holographic_phase_retrieval` with `H_train`, `Mag_train`, and relevant global config parameters.
    *   [x] **Subtask 5.2:** Retrieve `final_coefficients` and `stats`.

## Phase 3: Evaluation & Visualization

1.  [x] **Evaluation Module (`src/evaluation.py`)**
    *   [x] **Subtask 1.1:** Create `evaluate_on_test_planes(...)` function.
    *   [x] **Subtask 1.2:** Inside the function, iterate through `test_planes`.
    *   [x] **Subtask 1.3:** For each test plane, get its ground truth magnitude (load if real, compute if simulated using `points_true`, `tangents_true`, and `original_currents`).
    *   [x] **Subtask 1.4:** Create the specific `H_test_plane` using `points_perturbed` and the test plane's coordinates.
    *   [x] **Subtask 1.5:** Calculate `recon_field_test = H_test_plane @ final_coefficients`.
    *   [x] **Subtask 1.6:** Compute metrics (RMSE, Corr) using `src.utils.normalized_rmse` and `src.utils.normalized_correlation` comparing `abs(recon_field_test)` to ground truth magnitude.
    *   [x] **Subtask 1.7:** Store results per test plane (name, metrics, recon_field, true_field).
    *   [x] **Subtask 1.8:** Return aggregated results.

2.  [x] **Reporting (in `multi_plane_reconstruction.py`)**
    *   [x] **Subtask 2.1:** Print summary of GS stats.
    *   [x] **Subtask 2.2:** Print evaluation metrics for each test plane.
    *   [x] **Subtask 2.3:** Save aggregated results (config, stats, evaluation metrics) to a file (e.g., pickle or JSON).

3.  [ ] **Visualization Updates (`src/visualization/`)**
    *   [ ] **Subtask 3.1:** Modify `visualize_fields`: Adapt to potentially display multiple test plane results. Maybe select one plane to show or create subplots. (Placeholder exists)
    *   [ ] **Subtask 3.2:** Modify `visualize_iteration_history`: Ensure it uses the combined `Mag_train` for error calculation if displayed. (Placeholder exists)
    *   [ ] **Subtask 3.3:** Modify `visualize_current_and_field_history`: Adapt comparison logic if multiple test planes exist. (Placeholder exists)
    *   [ ] **Subtask 3.4:** Consider adding a new visualization showing the 3D layout of all defined measurement planes and the source point cloud.

## Phase 4: Testing & Refinement

1.  [~] **Unit Tests:** Add/update unit tests for new modules (`plane_processing`, `evaluation`, `source_handling`). (Basic tests created and passing)
2.  [ ] **Integration Tests:** Create test configurations (`multi_plane_config.yaml`) covering different scenarios (real/simulated planes, train/test combinations, vector/scalar models).
3.  [ ] **Legacy Mode Test:** Ensure `legacy_mode: true` correctly runs the old workflows.
4.  [x] **Refinement:** Address any issues found during testing. (Fixed initial test failures)

**Backward Compatibility Notes:**

*   The `legacy_mode` flag provides an explicit way to run old configurations.
*   Core functions like `create_channel_matrix` and `holographic_phase_retrieval` are largely reused, minimizing disruption.
*   Refactoring helper functions (like `sample_measurement_data`) into common modules (`src/io.py`, `src/utils/`) improves maintainability for both old and new workflows.
*   Avoid deleting old main scripts (`measured_data_reconstruction.py`, `simulated_data_reconstruction.py`) immediately; modify them to be callable functions if needed for legacy mode.