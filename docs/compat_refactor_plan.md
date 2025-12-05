# Compatibility Layer Refactoring Plan

## Overview

This document outlines the plan for updating the `pyglotaran-extras` compatibility layer (`compat` module) to accommodate the fundamental structural changes in the pyglotaran `Result` object between v0.7 and v0.8.

---

## Current Compat Implementation Analysis

### Module Structure

The compat module consists of three files:

- `__init__.py` - Exports the `convert` function
- `compat_result.py` - Contains `CompatResult` class that wraps the new Result
- `convert_result_dataset.py` - Contains conversion functions for datasets and results

### `CompatResult` Class (`compat_result.py`)

The `CompatResult` class inherits from the new `Result` and provides v0.7-style property access:

#### Properties Delegated to `optimization_info`:

| Property                         | v0.7 Name | Maps to (v0.8)                                     |
| -------------------------------- | --------- | -------------------------------------------------- |
| `number_of_function_evaluations` | Same      | `optimization_info.number_of_function_evaluations` |
| `success`                        | Same      | `optimization_info.success`                        |
| `termination_reason`             | Same      | `optimization_info.termination_reason`             |
| `glotaran_version`               | Same      | Hardcoded to "v0.7.3"                              |
| `number_of_residuals`            | Same      | `optimization_info.number_of_data_points`          |
| `number_of_free_parameters`      | Same      | `optimization_info.number_of_parameters`           |
| `number_of_clps`                 | Same      | `optimization_info.number_of_clps`                 |
| `degrees_of_freedom`             | Same      | `optimization_info.degrees_of_freedom`             |
| `chi_square`                     | Same      | `optimization_info.chi_square`                     |
| `reduced_chi_square`             | Same      | `optimization_info.reduced_chi_square`             |
| `reduced_chi_squared`            | Alias     | `optimization_info.reduced_chi_square`             |
| `root_mean_square_error`         | Same      | `optimization_info.root_mean_square_error`         |
| `additional_penalty`             | Same      | `optimization_info.additional_penalty`             |

#### Key Compatibility Properties:

- **`data`** → Returns `self.datasets` (v0.7 used `result.data`, v0.8 uses `result.optimization_results`)
- **`model`** → Returns first experiment from `self.experiments`

#### Class Method:

- **`from_result(cls, result: Result)`** - Factory method that creates a `CompatResult` from a v0.8 `Result`
  - Passes: `optimization_results`, `experiments`, `optimization_info`, `initial_parameters`, `optimized_parameters`

#### Markdown Rendering:

- Custom `markdown()` method that formats result information in v0.7 style
- Uses dataset attributes like `weighted_root_mean_square_error` and `root_mean_square_error`

---

### Dataset Conversion Functions (`convert_result_dataset.py`)

#### Main Entry Point:

- **`convert(input, cleanup=False)`** - Dispatches to either `convert_result` or `convert_dataset`

#### Dataset Conversion Functions:

1. **`_adjust_fitted_data(ds, cleanup=False)`**

   - Renames `fit` → `fitted_data`

2. **`_adjust_concentrations(ds, cleanup=False)`**

   - Looks for variables containing `species_associated_concentration` or `species_concentration`
   - Creates `species_concentration` variable

3. **`_adjust_estimations_to_spectra(ds, cleanup=False)`**

   - Handles `kinetic_associated_estimation` / `kinetic_associated_amplitude` → `decay_associated_spectra_mc{N}`
   - Handles `species_associated_estimation` / `species_associated_amplitude` → `species_associated_spectra`
   - Handles `damped_oscillation_associated_estimation` / `damped_oscillation_associated_amplitude` → `damped_oscillation_associated_spectra`

4. **`_adjust_activation_to_irf(ds, cleanup=False)`**

   - Converts gaussian activation parameters to IRF:
     - `gaussian_activation_center` → `irf_center`
     - `gaussian_activation_width` → `irf_width`
     - `gaussian_activation_scale` → `irf_scale`
     - `gaussian_activation_function` → `irf`
     - `gaussian_activation_dispersion` → `irf_center_location`

5. **`convert_dataset(dataset, cleanup=False)`**

   - Applies all adjustment functions
   - Ensures `weighted_root_mean_square_error` attribute exists

6. **`convert_result(result, cleanup=False)`**
   - Creates `CompatResult` from Result
   - Iterates over `optimization_results` and applies `convert_dataset` to each

---

## New v0.8 Result Structure

### Result Class (`glotaran/project/result.py`)

The new `Result` is a Pydantic `BaseModel` with these fields:

- `saving_options: SavingOptions`
- `optimization_results: dict[str, OptimizationResult]`
- `scheme: Scheme`
- `optimization_info: OptimizationInfo`
- `initial_parameters: Parameters`
- `optimized_parameters: Parameters`

#### Properties:

- `experiments` → `self.scheme.experiments` (dict of ExperimentModel)
- `input_data` → dict of input data from optimization_results

### OptimizationResult Class (`glotaran/optimization/objective.py`)

The new `OptimizationResult` is a Pydantic `BaseModel` with these fields:

- `elements: dict[str, xr.Dataset]` - Element results keyed by element name
- `activations: dict[str, xr.Dataset]` - Activation results keyed by activation name
- `input_data: xr.DataArray | xr.Dataset`
- `residuals: xr.DataArray | xr.Dataset | None`
- `meta: OptimizationResultMetaData`

#### OptimizationResultMetaData:

- `global_dimension: str`
- `model_dimension: str`
- `root_mean_square_error: float`
- `weighted_root_mean_square_error: float | None`
- `scale: float = 1`

#### Computed Property:

- `fitted_data` → `input_data - residuals`

---

## Key Changes to Address

### 1. Result Structure Change

**Old (v0.7):**

```python
result.datasets  # dict[str, xr.Dataset] - flat dataset per dataset name
```

**New (v0.8):**

```python
result.optimization_results  # dict[str, OptimizationResult]
# Where each OptimizationResult has:
#   .elements: dict[str, xr.Dataset]
#   .activations: dict[str, xr.Dataset]
#   .input_data: xr.DataArray | xr.Dataset
#   .residuals: xr.DataArray | xr.Dataset
#   .fitted_data: computed property
#   .meta: OptimizationResultMetaData
```

### 2. Data Variable Name Changes

The result datasets now contain data organized by element type rather than in a flat structure:

- Element-specific data is in `optimization_result.elements[element_name]`
- Activation-specific data is in `optimization_result.activations[activation_name]`

### 3. Element Dataset Identification via `element_uid`

Each element dataset has a dataset attribute `element_uid` that identifies the element type that created it:

| Element Type       | `element_uid` Attribute Value                                           |
| ------------------ | ----------------------------------------------------------------------- |
| Kinetic            | `glotaran.builtin.elements.kinetic.KineticElement`                      |
| Damped Oscillation | `glotaran.builtin.elements.damped_oscillation.DampedOscillationElement` |

#### Data Variable Locations:

**Kinetic Element** (`element_uid == "glotaran.builtin.elements.kinetic.KineticElement"`):

- `kinetic_associated_*` variables
- `species_concentration` variables
- `species_associated_*` variables
- **Note:** Resulting data variable names should be suffixed with the element name

**Damped Oscillation Element** (`element_uid == "glotaran.builtin.elements.damped_oscillation.DampedOscillationElement"`):

- `damped_oscillation_*` variables

**Activations** (first activation in `activations` dict):

- All IRF-related information:
  - `gaussian_activation_center` → `irf_center`
  - `gaussian_activation_width` → `irf_width`
  - `gaussian_activation_scale` → `irf_scale`
  - `gaussian_activation_function` → `irf`
  - `gaussian_activation_dispersion` → `irf_center_location`

### 4. Dataset Attributes Change

**Old:** Attributes on the flat dataset like `weighted_root_mean_square_error`
**New:** Attributes are in `meta` field of `OptimizationResult`

### 5. Conversion Strategy Change

**Critical:** The old `convert_dataset` and its sub-functions **cannot manipulate a dataset in-place anymore** since the new result is split up into multiple datasets.

**New Approach Required:**

- A **new dataset must be created** and populated from the result
- Data must be gathered from multiple source datasets (`elements`, `activations`, `input_data`, `residuals`)
- The conversion functions need to be rewritten to **build** a flat dataset rather than **modify** an existing one

### 6. Experiments Access

**Old (compat):**

```python
result.experiments  # was passed directly to CompatResult
```

**New (v0.8):**

```python
result.scheme.experiments  # accessed via scheme property
```

---

## Refactoring Tasks

### Phase 1: Update CompatResult Class

1. [ ] Update `CompatResult.from_result()` to handle new Result structure

   - Note: `experiments` is now a property that delegates to `scheme.experiments`
   - Need to pass `scheme` instead of `experiments`

2. [ ] Update the `data` property

   - Current: Returns `self.datasets`
   - Problem: New Result doesn't have `datasets` attribute
   - New Result has `optimization_results` which are `OptimizationResult` objects, not datasets
   - Need to reconstruct flat datasets from the nested structure

3. [ ] Verify RMSE access in `markdown()` method
   - Current: Accesses `dataset.attrs["weighted_root_mean_square_error"]`
   - New: This is in `OptimizationResult.meta.weighted_root_mean_square_error`

### Phase 2: Rewrite Dataset Conversion Functions

**Important:** The conversion functions must be completely rewritten. They can no longer modify datasets in-place since the new result structure splits data across multiple datasets.

1. [ ] Create new `build_compat_dataset(optimization_result: OptimizationResult)` function

   - Creates a new empty `xr.Dataset`
   - Populates it by gathering data from all sources

2. [ ] Extract data from `OptimizationResult.input_data`

   - Copy `data` variable
   - Copy coordinate information

3. [ ] Extract data from `OptimizationResult.residuals`

   - Copy `residual` variable

4. [ ] Extract `fitted_data` from computed property

5. [ ] Process `OptimizationResult.elements` by `element_uid`:

   **For Kinetic Elements** (`element_uid == "glotaran.builtin.elements.kinetic.KineticElement"`):

   - Extract `kinetic_associated_*` → `decay_associated_spectra_{element_name}`
   - Extract `species_concentration` → `species_concentration_{element_name}`
   - Extract `species_associated_*` → `species_associated_spectra_{element_name}`
   - **All variable names must be suffixed with the element name**

   **For Damped Oscillation Elements** (`element_uid == "glotaran.builtin.elements.damped_oscillation.DampedOscillationElement"`):

   - Extract `damped_oscillation_*` → `damped_oscillation_associated_spectra_{element_name}`

6. [ ] Process first activation from `OptimizationResult.activations`:

   - Extract IRF information:
     - `gaussian_activation_center` → `irf_center`
     - `gaussian_activation_width` → `irf_width`
     - `gaussian_activation_scale` → `irf_scale`
     - `gaussian_activation_function` → `irf`
     - `gaussian_activation_dispersion` → `irf_center_location`

7. [ ] Set dataset attributes from `OptimizationResult.meta`:

   - `root_mean_square_error`
   - `weighted_root_mean_square_error`
   - `global_dimension`
   - `model_dimension`

8. [ ] Update `convert_result()` to use new `build_compat_dataset()`:
   ```python
   def convert_result(result: Result) -> CompatResult:
       converted_result = CompatResult.from_result(result)
       converted_result._compat_datasets = {
           key: build_compat_dataset(opt_result)
           for key, opt_result in result.optimization_results.items()
       }
       return converted_result
   ```

### Phase 3: Update Variable Mapping

1. [ ] Create element UID constants for readability:

   ```python
   KINETIC_ELEMENT_UID = "glotaran.builtin.elements.kinetic.KineticElement"
   DAMPED_OSCILLATION_ELEMENT_UID = "glotaran.builtin.elements.damped_oscillation.DampedOscillationElement"
   ```

2. [ ] Implement element type detection:

   ```python
   def get_element_type(element_dataset: xr.Dataset) -> str:
       return element_dataset.attrs.get("element_uid", "")
   ```

3. [ ] Map new variable names to old ones with element name suffix:

   | Source (v0.8)                              | Target (v0.7 compat)                                   |
   | ------------------------------------------ | ------------------------------------------------------ |
   | `kinetic_associated_estimation`            | `decay_associated_spectra_{element_name}`              |
   | `species_concentration`                    | `species_concentration_{element_name}`                 |
   | `species_associated_estimation`            | `species_associated_spectra_{element_name}`            |
   | `damped_oscillation_associated_estimation` | `damped_oscillation_associated_spectra_{element_name}` |

4. [ ] Handle `fitted_data`:
   - Old: Variable named `fitted_data` in dataset
   - New: Computed property on `OptimizationResult` (`input_data - residuals`)
   - Copy to compat dataset as `fitted_data` variable

### Phase 4: Testing and Validation

1. [ ] Create test cases comparing old and new result structures
2. [ ] Verify all plotting functions work with converted results
3. [ ] Update any notebooks that use the compat layer

---

## Open Questions

1. ~~**Should we flatten the new structure?**~~ ✅ RESOLVED

   - **Answer:** Yes, we must flatten the structure to create a v0.7-compatible flat dataset
   - A new dataset must be created and populated from the split result structure

2. **How to handle the `scheme` vs `experiments` change?**

   - `CompatResult` currently accepts `experiments` directly
   - New Result gets experiments from `scheme.experiments`
   - Need to determine if CompatResult should store scheme or keep extracting experiments

3. **What about the missing `datasets` attribute?**

   - CompatResult inherits from Result and expects `datasets`
   - New Result has `optimization_results` instead
   - **Proposed solution:** Store flattened datasets in `_compat_datasets` and expose via `data` property

4. ~~**Element and activation dataset structure?**~~ ✅ RESOLVED

   - **Answer:** Element datasets have `element_uid` attribute to identify their type:
     - Kinetic data: `element_uid == "glotaran.builtin.elements.kinetic.KineticElement"`
     - Damped oscillation: `element_uid == "glotaran.builtin.elements.damped_oscillation.DampedOscillationElement"`
   - IRF data is in the first activation dataset

5. **How to handle multiple elements of the same type?**

   - With element name suffixes, we can have `species_associated_spectra_kinetic1`, `species_associated_spectra_kinetic2`, etc.
   - Need to verify this doesn't break existing plotting functions

6. **Should old `_adjust_*` functions be kept or removed?**
   - They were designed for in-place modification
   - The new approach builds a fresh dataset
   - Consider removing them entirely and using a clean implementation

---

## Files to Modify

1. `pyglotaran_extras/compat/compat_result.py`
2. `pyglotaran_extras/compat/convert_result_dataset.py`
3. `pyglotaran_extras/compat/__init__.py` (if API changes)
4. Related test files (if any)
5. Example notebooks using compat layer

---

## References

- **New Result class:** `pyglotaran/glotaran/project/result.py`
- **New OptimizationResult class:** `pyglotaran/glotaran/optimization/objective.py`
- **Current compat module:** `pyglotaran-extras/pyglotaran_extras/compat/`
