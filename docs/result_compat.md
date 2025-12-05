# Compatibility Layer Documentation

The `compat` module converts pyglotaran v0.8 `Result` objects to a flat dataset format compatible with v0.7-style plotting functions.

## v0.8 Result Structure

```python
Result
├── optimization_results: dict[str, OptimizationResult]
│   └── OptimizationResult
│       ├── elements: dict[str, xr.Dataset]      # Element results by name
│       ├── activations: dict[str, xr.Dataset]   # Activation (IRF) results
│       ├── input_data: xr.DataArray | xr.Dataset
│       ├── residuals: xr.DataArray | xr.Dataset
│       ├── fitted_data: property (input_data - residuals)
│       └── meta: OptimizationResultMetaData
├── scheme: Scheme
│   └── experiments: dict[str, ExperimentModel]
├── optimization_info: OptimizationInfo
├── initial_parameters: Parameters
└── optimized_parameters: Parameters
```

## Element Identification

Each element dataset has an `element_uid` attribute:

| Element Type       | `element_uid`                                                                   |
| ------------------ | ------------------------------------------------------------------------------- |
| Kinetic            | `glotaran.builtin.elements.kinetic.element.KineticElement`                      |
| Damped Oscillation | `glotaran.builtin.elements.damped_oscillation.element.DampedOscillationElement` |
| Coherent Artifact  | `glotaran.builtin.elements.coherent_artifact.element.CoherentArtifactElement`   |

## Variable Mappings

### Core Data

| v0.8 Source                      | v0.7 Target   |
| -------------------------------- | ------------- |
| `OptimizationResult.input_data`  | `data`        |
| `OptimizationResult.residuals`   | `residual`    |
| `OptimizationResult.fitted_data` | `fitted_data` |

### Kinetic Element

| v0.8 Source                             | v0.7 Target                                 |
| --------------------------------------- | ------------------------------------------- |
| `concentrations`                        | `species_concentration_{element_name}`      |
| `amplitudes`                            | `species_associated_spectra_{element_name}` |
| `kinetic_amplitudes.isel(activation=0)` | `decay_associated_spectra_{element_name}`   |

Dimension rename: `compartment` → `species_{element_name}`

### Damped Oscillation Element

| v0.8 Source          | v0.7 Target                             |
| -------------------- | --------------------------------------- |
| `cos_concentrations` | `damped_oscillation_cos`                |
| `sin_concentrations` | `damped_oscillation_sin`                |
| `amplitudes`         | `damped_oscillation_associated_spectra` |
| `phase_amplitudes`   | `damped_oscillation_phase`              |

Coordinate renames: `oscillation` → `damped_oscillation`, `oscillation_frequency` → `damped_oscillation_frequency`, `oscillation_rate` → `damped_oscillation_rate`

### Coherent Artifact Element

| v0.8 Source      | v0.7 Target                            |
| ---------------- | -------------------------------------- |
| `concentrations` | `coherent_artifact_response`           |
| `amplitudes`     | `coherent_artifact_associated_spectra` |

Dimension rename: `derivative` → `coherent_artifact_order`

### Activation (IRF) Data

| v0.8 Source         | v0.7 Target           |
| ------------------- | --------------------- |
| `attrs["center"]`   | `irf_center`          |
| `attrs["width"]`    | `irf_width`           |
| `attrs["scale"]`    | `irf_scale`           |
| `"trace"` variable  | `irf`                 |
| `"center"` variable | `center_dispersion_1` |

### Dataset Attributes

| v0.8 Source (`meta`)              | v0.7 Attribute                    |
| --------------------------------- | --------------------------------- |
| `root_mean_square_error`          | `root_mean_square_error`          |
| `weighted_root_mean_square_error` | `weighted_root_mean_square_error` |
| `global_dimension`                | `global_dimension`                |
| `model_dimension`                 | `model_dimension`                 |

## CompatResult Properties

| v0.7 Property                                                                                                                                                                   | v0.8 Source                                        |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| `data`                                                                                                                                                                          | Flattened datasets (via `_compat_datasets`)        |
| `model`                                                                                                                                                                         | First experiment from `scheme.experiments`         |
| `number_of_function_evaluations`                                                                                                                                                | `optimization_info.number_of_function_evaluations` |
| `number_of_residuals`                                                                                                                                                           | `optimization_info.number_of_data_points`          |
| `number_of_free_parameters`                                                                                                                                                     | `optimization_info.number_of_parameters`           |
| `success`, `termination_reason`, `glotaran_version`, `number_of_clps`, `degrees_of_freedom`, `chi_square`, `reduced_chi_square`, `root_mean_square_error`, `additional_penalty` | Delegated to `optimization_info`                   |
