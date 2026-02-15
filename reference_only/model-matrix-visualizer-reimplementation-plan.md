# Pyglotaran Model Matrix Visualizer Reimplementation Plan

## 1. Scope

Re-implement the current `kinetic-scheme-visualizer` functionality in `pyglotaran-extras` with:

- modern code quality and test coverage
- minimal dependencies (pure Python core)
- deterministic output for reproducibility
- optional interactivity that does not infect core architecture

Primary target: visualize kinetic state-transition structure and rates from pyglotaran model definitions.

## 2. What Works Well In The Old Implementation

1. Direct pyglotaran integration:
   - The flow from model/parameters to filled k-matrices is straightforward (`kinetic-scheme-visualizer/kineticschemevisualizer/utils.py:34`, `kinetic-scheme-visualizer/kineticschemevisualizer/utils.py:41`).
2. Domain-first graph construction:
   - Transition extraction from `k_matrix` entries is clear and compact (`kinetic-scheme-visualizer/kineticschemevisualizer/utils.py:16`, `kinetic-scheme-visualizer/kineticschemevisualizer/utils.py:22`).
3. Useful first-level UX features:
   - Node alias/size customization and color mapping are user-relevant (`kinetic-scheme-visualizer/kineticschemevisualizer/visualizer.py:12`, `kinetic-scheme-visualizer/kineticschemevisualizer/visualizer.py:18`, `kinetic-scheme-visualizer/js/widget.js:43`).
4. Jupyter friendliness:
   - `anywidget` wrapper gives notebook integration with limited Python boilerplate (`kinetic-scheme-visualizer/kineticschemevisualizer/widget.py:12`).
5. Export capability:
   - PNG and JSON export are practical (`kinetic-scheme-visualizer/js/utils.js:11`, `kinetic-scheme-visualizer/js/widget.js:147`).

## 3. What Should Be Improved

### 3.1 Correctness and API consistency

1. Inconsistent serialization paths:
   - `visualize_megacomplex` passes Python dicts, but `visualize_dataset_model` passes JSON strings to traitlets Dict fields (`kinetic-scheme-visualizer/kineticschemevisualizer/visualizer.py:35`, `kinetic-scheme-visualizer/kineticschemevisualizer/visualizer.py:55`).
2. Mutable default arguments:
   - `VisualizationOptions()` as a function default appears in both public entry points (`kinetic-scheme-visualizer/kineticschemevisualizer/visualizer.py:23`, `kinetic-scheme-visualizer/kineticschemevisualizer/visualizer.py:39`).
   - Dict/list defaults in Pydantic model should be explicit factories (`kinetic-scheme-visualizer/kineticschemevisualizer/visualizer.py:17`, `kinetic-scheme-visualizer/kineticschemevisualizer/visualizer.py:18`, `kinetic-scheme-visualizer/kineticschemevisualizer/visualizer.py:19`).
3. Event hook bug:
   - `cy.on("free ", ...)` has trailing whitespace; change tracking may never fire (`kinetic-scheme-visualizer/js/widget.js:133`).
4. Scientific fidelity issue:
   - Rates are rounded and unit-converted during graph construction, not formatting (`kinetic-scheme-visualizer/kineticschemevisualizer/utils.py:12`, `kinetic-scheme-visualizer/kineticschemevisualizer/utils.py:24`).

### 3.2 Maintainability

1. Mixed responsibilities:
   - Domain extraction, transformation, and display-oriented rules are coupled in one utility module (`kinetic-scheme-visualizer/kineticschemevisualizer/utils.py:16`, `kinetic-scheme-visualizer/kineticschemevisualizer/utils.py:45`).
2. Dead or unintegrated layout logic:
   - Python layout methods are not wired into the active widget path (`kinetic-scheme-visualizer/kineticschemevisualizer/utils.py:70`, `kinetic-scheme-visualizer/kineticschemevisualizer/utils.py:124`).
3. Dead JS helpers/imports:
   - Unused helpers and unused layout entry points remain (`kinetic-scheme-visualizer/js/utils.js:26`, `kinetic-scheme-visualizer/js/layouts.js:42`).
4. Debug output left in shipping code:
   - `console.log` calls in render path (`kinetic-scheme-visualizer/js/widget.js:76`, `kinetic-scheme-visualizer/js/widget.js:140`).

### 3.3 Dependency and build footprint

1. Large JS bundle:
   - Built widget is ~1.6 MB (`kinetic-scheme-visualizer/kineticschemevisualizer/static/widget.js`).
2. Node toolchain requirement:
   - Packaging depends on npm/esbuild build hooks (`kinetic-scheme-visualizer/pyproject.toml`).
3. Plugin-heavy JS dependency tree:
   - Cytoscape plugin stack and transitive jQuery/Konva-style baggage (`kinetic-scheme-visualizer/package.json`).

### 3.4 Engineering process

1. No test suite discovered for visualizer behavior.
2. No explicit stable serialization contract for nodes/edges/options.
3. No documented compatibility policy with pyglotaran model variants.

## 4. What Should Be Done Completely Differently

1. Make the core visualizer fully Python:
   - No required JS runtime in the core feature path.
2. Separate architecture into strict layers:
   - `extract -> normalize -> layout -> render`.
3. Treat rates/labels as data + formatter:
   - Never round/convert values before rendering.
4. Make rendering backend-pluggable:
   - Start with Matplotlib backend (likely existing in `pyglotaran-extras` workflows), add optional interactive backend later.
5. Ship with test-first contract:
   - Deterministic layout output tests, extraction tests, and regression fixtures from real models.

## 5. Proposed Modern Architecture (for `pyglotaran-extras`)

## 5.1 Package structure

Proposed module layout:

- `pyglotaran_extras/model_matrix/types.py`
- `pyglotaran_extras/model_matrix/extract.py`
- `pyglotaran_extras/model_matrix/normalize.py`
- `pyglotaran_extras/model_matrix/layout.py`
- `pyglotaran_extras/model_matrix/render_matplotlib.py`
- `pyglotaran_extras/model_matrix/api.py`
- `pyglotaran_extras/model_matrix/style.py`

Optional later:

- `pyglotaran_extras/model_matrix/render_anywidget.py`

## 5.2 Data model (pure Python)

Use `dataclasses` (or `typing.NamedTuple` where appropriate), not Pydantic in core:

- `NodeSpec`: `id`, `label`, `kind` (`state`, `sink`, `virtual`), style hints
- `EdgeSpec`: `source`, `target`, `rate_value`, `rate_label`, `edge_kind`
- `GraphSpec`: immutable lists + metadata (`dataset`, `megacomplexes`, units)
- `LayoutSpec`: node positions and routing hints

Core library should accept and return Python objects and plain dicts; JSON only at optional IO boundaries.

## 5.3 Pipeline

1. `extract.py`:
   - Parse pyglotaran model + parameters, resolve filled k-matrices.
   - Capture full-rate metadata (raw value + label + source megacomplex).
2. `normalize.py`:
   - Merge policy is explicit and configurable:
     - keep-separate edges
     - aggregate-by-label
     - aggregate-by-target
3. `layout.py`:
   - deterministic layered layout for DAG
   - SCC-based condensation + layered layout for cyclic graphs
   - no dependence on browser layout engines
4. `render_matplotlib.py`:
   - draw nodes/edges/labels with style presets
   - return `(fig, ax)` + optional structured export dict

## 5.4 Dependency strategy

Minimum hard dependencies:

- `pyglotaran` (already domain dependency)
- `numpy` (likely already in stack)
- `matplotlib` (if already part of `pyglotaran-extras`; otherwise keep renderer optional)

Avoid in core:

- `anywidget`, `networkx`, JS toolchain, Cytoscape plugins

Optional extras:

- `interactive`: anywidget + compact frontend (only if truly needed)
- `graph`: `networkx` adapter for external tooling (optional, not core)

## 5.5 Public API sketch

```python
from pyglotaran_extras.model_matrix import (
    extract_graph,
    layout_graph,
    plot_graph,
    visualize_model_matrix,
    VisualizerOptions,
)
```

- `extract_graph(model, parameters, megacomplexes=..., dataset_model=...) -> GraphSpec`
- `layout_graph(graph, algorithm="layered") -> LayoutSpec`
- `plot_graph(graph, layout=None, options=None) -> matplotlib.figure.Figure`
- `visualize_model_matrix(...)` convenience wrapper

## 5.6 Styling model

Style config should be explicit and type-safe:

- node palette by class
- label formatter callback for rates
- scale profile (`compact`, `publication`, `presentation`)
- deterministic defaults (fixed font sizes/arrow sizes)

No hidden style values from frontend defaults.

## 6. Reimplementation Plan (Phased)

## Phase 0: Requirements & parity checklist (1-2 days)

1. Define supported pyglotaran constructs for v1:
   - decay megacomplex and dataset model subsets.
2. Freeze parity cases from old examples as fixtures.
3. Define visual acceptance metrics:
   - no overlaps on standard examples
   - deterministic layout hashes

Deliverable: `docs/model_matrix_requirements.md`.

## Phase 1: Core extraction and graph spec (2-3 days)

1. Build `types.py` dataclasses.
2. Implement `extract.py` without rendering concerns.
3. Add robust validation errors with actionable messages.

Tests:

- fixture-driven extraction snapshots
- invalid dataset/megacomplex cases
- edge direction and self-decay handling

## Phase 2: Layout engine (2-4 days)

1. Implement deterministic layered layout.
2. Add cyclic fallback using SCC condensation.
3. Add layout quality tests (overlap checks + stable coordinates).

## Phase 3: Matplotlib renderer (2-4 days)

1. Implement `render_matplotlib.py`.
2. Add theme presets and publication-friendly defaults.
3. Implement export helpers (SVG/PNG + JSON spec).

Tests:

- image-regression snapshots for canonical fixtures
- style option regression tests

## Phase 4: High-level API + docs (1-2 days)

1. Add `api.py` convenience methods.
2. Notebook examples and migration notes from old package.
3. Document extension hooks for custom label/style functions.

## Phase 5: Optional interactive backend (separate track)

Only if needed after core is stable:

1. Add optional `anywidget` backend from existing `GraphSpec`.
2. Keep JS minimal and plugin-light.
3. Ensure no API drift between static and interactive renderers.

## 7. Testing Strategy

1. Unit tests:
   - extraction correctness and edge metadata
   - merge policies
   - label formatter behavior
2. Integration tests:
   - end-to-end from model+params to figure
3. Regression tests:
   - snapshot of `GraphSpec`
   - image snapshots for selected canonical schemes
4. Property tests (if time permits):
   - layout determinism under node insertion order changes

## 8. Migration Strategy

1. Provide mapping from old options to new `VisualizerOptions`.
2. Add deprecation adapter helpers for:
   - `colour_node_mapping`
   - `omitted_rate_constants`
3. Publish migration notebook:
   - old vs new output comparison on the same model files.

## 9. Recommended MVP Boundary

Include:

- extraction from `decay` megacomplexes
- deterministic layered layout
- Matplotlib rendering
- PNG/SVG export
- typed options + tests

Exclude initially:

- full browser editing UI
- drag-and-drop graph editing
- rich frontend plugin ecosystem

This gives a robust, low-dependency, production-ready base that can be extended later.
