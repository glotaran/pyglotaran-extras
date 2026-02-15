# Kinetic Scheme Visualizer — Analysis & Modernization Plan

## Architecture Overview

The current implementation is a **Jupyter widget** built on three layers:

| Layer | Tech | Role |
|---|---|---|
| **Python backend** | `anywidget` + `traitlets` + `networkx` + `pydantic` | Parse pyglotaran models, build graph data, expose widget |
| **JS build pipeline** | `esbuild` + `npm` + `hatch-jupyter-builder` | Bundle JS into a 44,600-line static file |
| **JS frontend** | `cytoscape.js` + 7 plugins + `jQuery` | Render interactive graph in browser |

---

## What Works Well

### 1. Core data extraction logic (`utils.py`)

The functions `get_filled_megacomplex_k_matrices()` and `build_all_transitions()` correctly walk a pyglotaran model, fill parameters, extract k-matrices, and convert them to a graph edge list. This is the real domain logic and is well-structured.

### 2. Pydantic options model (`visualizer.py`)

`VisualizationOptions` with `Node`, `colour_node_mapping`, and `omitted_rate_constants` is a clean API for the user to customize the output.

### 3. Two entry points

`visualize_megacomplex()` and `visualize_dataset_model()` cover the two main use cases (explicit megacomplex list vs. dataset-level).

### 4. anywidget choice

Using `anywidget` was a good modern choice for Jupyter widget development (no cookiecutter widget boilerplate).

---

## Problems & What Should Change

### 1. Massive, unnecessary JS dependency tree

The bundled `static/widget.js` is **44,600+ lines** because it embeds the entire Cytoscape.js library plus 7 plugins:

```
cytoscape          (core graph library - ~30K lines alone)
cytoscape-dagre    (layout)
cytoscape-avsdf    (layout - imported but never used)
cytoscape-panzoom  (zoom controls + jQuery dependency!)
cytoscape-grid-guide (snap-to-grid)
cytoscape-node-editing (not used in widget.js)
cytoscape-noderesize   (not used in widget.js)
konva              (imported but never used)
jquery             (required only by panzoom)
```

Several dependencies are **imported but never actually used** (`avsdf`, `node-editing`, `noderesize`, `konva`). jQuery is dragged in solely for the panzoom plugin. This is extreme bloat for what is essentially drawing labeled boxes with arrows.

### 2. Dual layout systems — both partially implemented, neither used well

`utils.py` contains two Python-side layout algorithms (`layout_directed_acyclic_graph` and `layout_directed_cyclic_graph`) that compute node positions. However, these are **never called** — the widget uses Cytoscape's dagre layout on the JS side instead. This is dead code that adds complexity.

### 3. Inconsistent data passing between the two `visualize_*` functions

In `visualizer.py`:

- `visualize_megacomplex()` passes `graph_data` as a **dict** and `visualization_options` as `.__dict__`
- `visualize_dataset_model()` passes `graph_data` as a **JSON string** (`json.dumps()`) and `visualization_options` as `.model_dump_json()`

This inconsistency would cause bugs — the widget's traitlets expect `Dict`, not strings.

### 4. NetworkX used only for trivial graph operations

The entire `networkx` dependency is used for:

- Creating a `DiGraph` and adding edges
- Exporting to Cytoscape JSON format (`cytoscape_data()`)
- Checking `is_directed_acyclic_graph()` (in dead code)
- `topological_sort()` (in dead code)

These are all trivially implementable without networkx.

### 5. Ground state merging logic is fragile

The `apply_some_adjustments()` function merges multiple ground-state edges by checking for `'GS'` in neighbor names, which is brittle — it relies on the synthetic naming convention from `build_all_transitions()`.

### 6. The JS frontend has hardcoded styling decisions

Text color is conditionally set to white only for `"black"`, `"brown"`, and `"blue"` backgrounds. This should use luminance-based contrast calculation. Node opacity checks for `"GS"` in the ID string to hide ground states — another fragile convention.

### 7. Build toolchain complexity

The project requires: npm, esbuild, hatch-jupyter-builder hook, and a two-stage build (npm build → Python package). This makes development, CI, and installation significantly harder than it needs to be.

### 8. No tests

The Makefile references pytest, but there are no test files in the project.

---

## Proposed Modern Architecture (Pure Python)

The key insight is that the actual rendering requirement is quite simple: **labeled rectangular nodes connected by labeled directed arrows**. This does not require a full graph visualization framework.

### Approach: SVG rendering in pure Python

```
┌─────────────────────────────────────────────┐
│  pyglotaran model + parameters              │
│         │                                    │
│         ▼                                    │
│  ModelParser (extract k-matrices → graph)    │
│         │                                    │
│         ▼                                    │
│  Graph (pure Python, no networkx)            │
│    - nodes: list[Node]                       │
│    - edges: list[Edge]                       │
│         │                                    │
│         ▼                                    │
│  LayoutEngine (auto-position nodes)          │
│    - Sugiyama/layered layout for DAGs        │
│    - Force-directed fallback for cycles      │
│         │                                    │
│         ▼                                    │
│  SVGRenderer (generate SVG string)           │
│    - Nodes as <rect> + <text>                │
│    - Edges as <path> with <marker> arrows    │
│    - Edge labels as <text>                   │
│    - CSS styling inline                      │
│         │                                    │
│         ▼                                    │
│  IPython.display.SVG  (or HTML for interact) │
│  OR save to .svg / .png / .pdf               │
└─────────────────────────────────────────────┘
```

### Dependencies

| Current | Proposed |
|---|---|
| `anywidget` | **None** — use `IPython.display.SVG` / `HTML` |
| `networkx` | **None** — simple `dict`-based adjacency list |
| `cytoscape.js` + 7 plugins + jQuery | **None** — generate SVG directly |
| `pydantic` | **Keep** (already a pyglotaran dep, good for options validation) |
| `pyglotaran` | **Keep** (core domain) |
| `esbuild` + npm | **None** |
| `xarray` | **Drop** (never used in the visualizer code) |

**Result: 0 additional dependencies beyond pyglotaran + pydantic** (both already in the user's environment).

### Key Design Decisions

#### 1. Pure SVG output

SVG is natively renderable in Jupyter, browsers, and can be saved as files. No JavaScript needed for static diagrams. For interactivity (dragging nodes), a lightweight `anywidget` with inline JS (~100 lines, no npm build) could be an optional mode.

#### 2. Simple layered layout algorithm

Implement a basic Sugiyama-style layout (topological ordering → layer assignment → crossing reduction → coordinate assignment). For the typical kinetic schemes (5–20 nodes), even a naive implementation will work well. This replaces both the dead Python layout code and the JS dagre dependency.

#### 3. Contrast-aware styling

Calculate relative luminance from any color to determine text color automatically.

#### 4. Clean graph abstraction

A `KineticScheme` dataclass with `nodes` and `edges`, where edges carry rate constant values and parameter labels. No synthetic "GS" nodes — represent ground-state decay as a node attribute instead, and render it differently (e.g., an arrow pointing to nothing, or a dashed border).

#### 5. Multiple output formats

`.svg` (vector), `.png` (via cairosvg if available, optional), inline Jupyter display, and `.tikz` (LaTeX) for publications.

#### 6. Testable

Pure Python data transformations are trivially unit-testable. The SVG output can be snapshot-tested.

### Suggested Module Structure

```
pyglotaran-model-visualizer/
├── pyproject.toml
├── src/
│   └── model_visualizer/
│       ├── __init__.py          # Public API
│       ├── parser.py            # Extract graph from pyglotaran model
│       ├── graph.py             # Node, Edge, KineticScheme dataclasses
│       ├── layout.py            # Auto-layout algorithms
│       ├── render_svg.py        # SVG generation
│       ├── render_tikz.py       # Optional LaTeX/TikZ output
│       ├── style.py             # Color schemes, contrast, themes
│       └── options.py           # Pydantic VisualizationOptions
└── tests/
    ├── test_parser.py
    ├── test_layout.py
    └── test_render.py
```

---

## Summary Comparison

| Aspect | Old | Proposed |
|---|---|---|
| Dependencies | 9+ (Python + JS) | 2 (pyglotaran, pydantic) |
| Build steps | npm install → npm build → pip install | pip install |
| Bundled JS | 44,600 lines | 0 (or ~100 for optional interactivity) |
| Rendering | Cytoscape.js in browser | Pure Python SVG |
| Layout | Dagre.js (or dead Python code) | Python Sugiyama/force-directed |
| Testability | None | Full unit tests possible |
| Output formats | PNG export (via canvas) | SVG, PNG, TikZ, Jupyter inline |
| Jupyter integration | anywidget (complex) | IPython.display.SVG (trivial) |

The core domain logic from `utils.py` (model parsing, k-matrix extraction, transition building) should be preserved and refined — it's the valuable part. Everything else should be rebuilt from scratch as pure Python.
