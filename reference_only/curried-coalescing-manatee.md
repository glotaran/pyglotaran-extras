# Kinetic Scheme Visualizer — Re-implementation Plan

## Context

The existing `kinetic-scheme-visualizer` (thesis project) provides interactive visualization of
pyglotaran decay kinetic schemes in Jupyter notebooks. While functionally useful, it relies on a
heavy JavaScript stack (cytoscape.js + 6 plugins, jquery, npm/esbuild build system, anywidget)
that makes it complex to build, maintain, and distribute.

**Goal:** Re-implement the visualizer as a new module within pyglotaran-extras, using only
matplotlib (already a dependency) for rendering. Zero new dependencies. Publication-quality
static output. Pure Python layout algorithms.

---

## Analysis of Existing Implementation

### What works well (preserve)
- Clean separation: model parsing → graph construction → rendering
- K-matrix extraction via `fill_item()` from pyglotaran
- Pydantic-based configuration (`VisualizationOptions`)
- Color mapping per node, custom labels, node dimensions
- Ground state decay handling (diagonal k-matrix → GS nodes)
- Rate constant unit conversion (ps⁻¹ → ns⁻¹)

### What needs fixing
- **Dependencies**: cytoscape.js + 6 plugins, jquery, konva, npm, esbuild (→ matplotlib only)
- **Dead code**: `layout_directed_acyclic_graph()`, `layout_directed_cyclic_graph()` never used
- **Unused deps**: konva, jquery, cytoscape-node-editing, cytoscape-avsdf
- **Inconsistent serialization**: `.__dict__` vs `.model_dump_json()` in two entry points
- **Magic strings**: "GS", "colour_node_mapping", etc. hardcoded throughout
- **Data loss**: `round_and_convert()` rounds rate constants irreversibly, no user control
- **No type hints** on most functions, no docstrings
- **No tests**
- **Export only client-side**: PNG/JSON export via JS buttons, not accessible from Python
- **Console.log** debugging left in production JS
- **Silent skipping**: non-decay megacomplexes silently ignored with `continue`

---

## Architecture

Four clean layers with well-defined interfaces:

```
Layer 1: Model Extraction     k_matrix_parser.py    Model + Params → list[Transition]
            ↓
Layer 2: Graph Construction   kinetic_graph.py      list[Transition] → KineticGraph
            ↓
Layer 3: Layout Engine        layout.py             KineticGraph → NodePositions
            ↓
Layer 4: Rendering            plot_kinetic_scheme.py KineticGraph + NodePositions → Figure
```

---

## Module Structure

New files to create under `c:\src\pyglotaran-extras\`:

```
pyglotaran_extras/inspect/kinetic_scheme/
├── __init__.py                  # Public API exports
├── _constants.py                # Named constants (no magic strings)
├── k_matrix_parser.py           # Layer 1: extract transitions from pyglotaran models
├── kinetic_graph.py             # Layer 2: lightweight directed graph datastructure
├── layout.py                    # Layer 3: hierarchical + spring layout algorithms
└── plot_kinetic_scheme.py       # Layer 4: matplotlib rendering + config models

tests/inspect/kinetic_scheme/
├── __init__.py
├── test_k_matrix_parser.py
├── test_kinetic_graph.py
├── test_layout.py
└── test_plot_kinetic_scheme.py
```

Files to modify:
- `pyglotaran_extras/inspect/__init__.py` — add kinetic_scheme exports
- `pyglotaran_extras/__init__.py` — add `plot_kinetic_scheme` to top-level API

---

## Detailed Design

### Layer 1: `k_matrix_parser.py`

**Purpose:** Extract rate constant transitions from pyglotaran decay megacomplexes.

**Key type:**
```python
@dataclass(frozen=True)
class Transition:
    source: str                    # compartment name
    target: str                    # compartment name
    rate_constant: float           # raw value in ps⁻¹ (no premature rounding)
    parameter_label: str           # e.g. "rates.k21"
    is_ground_state_decay: bool    # True when source == target (diagonal)
    megacomplex_label: str         # which megacomplex this belongs to
```

**Key functions:**
```python
def extract_transitions(
    megacomplexes: str | list[str],
    model: Model,
    parameters: Parameters,
    *,
    omit_parameters: set[str] | None = None,
) -> list[Transition]

def extract_dataset_transitions(
    dataset_name: str,
    model: Model,
    parameters: Parameters,
    *,
    exclude_megacomplexes: set[str] | None = None,
    omit_parameters: set[str] | None = None,
) -> list[Transition]
```

**Reuses from existing code:**
- `fill_item()` from `glotaran.model.item` (same as current `utils.py:41`)
- `model.megacomplex[mc].get_k_matrix()` (same as current `utils.py:42`)
- K-matrix `.matrix` iteration pattern (same as current `utils.py:22`)

**Improvements over existing:**
- Preserves raw rate constants (no rounding until render time)
- Raises `TypeError` for non-decay megacomplexes instead of silent `continue`
- Accepts `str | list[str]` with normalization, like current `visualizer.py:24-27`
- Uses `set[str]` for omit_parameters (O(1) lookup)

### Layer 2: `kinetic_graph.py`

**Purpose:** Lightweight directed graph for kinetic schemes. No networkx dependency.

**Key types:**
```python
@dataclass
class KineticNode:
    label: str
    display_label: str | None = None
    is_ground_state: bool = False
    megacomplex: str | None = None

@dataclass
class KineticEdge:
    source: str
    target: str
    rate_constant_ps: float            # raw value
    parameter_label: str

    def format_rate_label(self, unit: str = "ns", decimal_places: int | None = None) -> str: ...

@dataclass
class KineticGraph:
    nodes: dict[str, KineticNode]
    edges: list[KineticEdge]
    _adjacency: dict[str, list[str]]   # forward adjacency list
    _reverse_adjacency: dict[str, list[str]]

    def add_node(self, node: KineticNode) -> None: ...
    def add_edge(self, edge: KineticEdge) -> None: ...
    def successors(self, label: str) -> list[str]: ...
    def predecessors(self, label: str) -> list[str]: ...
    def compartment_nodes(self) -> list[KineticNode]: ...  # non-GS nodes
    def ground_state_nodes(self) -> list[KineticNode]: ...
    def is_dag(self) -> bool: ...           # DFS cycle detection
    def topological_sort(self) -> list[str]: ...  # Kahn's algorithm

    @classmethod
    def from_transitions(
        cls,
        transitions: list[Transition],
        *,
        merge_ground_state_decays: bool = True,
    ) -> KineticGraph: ...
```

**Replaces:**
- `networkx.DiGraph` → `dict`-based adjacency lists (adequate for 3-20 node graphs)
- `networkx.is_directed_acyclic_graph()` → simple DFS (~10 lines)
- `networkx.topological_sort()` → Kahn's algorithm (~15 lines)
- `cytoscape_data()` → no longer needed (matplotlib draws directly)
- `apply_some_adjustments()` → `merge_ground_state_decays` parameter in factory method

### Layer 3: `layout.py`

**Purpose:** Compute (x, y) positions for graph nodes. Pure Python + numpy.

```python
class LayoutAlgorithm(str, Enum):
    HIERARCHICAL = "hierarchical"
    SPRING = "spring"
    MANUAL = "manual"

NodePositions = dict[str, tuple[float, float]]

def compute_layout(
    graph: KineticGraph,
    algorithm: LayoutAlgorithm = LayoutAlgorithm.HIERARCHICAL,
    *,
    horizontal_spacing: float = 2.0,
    vertical_spacing: float = 1.5,
    ground_state_offset: float = 1.2,
) -> NodePositions: ...
```

**Algorithms:**
- **Hierarchical** (default): Topological sort assigns layers; nodes within each layer are
  centered to minimize crossings. Falls back to cycle-breaking for non-DAGs. Replaces the
  cytoscape-dagre plugin.
- **Spring** (Fruchterman-Reingold): ~40 lines of numpy for force-directed layout. For complex
  cyclic schemes with back-transfer.
- **Manual**: User-supplied positions passed through unchanged.

Ground state nodes are positioned below their parent compartment with a configurable offset.

### Layer 4: `plot_kinetic_scheme.py`

**Purpose:** Matplotlib rendering with Pydantic configuration.

**Configuration:**
```python
class NodeStyle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_label: str | None = None
    width: float = 1.2
    height: float = 0.6
    facecolor: str | None = None
    fontsize: int = 10

class KineticSchemeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_styles: dict[str, NodeStyle] = Field(default_factory=dict)
    color_mapping: dict[str, list[str]] = Field(default_factory=dict)
    omit_parameters: set[str] = Field(default_factory=set)
    rate_unit: Literal["ps", "ns"] = "ns"
    rate_decimal_places: int | None = None
    show_ground_state: bool = False
    layout_algorithm: Literal["hierarchical", "spring", "manual"] = "hierarchical"
    manual_positions: dict[str, tuple[float, float]] | None = None
    figsize: tuple[float, float] = (10, 8)
    title: str | None = None
    node_facecolor: str = "#4A90D9"
    node_edgecolor: str = "#2C3E50"
    edge_color: str = "#333333"
    ground_state_facecolor: str = "#CCCCCC"
    horizontal_spacing: float = 2.0
    vertical_spacing: float = 1.5
```

**Public functions:**
```python
def plot_kinetic_scheme(
    megacomplexes: str | list[str],
    model: Model,
    parameters: Parameters,
    *,
    ax: Axes | None = None,
    config: KineticSchemeConfig | None = None,
    figsize: tuple[float, float] = (10, 8),
    title: str | None = None,
) -> tuple[Figure, Axes]: ...

def plot_dataset_kinetic_scheme(
    dataset_name: str,
    model: Model,
    parameters: Parameters,
    *,
    exclude_megacomplexes: set[str] | None = None,
    ax: Axes | None = None,
    config: KineticSchemeConfig | None = None,
    figsize: tuple[float, float] = (10, 8),
    title: str | None = None,
) -> tuple[Figure, Axes]: ...
```

**Rendering approach:**
- Nodes: `matplotlib.patches.FancyBboxPatch` with `boxstyle="round,pad=0.05"`
- Edges: `matplotlib.patches.FancyArrowPatch` with `arrowstyle="->"`
- Edge labels: `ax.text()` at edge midpoint with white background bbox
- Text contrast: luminance-based calculation (`0.299*R + 0.587*G + 0.114*B`) instead of
  hardcoded color list
- Arrow endpoints: computed to connect at node boundaries, not centers

**Export (fixes major weakness of old implementation):**
```python
fig, ax = plot_kinetic_scheme("decay_mc", model, params)
fig.savefig("scheme.svg", bbox_inches="tight")   # SVG for papers
fig.savefig("scheme.png", dpi=300)                # PNG
fig.savefig("scheme.pdf")                         # PDF
```

### `_constants.py`

Named constants replacing all magic strings/numbers:
- `GROUND_STATE_PREFIX = "GS"`
- `PS_INVERSE_TO_NS_INVERSE = 1e3`
- Default dimensions, colors, spacing values

---

## API Integration

Add to `pyglotaran_extras/inspect/__init__.py`:
```python
from pyglotaran_extras.inspect.kinetic_scheme import plot_kinetic_scheme
from pyglotaran_extras.inspect.kinetic_scheme import plot_dataset_kinetic_scheme
from pyglotaran_extras.inspect.kinetic_scheme import KineticSchemeConfig
```

Add to `pyglotaran_extras/__init__.py`:
```python
from pyglotaran_extras.inspect.kinetic_scheme import plot_kinetic_scheme
from pyglotaran_extras.inspect.kinetic_scheme import plot_dataset_kinetic_scheme
from pyglotaran_extras.inspect.kinetic_scheme import KineticSchemeConfig
```

---

## Code Quality Requirements

Following pyglotaran-extras standards (from `pyproject.toml`):
- `from __future__ import annotations` in every file
- Full type annotations (`disallow_untyped_defs = true` in mypy)
- NumPy-style docstrings on all public functions/classes (`interrogate fail-under = 100`)
- `ConfigDict(extra="forbid")` on Pydantic models
- `TYPE_CHECKING` guards for type-only imports
- Python 3.10+ syntax

---

## Implementation Order

1. `_constants.py` — define all named constants
2. `k_matrix_parser.py` + `test_k_matrix_parser.py` — Transition dataclass, extract functions
3. `kinetic_graph.py` + `test_kinetic_graph.py` — graph datastructure, factory method
4. `layout.py` + `test_layout.py` — hierarchical and spring layout algorithms
5. `plot_kinetic_scheme.py` + `test_plot_kinetic_scheme.py` — config models, rendering, public API
6. `__init__.py` files — wire up exports at kinetic_scheme, inspect, and top-level
7. Integration test with example model YAML from thesis repo

---

## Verification

1. **Unit tests**: Run `pytest tests/inspect/kinetic_scheme/ -v`
2. **Type checking**: Run `mypy pyglotaran_extras/inspect/kinetic_scheme/`
3. **Docstring coverage**: Run `interrogate pyglotaran_extras/inspect/kinetic_scheme/`
4. **Integration test**: Load example model YAML files from
   `kinetic-scheme-visualizer/example/` and verify output renders correctly:
   ```python
   from glotaran.io import load_model, load_parameters
   from pyglotaran_extras import plot_kinetic_scheme, KineticSchemeConfig

   model = load_model("target_rcg_gcrcg_rcgcr_refine.yml")
   params = load_parameters("target_rcg_gcrcg_rcgcr_refine-params.yml")
   fig, ax = plot_kinetic_scheme("complex_rcg_dcm", model, params)
   fig.savefig("test_output.svg")
   ```
5. **Full test suite**: Run `pytest tests/ -v` to ensure no regressions

---

## Key Files Reference

| Purpose | Path |
|---------|------|
| Existing k-matrix extraction logic | `kinetic-scheme-visualizer/kineticschemevisualizer/utils.py` |
| Existing entry points | `kinetic-scheme-visualizer/kineticschemevisualizer/visualizer.py` |
| Existing JS rendering | `kinetic-scheme-visualizer/js/widget.js` |
| pyglotaran-extras inspect module | `c:\src\pyglotaran-extras\pyglotaran_extras\inspect\__init__.py` |
| pyglotaran-extras top-level API | `c:\src\pyglotaran-extras\pyglotaran_extras\__init__.py` |
| pyglotaran-extras a_matrix pattern | `c:\src\pyglotaran-extras\pyglotaran_extras\inspect\a_matrix.py` |
| pyglotaran-extras plot_overview pattern | `c:\src\pyglotaran-extras\pyglotaran_extras\plotting\plot_overview.py` |
| Test conftest (fixtures, close figs) | `c:\src\pyglotaran-extras\tests\conftest.py` |
| Example model YAMLs | `kinetic-scheme-visualizer/example/*.yml` |
| pyglotaran-extras dependencies | `c:\src\pyglotaran-extras\pyproject.toml` |
