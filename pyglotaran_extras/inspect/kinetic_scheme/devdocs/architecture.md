# Kinetic Scheme Visualizer -- Architecture & Design

## What This Module Does

Renders kinetic decay schemes from [pyglotaran](https://github.com/glotaran/pyglotaran) models as node-and-arrow diagrams using matplotlib. Given a `Model` and `Parameters`, it extracts rate constant transitions from k-matrices, builds a graph, computes a layout, and draws the result.

## Module Location

```
pyglotaran_extras/inspect/kinetic_scheme/
```

## Public API

Exported from `__init__.py`:

| Symbol                          | Type           | Purpose                                          |
| ------------------------------- | -------------- | ------------------------------------------------ |
| `show_kinetic_scheme()`         | function       | Visualize one or more named megacomplexes        |
| `show_dataset_kinetic_scheme()` | function       | Visualize all decay megacomplexes for a dataset  |
| `KineticSchemeConfig`           | Pydantic model | Full configuration (styling, layout, formatting) |
| `NodeStyleConfig`               | Pydantic model | Per-node style overrides                         |

Both `show_*` functions return `(Figure, Axes)`.

---

## Four-Layer Architecture

Data flows through four layers in strict sequence. Each layer has exactly one responsibility and one source file.

```
Model + Parameters
       |
       v
 Layer 1: EXTRACT          _k_matrix_parser.py
       |  Transition[]
       v
 Layer 2: BUILD GRAPH      _kinetic_graph.py
       |  KineticGraph
       v
 Layer 3: LAYOUT            _layout.py
       |  NodePositions {label: (x,y)}
       v
 Layer 4: RENDER            plot_kinetic_scheme.py
       |  Figure + Axes
       v
   matplotlib output
```

### Layer 1 -- Extract (`_k_matrix_parser.py`)

**Input**: Megacomplex labels + `Model` + `Parameters`

**Output**: `list[Transition]`

Reads pyglotaran k-matrices via `fill_item()` + `get_k_matrix()`. Each matrix entry becomes a `Transition` dataclass:

- **Off-diagonal** `(to, from)`: compartment-to-compartment transfer
- **Diagonal** `(comp, comp)`: ground state decay (generates a synthetic `"GS{n}"` label)

Key behaviors:

- `extract_transitions()` accepts explicit megacomplex labels
- `extract_dataset_transitions()` auto-discovers decay megacomplexes for a dataset (silently skips non-decay types)
- `omit_parameters` filters out specified parameter labels
- Rate constants are stored in raw ps^-1 units, never rounded at extraction

### Layer 2 -- Build Graph (`_kinetic_graph.py`)

**Input**: `list[Transition]`

**Output**: `KineticGraph`

Three dataclasses:

| Class          | Fields                                                                     | Role                    |
| -------------- | -------------------------------------------------------------------------- | ----------------------- |
| `KineticNode`  | `label`, `display_label`, `is_ground_state`, `megacomplex_labels`, `color` | Node in the scheme      |
| `KineticEdge`  | `source`, `target`, `rate_constant_ps_inverse`, `parameter_label`          | Directed edge with rate |
| `KineticGraph` | `nodes`, `edges`, `_adjacency`, `_reverse_adjacency`                       | Lightweight digraph     |

**Design decision: no networkx dependency.** Kinetic schemes are typically 3-20 nodes. A custom graph with adjacency lists keeps the dependency tree minimal and gives full control over the API.

`KineticGraph` methods:

- `successors()`, `predecessors()` -- adjacency queries
- `compartment_nodes()`, `ground_state_nodes()` -- filtered node lists
- `edges_between()`, `ground_state_edges_for_node()` -- edge queries
- `is_dag()` -- DFS-based cycle detection (WHITE/GRAY/BLACK coloring)
- `topological_sort()` -- Kahn's algorithm
- `from_transitions()` -- factory that merges duplicate GS decays and auto-creates nodes

`KineticEdge.format_rate()` handles display formatting:

- Unit conversion (ps^-1 to ns^-1 via `* 1e3`)
- Smart rounding (`>=1` -> integer, `<1` -> 2 decimals, or explicit `decimal_places`)
- Optional parameter label prefix (`show_label`)
- Optional unit suffix suppression (`include_unit=False`)

### Layer 3 -- Layout (`_layout.py`)

**Input**: `KineticGraph` + algorithm choice + spacing params

**Output**: `NodePositions` = `dict[str, tuple[float, float]]`

Three layout algorithms via `LayoutAlgorithm` enum:

| Algorithm                | When to use                                                 |
| ------------------------ | ----------------------------------------------------------- |
| `HIERARCHICAL` (default) | DAGs and simple cyclic schemes. Layered top-down layout     |
| `SPRING`                 | Complex cyclic schemes. Fruchterman-Reingold force-directed |
| `MANUAL`                 | Full user control. Validates and passes through             |

**Hierarchical layout pipeline:**

1. **Connected component detection** (`_find_connected_components`): BFS treating edges as undirected. Separate megacomplexes that share no compartments become separate components laid out side-by-side.

2. **Back-edge detection** (`_find_back_edges`): DFS cycle finder. Back edges are temporarily removed so the graph can be layered as a DAG.

3. **Layer assignment** (`_assign_layers`): Longest-path method. Source nodes (in-degree 0) get layer 0; each successor gets `parent_layer + 1`.

4. **Within-layer ordering** (`_order_within_layer`):
   - If `horizontal_layout_preference` is set (e.g. `"S2|S1|T1"`), uses that order.
   - Otherwise, barycenter heuristic: order by average position of predecessors to minimize crossings.

5. **Coordinate assignment**: Center each layer horizontally. Y = `-layer * vertical_spacing`.

6. **Ground state positioning** (`_position_ground_state_nodes`): GS nodes placed directly below their parent at `parent_y - ground_state_offset`.

7. **Multi-component placement**: Each component is laid out independently, then shifted so they appear side-by-side with `component_gap` spacing. Components are ordered by `horizontal_layout_preference` if provided.

**Spring layout**: Standard Fruchterman-Reingold with repulsive (all pairs) and attractive (edges) forces, simulated annealing, deterministic seed.

### Layer 4 -- Render (`plot_kinetic_scheme.py`)

**Input**: `KineticGraph` + `NodePositions` + `KineticSchemeConfig`

**Output**: matplotlib `(Figure, Axes)`

Rendering elements:

| Element           | Function                                                                            | z-order |
| ----------------- | ----------------------------------------------------------------------------------- | ------- |
| Ground state bars | `_draw_shared_ground_state_bar`, `_draw_per_megacomplex_ground_state_bars`          | 1       |
| Edge arrows       | `_draw_transfer_edge`, `_draw_ground_state_decay_arrow`, `_draw_ground_state_arrow` | 2       |
| Node rectangles   | `_draw_node` (via `FancyBboxPatch`)                                                 | 3       |
| Node labels       | `ax.text()` inside `_draw_node`                                                     | 4       |
| Rate labels       | `ax.text()` with white background bbox                                              | 5       |
| Unit annotation   | `ax.annotate()` in bottom-right corner                                              | --      |

Key rendering behaviors:

**Arrow endpoints**: `_compute_arrow_endpoints()` uses `_rect_edge_intersection()` to compute where the arrow exits/enters node rectangles, so arrows connect at boundaries rather than centers.

**Parallel edge curvature**: When edges exist in both directions (A->B and B->A), `edge_index` assigns alternating curvature via `arc3,rad=...` connection style.

**Label anti-overlap**: When multiple edges converge on the same target node, labels are spread along each edge at parametric positions t in [0.20, 0.50] instead of all clustering at t=0.35. Each label also has a perpendicular offset of 0.25 data units from the arrow line.

**Text contrast**: `_compute_text_color()` uses the W3C relative luminance formula on linearized sRGB to decide white vs. black text on node backgrounds.

**Unit annotation**: By default (`show_rate_unit_per_label=False`), the unit suffix (e.g., "ns^-1") is omitted from individual edge labels and shown once as an italic annotation in the bottom-right figure corner.

---

## Configuration Reference

### `KineticSchemeConfig` Fields

All fields have defaults. Config uses `extra="forbid"` to catch typos.

| Field                          | Type                                     | Default          | Purpose                             |
| ------------------------------ | ---------------------------------------- | ---------------- | ----------------------------------- |
| `node_styles`                  | `dict[str, NodeStyleConfig]`             | `{}`             | Per-node style overrides            |
| `color_mapping`                | `dict[str, list[str]]`                   | `{}`             | Batch color assignment              |
| `node_facecolor`               | `str`                                    | `"#4A90D9"`      | Default node fill                   |
| `node_edgecolor`               | `str`                                    | `"#2C3E50"`      | Default node border                 |
| `node_width`                   | `float`                                  | `1.2`            | Default node width                  |
| `node_height`                  | `float`                                  | `0.6`            | Default node height                 |
| `edge_color`                   | `str`                                    | `"#555555"`      | Arrow color                         |
| `edge_linewidth`               | `float`                                  | `1.5`            | Arrow thickness                     |
| `rate_fontsize`                | `int`                                    | `9`              | Rate label font size                |
| `rate_unit`                    | `"ps" \| "ns"`                           | `"ns"`           | Display unit for rates              |
| `rate_decimal_places`          | `int \| None`                            | `None`           | Fixed decimals (None = smart)       |
| `show_rate_labels`             | `bool`                                   | `False`          | Show parameter name prefix          |
| `show_rate_unit_per_label`     | `bool`                                   | `False`          | Unit on every label vs. legend      |
| `show_ground_state`            | `False \| "shared" \| "per_megacomplex"` | `False`          | Ground state bar mode               |
| `layout_algorithm`             | `str`                                    | `"hierarchical"` | Layout algorithm                    |
| `horizontal_layout_preference` | `str \| None`                            | `None`           | Left-to-right ordering hint         |
| `manual_positions`             | `dict \| None`                           | `None`           | For manual layout                   |
| `horizontal_spacing`           | `float`                                  | `2.0`            | Node horizontal gap                 |
| `vertical_spacing`             | `float`                                  | `1.5`            | Layer vertical gap                  |
| `ground_state_offset`          | `float`                                  | `1.2`            | GS bar vertical offset              |
| `component_gap`                | `float`                                  | `3.0`            | Gap between disconnected components |
| `figsize`                      | `tuple[float, float]`                    | `(10.0, 8.0)`    | Figure size in inches               |
| `title`                        | `str \| None`                            | `None`           | Plot title                          |
| `omit_parameters`              | `set[str]`                               | `set()`          | Parameters to exclude               |

### `NodeStyleConfig` Fields

| Field           | Type          | Default | Purpose                  |
| --------------- | ------------- | ------- | ------------------------ |
| `display_label` | `str \| None` | `None`  | Custom display name      |
| `width`         | `float`       | `1.2`   | Node width override      |
| `height`        | `float`       | `0.6`   | Node height override     |
| `facecolor`     | `str \| None` | `None`  | Fill color override      |
| `fontsize`      | `int`         | `10`    | Label font size override |

### Node Color Resolution Order

1. `NodeStyleConfig.facecolor` (per-node override)
2. `KineticNode.color` (set programmatically)
3. `KineticSchemeConfig.color_mapping` (batch assignment)
4. `KineticSchemeConfig.node_facecolor` (global default)

---

## File Map

```
pyglotaran_extras/inspect/kinetic_scheme/
  __init__.py              # Public API exports
  _constants.py            # Named defaults (dimensions, colors, thresholds)
  _k_matrix_parser.py      # Layer 1: Transition extraction
  _kinetic_graph.py        # Layer 2: Graph datastructure
  _layout.py               # Layer 3: Layout algorithms
  plot_kinetic_scheme.py   # Layer 4: Rendering + public functions
  devdocs/                 # This documentation
    architecture.md        # Architecture and design (this file)

tests/inspect/kinetic_scheme/
  test_k_matrix_parser.py  # Transition extraction tests
  test_kinetic_graph.py    # Graph construction, format_rate, DAG detection
  test_layout.py           # Layout algorithms, connected components
  test_plot_kinetic_scheme.py  # Rendering, config validation, integration
```

---

## Design Decisions

### Why no networkx?

Kinetic schemes in pyglotaran are small (3-20 nodes). A custom 100-line `KineticGraph` with adjacency/reverse-adjacency dicts provides all needed operations (successors, predecessors, cycle detection, topological sort) without pulling in networkx + scipy as dependencies.

### Why Pydantic for config?

`extra="forbid"` catches typos at construction time. Field defaults make the zero-config case trivial. Type validation ensures valid values without manual checks.

### Why connected component detection?

A single model often contains multiple independent megacomplexes (e.g., three reaction centers). Without component detection, all nodes get interleaved into one layout. BFS-based component detection renders them side-by-side with `component_gap` spacing.

### Why parametric t-value spreading?

When 4 edges converge on the same target node, all labels would cluster at `t=0.35` along their respective arrows. Spreading to `t in [0.20, 0.50]` naturally separates labels since each arrow has a different source position. Combined with perpendicular offset, this eliminates label overlap without force-based layout for text.

### Why unit-as-legend?

Showing "ns^-1" on every edge label adds visual clutter without information. A single italic annotation in the corner communicates the unit once. The `show_rate_unit_per_label=True` option restores per-label units for users who prefer them.

### Why ground state deduplication?

Multiple megacomplexes sharing a compartment may have identical diagonal k-matrix entries. `merge_ground_state_decays=True` (default) deduplicates these to avoid rendering duplicate GS decay arrows from the same node.

---

## Testing Strategy

251 tests across 4 test files. All tests use in-memory mock models (no I/O). Test categories:

- **Unit tests**: Individual functions (`format_rate`, `_compute_text_color`, `_rect_edge_intersection`, etc.)
- **Graph construction**: `from_transitions()` with various edge cases (empty, cycles, parallel edges, multi-megacomplex)
- **Layout correctness**: Layer assignment, ordering, component detection, ground state positioning
- **Rendering integration**: Full pipeline from transitions to matplotlib patches/artists
- **Config validation**: Pydantic `extra="forbid"` rejection, field defaults, field threading
- **Edge cases**: Single-node graphs, empty graphs, fully disconnected components, all-GS-decay models

Quality gates:

- `uv run pytest tests/inspect/kinetic_scheme/ -v` -- all pass
- `uv run mypy pyglotaran_extras/inspect/kinetic_scheme/` -- clean
- `uv run ruff check pyglotaran_extras/inspect/kinetic_scheme/` -- clean
- `uv run interrogate pyglotaran_extras/inspect/kinetic_scheme/` -- 100% docstring coverage
