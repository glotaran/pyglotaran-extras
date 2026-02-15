# Fix Visual Quality: Side-by-Side Components + Rate Label Overlap

## Context

The kinetic scheme visualizer is fully implemented and all 84 tests pass. However, when rendering multiple megacomplexes (e.g., the 3 triads from `target_rcg_gcrcg_rcgcr_refine.yml`), two visual problems appear:

1. **Interleaved layout**: All 18 compartment nodes from 3 disconnected megacomplexes get pooled into a single hierarchical layout, interleaving nodes from different megacomplexes across the same rows. The old implementation rendered each megacomplex as a visually separate group.

2. **Rate label overlap**: When multiple edges converge on the same target node (e.g., 4 edges from R1-R4 all targeting `rcg_g`), their midpoint-placed labels pile up on top of each other.

**Root cause analysis**:
- The 3 megacomplexes produce 3 completely disconnected graph components (no shared nodes). The layout treats them as one graph.
- Rate labels are placed at t=0.5 (midpoint) with only 0.15 perpendicular offset, insufficient for converging edges.

---

## Changes

### 1. `_constants.py` — Add component gap constant

Add `DEFAULT_COMPONENT_GAP = 3.0` after the existing layout defaults.

### 2. `_layout.py` — Connected component detection + per-component layout

**New function `_find_connected_components()`**:
- BFS on compartment nodes using the graph's undirected connectivity (successors + predecessors, ignoring GS nodes)
- Returns `list[set[str]]` — each set is a connected component's node labels
- Components sorted alphabetically by first label for determinism

**Refactor `_hierarchical_layout()`**:
- Extract current logic into `_layout_single_component(graph, component_labels, ...)`
- New `_hierarchical_layout()` calls `_find_connected_components()`, lays out each component independently via `_layout_single_component()`, then places components side-by-side with `component_gap` spacing between them
- Each component is individually centered vertically (top-aligned)
- Components are ordered alphabetically by their first node label (deterministic)

**Update `compute_layout()` signature**:
- Add `component_gap: float = DEFAULT_COMPONENT_GAP` parameter
- Pass through to `_hierarchical_layout()`

### 3. `plot_kinetic_scheme.py` — Rate label positioning + config

**`KineticSchemeConfig`**:
- Add `component_gap: float = DEFAULT_COMPONENT_GAP` field (in the Layout section, after `ground_state_offset`)
- Docstring updated accordingly

**`_draw_transfer_edge()` (line 738-768)** — Fix rate label overlap:
- Change parametric position from t=0.5 (midpoint) to t=0.35 (closer to source)
- Increase perpendicular offset from 0.15 to 0.25
- This spreads labels apart when multiple edges converge on the same target

**`show_kinetic_scheme()` and `show_dataset_kinetic_scheme()`**:
- Pass `config.component_gap` to `compute_layout()`

### 4. Tests — `test_layout.py`

**New test helpers**:
- `_make_two_component_graph()`: Two disconnected components (A→B→C, D→E→F)
- `_make_three_component_graph()`: Three disconnected components

**New test class `TestConnectedComponentLayout`**:
- `test_two_components_separated`: Components should have non-overlapping x ranges
- `test_three_components_separated`: All 3 components have non-overlapping x ranges
- `test_single_component_unchanged`: Single component layout unaffected
- `test_component_gap_respected`: The gap between rightmost node of component 1 and leftmost of component 2 is approximately `component_gap`

**New test class `TestFindConnectedComponents`** (testing `_find_connected_components` directly):
- `test_single_component`: Connected graph returns 1 component
- `test_disconnected_components`: Disconnected graph returns correct number of components
- `test_empty_graph`: Returns empty list

---

## Files Modified

| File | Change |
|------|--------|
| `pyglotaran_extras/inspect/kinetic_scheme/_constants.py` | Add `DEFAULT_COMPONENT_GAP` |
| `pyglotaran_extras/inspect/kinetic_scheme/_layout.py` | Add `_find_connected_components()`, refactor `_hierarchical_layout()`, update `compute_layout()` |
| `pyglotaran_extras/inspect/kinetic_scheme/plot_kinetic_scheme.py` | Add `component_gap` to config, fix label positioning, pass gap to layout |
| `tests/inspect/kinetic_scheme/test_layout.py` | Add component layout tests |

---

## Verification

1. `uv run pytest tests/inspect/kinetic_scheme/ -v` — all existing + new tests pass
2. `uv run mypy pyglotaran_extras/inspect/kinetic_scheme/` — clean
3. `uv run ruff check pyglotaran_extras/inspect/kinetic_scheme/` — clean
4. `uv run interrogate pyglotaran_extras/inspect/kinetic_scheme/` — 100%
5. `uv run pytest tests/ -v` — full suite, no regressions
6. Visual verification: Re-run the notebook cells with the 3-triad model to confirm side-by-side rendering
