# Improve Edge Label Readability

## Context

The kinetic scheme visualizer renders rate constant labels on every edge. Three readability issues were identified:

1. **Unit clutter**: Every label shows "ns⁻¹" (e.g., `"500 ns⁻¹"`) — the unit should appear once as a small legend annotation, not repeated on every edge.
2. **Font size too small**: `DEFAULT_RATE_FONTSIZE = 8` is too small relative to node labels (`DEFAULT_FONTSIZE = 10`). Should be closer to node label size.
3. **Labels overlap**: When multiple edges converge on the same target (e.g., 4 edges from R1-R4 all targeting `rcg_g`), labels cluster at similar positions. Need parametric t-value spreading so each label sits at a distinct point along its edge.

---

## Changes

### 1. `_constants.py` — Increase rate fontsize

Change `DEFAULT_RATE_FONTSIZE` from `8` to `9`.

### 2. `_kinetic_graph.py` — Add `include_unit` parameter to `format_rate()`

Add `include_unit: bool = True` parameter. When `False`, the formatted string omits the unit suffix (e.g., `"500"` instead of `"500 ns⁻¹"`). Default `True` preserves backward compatibility for all existing callers.

```python
if include_unit:
    result = f"{formatted} {unit_suffix}"
else:
    result = formatted
```

### 3. `plot_kinetic_scheme.py` — Three changes

#### 3a. Config additions

Add two fields to `KineticSchemeConfig`:

```python
rate_fontsize: int = DEFAULT_RATE_FONTSIZE
show_rate_unit_per_label: bool = False
```

- `rate_fontsize`: lets users override the rate label font size
- `show_rate_unit_per_label`: `False` (new default) = show unit once as legend annotation; `True` = old behavior (unit on every label)

#### 3b. Unit legend annotation

Add helper `_get_rate_unit_annotation(unit)` returning `"rates in ns⁻¹"`.

In `_render_kinetic_scheme()`, after drawing edges and nodes, add:

```python
if not config.show_rate_unit_per_label:
    ax.annotate(
        _get_rate_unit_annotation(config.rate_unit),
        xy=(1.0, 0.0), xycoords="axes fraction",
        ha="right", va="bottom",
        fontsize=config.rate_fontsize,
        color="#666666", fontstyle="italic",
    )
```

#### 3c. Label anti-overlap via t-value spreading

In `_draw_all_edges()`, pre-compute per-target convergence counts, then pass `convergence_index` and `convergence_total` to `_draw_transfer_edge()`.

In `_draw_transfer_edge()`, replace fixed `t = 0.35` with:

```python
if convergence_total <= 1:
    t = 0.35
else:
    t = 0.20 + (convergence_index - 1) * 0.30 / (convergence_total - 1)
```

This spreads labels evenly across t ∈ [0.20, 0.50] when edges converge. For 4 convergent edges → t = 0.20, 0.30, 0.40, 0.50. Labels naturally separate along each edge's path.

#### 3d. Thread config through all three rendering functions

Replace hardcoded `fontsize=DEFAULT_RATE_FONTSIZE` with `fontsize=config.rate_fontsize` in `_draw_transfer_edge()`, `_draw_ground_state_decay_arrow()`, and `_draw_ground_state_arrow()`.

Pass `include_unit=config.show_rate_unit_per_label` to all three `edge.format_rate()` calls.

### 4. Tests

**`test_kinetic_graph.py`** — Add tests for `include_unit=False`:
- `test_include_unit_false`: Output has no unit suffix
- `test_include_unit_false_with_label`: Parameter prefix still works without unit

**`test_plot_kinetic_scheme.py`** — Add:
- `test_rate_fontsize_config`: Custom value accepted
- `test_unit_annotation_present_by_default`: "rates in" text appears in axes
- `test_no_unit_annotation_when_per_label`: No annotation when `show_rate_unit_per_label=True`
- `test_convergent_labels_at_distinct_positions`: Labels on edges sharing a target have distinct positions

---

## Files Modified

| File | Change |
|------|--------|
| `_constants.py` | `DEFAULT_RATE_FONTSIZE` 8 → 9 |
| `_kinetic_graph.py` | Add `include_unit` param to `format_rate()` |
| `plot_kinetic_scheme.py` | Add `rate_fontsize` + `show_rate_unit_per_label` config fields, unit annotation, convergence tracking + t-spreading, thread config fontsize |
| `test_kinetic_graph.py` | Tests for `include_unit=False` |
| `test_plot_kinetic_scheme.py` | Tests for annotation, fontsize config, label positions |
| `docs/notebooks/kinetic_scheme_visualization.ipynb` | Update config reference table |

---

## Verification

1. `uv run pytest tests/inspect/kinetic_scheme/ -v` — all tests pass
2. `uv run mypy pyglotaran_extras/inspect/kinetic_scheme/` — clean
3. `uv run ruff check pyglotaran_extras/inspect/kinetic_scheme/` — clean
4. `uv run interrogate pyglotaran_extras/inspect/kinetic_scheme/` — 100%
5. `uv run pytest tests/ -v` — full suite, no regressions
6. Visual: Re-run notebook with 3-triad model — labels should be larger, no unit clutter, no overlap
