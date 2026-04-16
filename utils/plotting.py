"""General matplotlib plotting primitives for experiment figures.

Shared across all experiments. Paper-specific labels, colors, and data
loading live in the experiment's stage scripts — this module provides
only reusable building blocks.

Matplotlib is imported lazily inside each function so headless compute
jobs that never plot don't pay the import cost.

Usage:
    from utils import plotting as plt_

    plt_.apply_style("ant")
    fig, axes = plt_.multi_panel(1, 2, figsize=(16, 7))
    plt_.scatter_with_regression(axes[0], x, y, annotate_r=True)
    plt_.save_figure(fig, "out.png")
"""

# ---------- Color/style constants ----------
TITLE_CORAL = "#c44e52"
STEEL_BLUE = "#4682B4"


# rcParams presets. Use `apply_style(name)` to activate.
STYLE_ANT = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

STYLE_DEFAULT = {
    "font.family": "sans-serif",
    "font.size": 11,
    "figure.facecolor": "white",
}

_PRESETS = {"ant": STYLE_ANT, "default": STYLE_DEFAULT}


def apply_style(preset: str = "ant", backend: str = "Agg"):
    """Apply an rcParams preset and lock matplotlib to a headless backend.

    Call once at top of a plotting script, BEFORE any plt.subplots()/figure()
    call that would otherwise lock the default backend. The Agg backend is
    required for deterministic rendering across Linux/Mac and headless GPUs
    — font metrics differ across GUI backends and produce different image
    sizes.

    Pass backend=None to skip the backend switch entirely (e.g. if you need
    interactive display in Jupyter).
    """
    import matplotlib
    import matplotlib.pyplot as plt
    if preset not in _PRESETS:
        raise ValueError(f"Unknown style preset {preset!r}; known: {list(_PRESETS)}")
    if backend is not None and matplotlib.get_backend().lower() != backend.lower():
        plt.switch_backend(backend)
    plt.rcParams.update(_PRESETS[preset])


# ---------- Figure creation / saving ----------

def multi_panel(nrows: int = 1, ncols: int = 1, figsize=None, **subplots_kwargs):
    """Create (fig, axes). Caller is responsible for setting `matplotlib.use('Agg')`
    (or equivalent) BEFORE any pyplot import if a headless backend is needed;
    by the time this is called, the backend is already locked in.

    Returns (fig, axes). When nrows*ncols == 1, `axes` is a single Axes;
    otherwise an ndarray of Axes.
    """
    import matplotlib.pyplot as plt
    return plt.subplots(nrows, ncols, figsize=figsize, **subplots_kwargs)


def save_figure(fig, path, dpi: int = 150, close: bool = True):
    """Save fig to path with consistent defaults (tight bbox, white bg)."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if close:
        plt.close(fig)
    return path


def suptitle(fig, title: str, color: str = TITLE_CORAL, fontsize: int = 20,
             fontweight: str = "bold", y: float = 0.98):
    """Standard coral suptitle used across paper figures."""
    fig.suptitle(title, fontsize=fontsize, fontweight=fontweight, color=color, y=y)


# ---------- Scatter primitives ----------

def scatter_with_regression(ax, x, y, *,
                             color: str = STEEL_BLUE,
                             s: int = 80,
                             alpha: float = 0.7,
                             edgecolor: str = "white",
                             show_identity: bool = False,
                             show_fit: bool = True,
                             annotate_r: bool = False,
                             r_loc: tuple = (0.05, 0.95),
                             r_fontsize: int = 14,
                             grid: bool = True,
                             identity_style: dict | None = None,
                             fit_style: dict | None = None):
    """Scatter plot with optional OLS regression line, identity line, r annotation.

    Args:
        x, y: 1D arrays of equal length.
        show_identity: draw dashed y=x line across data range.
        show_fit: draw a linear least-squares fit.
        annotate_r: add boxed Pearson r in the axes corner at r_loc.

    Returns:
        dict with keys 'r', 'slope', 'intercept' (None if show_fit=False).
    """
    import numpy as np
    from scipy import stats

    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0:
        raise ValueError("scatter_with_regression requires non-empty x, y")
    if len(x) != len(y):
        raise ValueError(f"x and y length mismatch: {len(x)} vs {len(y)}")

    ax.scatter(x, y, s=s, alpha=alpha, c=color, edgecolors=edgecolor, linewidths=0.5)

    out = {"r": None, "slope": None, "intercept": None}

    if len(x) >= 2:
        r, _ = stats.pearsonr(x, y)
        out["r"] = r

    if show_identity:
        style = dict(linestyle="--", color="#999", linewidth=0.8)
        if identity_style:
            style.update(identity_style)
        mn = float(min(x.min(), y.min()))
        mx = float(max(x.max(), y.max()))
        pad = (mx - mn) * 0.05
        ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], **style)

    if show_fit and len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        style = dict(linestyle="-", color="#333", linewidth=1.5, alpha=0.8)
        if fit_style:
            style.update(fit_style)
        ax.plot(xs, slope * xs + intercept, **style)
        out["slope"] = float(slope)
        out["intercept"] = float(intercept)

    if annotate_r and out["r"] is not None:
        annotate_r_value(ax, out["r"], loc=r_loc, fontsize=r_fontsize)

    if grid:
        ax.grid(True, alpha=0.15)

    return out


def labeled_scatter(ax, x, y, labels, *,
                     colors=None,
                     s: int = 80,
                     alpha: float = 0.85,
                     label_fontsize: int = 9,
                     label_color: str | None = None,
                     use_adjust_text: bool = True,
                     highlight: list | None = None):
    """Scatter with per-point text labels.

    Args:
        labels: list of strings, one per point.
        colors: single color string OR list of per-point colors (length must
                match). If a list of wrong length is given, raises ValueError.
        label_color: forced color for labels. If None, labels inherit the
                     per-point color when `colors` is a list, else use #333.
        highlight: if given, only labels in this list are drawn.
        use_adjust_text: if True, require adjustText and deconflict labels.
                         If adjustText is missing, raises ImportError with
                         install hint — pass use_adjust_text=False to skip.

    Returns:
        list of Text objects (the annotations actually drawn).
    """
    import numpy as np

    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) != len(y) or len(x) != len(labels):
        raise ValueError(
            f"x, y, labels length mismatch: {len(x)}/{len(y)}/{len(labels)}"
        )

    # Color handling: list must match length; string is fine as-is
    if isinstance(colors, (list, tuple)):
        if len(colors) != len(x):
            raise ValueError(
                f"colors list length {len(colors)} does not match data length {len(x)}"
            )
        color_arg = list(colors)
    else:
        color_arg = colors if colors is not None else STEEL_BLUE

    ax.scatter(x, y, s=s, alpha=alpha, c=color_arg, edgecolors="white", linewidths=0.5, zorder=2)

    def resolve_label_color(i):
        if label_color is not None:
            return label_color
        if isinstance(color_arg, list):
            return color_arg[i]
        return color_arg if isinstance(color_arg, str) else "#333"

    text_objs = []
    draw_set = set(highlight) if highlight is not None else None
    for i, lbl in enumerate(labels):
        if draw_set is not None and lbl not in draw_set:
            continue
        t = ax.annotate(str(lbl), (x[i], y[i]),
                        fontsize=label_fontsize, color=resolve_label_color(i),
                        alpha=0.9, xytext=(3, 3), textcoords="offset points")
        text_objs.append(t)

    if use_adjust_text and text_objs:
        try:
            from adjustText import adjust_text
        except ImportError as e:
            raise ImportError(
                "labeled_scatter(use_adjust_text=True) requires the 'adjustText' "
                "package. Install it with `pip install adjustText`, or pass "
                "use_adjust_text=False."
            ) from e
        adjust_text(text_objs, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="gray", alpha=0.3, lw=0.5),
                    only_move={"text": "xy"})

    return text_objs


def annotate_r_value(ax, r: float, *,
                     loc: tuple = (0.05, 0.95),
                     fontsize: int = 14,
                     text: str | None = None):
    """Boxed r-value annotation in axes-relative coords."""
    label = text if text is not None else f"r = {r:.2f}"
    ax.text(loc[0], loc[1], label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#999", alpha=0.9))


# ---------- Heatmap ----------

def heatmap(ax, matrix, *,
            cmap: str = "RdBu_r",
            vmin=None, vmax=None,
            row_labels=None, col_labels=None,
            tick_step: int = 1,
            tick_fontsize: int | None = None,
            xtick_rotation: float = 45,
            cbar: bool = True,
            cbar_label: str | None = None,
            cbar_shrink: float = 0.8,
            cbar_pad: float | None = None,
            cbar_label_fontsize: int | None = None,
            show_spines: bool = True,
            spine_linewidth: float | None = None,
            aspect: str = "equal",
            interpolation: str = "nearest"):
    """2D heatmap with optional colorbar.

    Args:
        matrix: 2D array-like.
        vmin, vmax: if both None, uses symmetric limits from |matrix| for diverging cmaps,
                    else matplotlib defaults.
        row_labels, col_labels: tick labels. Pass [] to hide ticks.
        tick_step: show every N-th label (default 1 = all).

    Returns:
        (image, colorbar_or_None).
    """
    import numpy as np

    mat = np.asarray(matrix)

    if vmin is None and vmax is None and cmap in ("RdBu_r", "RdBu", "bwr", "seismic", "coolwarm"):
        vmax = float(np.abs(mat).max())
        if vmax == 0:
            # Degenerate all-zero matrix — pick a small symmetric range so
            # imshow doesn't complain about vmin==vmax.
            vmax = 1.0
        vmin = -vmax

    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation=interpolation, aspect=aspect)

    tick_kw = {"fontsize": tick_fontsize} if tick_fontsize is not None else {}

    if col_labels is not None:
        if len(col_labels) == 0:
            ax.set_xticks([])
        else:
            idx = list(range(0, len(col_labels), tick_step))
            ax.set_xticks(idx)
            ax.set_xticklabels([col_labels[i] for i in idx],
                               rotation=xtick_rotation, ha="right", **tick_kw)

    if row_labels is not None:
        if len(row_labels) == 0:
            ax.set_yticks([])
        else:
            idx = list(range(0, len(row_labels), tick_step))
            ax.set_yticks(idx)
            ax.set_yticklabels([row_labels[i] for i in idx], **tick_kw)

    cb = None
    if cbar:
        fig = ax.figure
        cbar_kw = {"shrink": cbar_shrink}
        if cbar_pad is not None:
            cbar_kw["pad"] = cbar_pad
        cb = fig.colorbar(im, ax=ax, **cbar_kw)
        if cbar_label:
            if cbar_label_fontsize is not None:
                cb.set_label(cbar_label, fontsize=cbar_label_fontsize)
            else:
                cb.set_label(cbar_label)

    if show_spines:
        for spine in ax.spines.values():
            spine.set_visible(True)
            if spine_linewidth is not None:
                spine.set_linewidth(spine_linewidth)

    return im, cb


# ---------- Line plots ----------

def line_plot(ax, x, series: dict, *,
              colors: dict | None = None,
              linestyles: dict | None = None,
              linewidth: float = 2.5,
              markersize: float = 6,
              marker: str = "o",
              zero_line: bool = True,
              grid: bool = True,
              legend: bool = True,
              legend_kwargs: dict | None = None,
              endpoint_labels: bool = False):
    """Multi-series line plot over shared x-axis.

    Works for both layer-indexed (integer layers) and token-indexed
    (position labels) plots — the caller supplies the tick labels.

    Args:
        x: shared x-values (numeric or list of strings).
        series: dict mapping series_name -> list of y-values.
        colors: dict mapping series_name -> color (optional).
        linestyles: dict mapping series_name -> linestyle (e.g. '--' for dashed).
        endpoint_labels: annotate each series at its rightmost point.
    """
    colors = colors or {}
    linestyles = linestyles or {}
    for name, y in series.items():
        c = colors.get(name)
        ls = linestyles.get(name, "-")
        line, = ax.plot(x, y, linestyle=ls, marker=marker, color=c,
                        linewidth=linewidth, markersize=markersize, label=name, zorder=3)
        if endpoint_labels:
            ax.annotate(name, xy=(len(x) - 1 if not hasattr(x[-1], "__sub__") else x[-1], y[-1]),
                        xytext=(5, 0), textcoords="offset points",
                        fontsize=11, color=line.get_color(), fontweight="bold", va="center")

    if zero_line:
        ax.axhline(0, color="#ccc", linewidth=0.8, zorder=0)
    if grid:
        ax.grid(True, alpha=0.15)
    if legend:
        ax.legend(**(legend_kwargs or {}))


# ---------- Bar chart ----------

def bar_chart(ax, names, values, *,
              horizontal: bool = False,
              colors=None,
              width: float = 0.8,
              alpha: float = 0.85,
              zero_line: bool = True,
              label_fontsize: int = 10,
              tick_step: int = 1,
              xtick_rotation: float = 45):
    """Bar chart (vertical by default, horizontal if requested).

    Args:
        names: list of category labels.
        values: list of bar heights (or lengths).
        colors: single color string OR list of per-bar colors.
        tick_step: show every N-th label (only for vertical bars).
    """
    n = len(names)
    positions = list(range(n))
    color_arg = colors if colors is not None else STEEL_BLUE

    if horizontal:
        ax.barh(positions, values, color=color_arg, alpha=alpha, height=width)
        ax.set_yticks(positions)
        ax.set_yticklabels(names, fontsize=label_fontsize)
        if zero_line:
            ax.axvline(0, color="black", linewidth=0.5)
    else:
        ax.bar(positions, values, color=color_arg, width=width, edgecolor="none", alpha=alpha)
        idx_set = set(range(0, n, tick_step))
        labels = [names[i] if i in idx_set else "" for i in range(n)]
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=xtick_rotation, ha="right", fontsize=label_fontsize)
        if zero_line:
            ax.axhline(0, color="black", linewidth=0.5)


# ---------- Dumbbell chart (paired points with connector) ----------

def dumbbell_chart(ax, names, left_values, right_values, *,
                    left_color: str = "#1f77b4",
                    right_color: str = "#d62728",
                    connector_color: str = "#cccccc",
                    marker_size: int = 100,
                    label_fontsize: int = 10,
                    left_label: str = "Left",
                    right_label: str = "Right",
                    legend: bool = True):
    """Horizontal dumbbell chart: pairs of points with connecting line per category."""
    import numpy as np

    n = len(names)
    y = np.arange(n)

    for i in range(n):
        ax.plot([left_values[i], right_values[i]], [y[i], y[i]],
                color=connector_color, linewidth=2, zorder=1)
    ax.scatter(left_values, y, s=marker_size, color=left_color,
               zorder=3, label=left_label, edgecolors="white", linewidths=0.5)
    ax.scatter(right_values, y, s=marker_size, color=right_color,
               zorder=3, label=right_label, edgecolors="white", linewidths=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=label_fontsize)
    ax.grid(axis="x", alpha=0.15)

    if legend:
        ax.legend(loc="best", fontsize=label_fontsize + 1)
