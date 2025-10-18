import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.font_manager as font_manager
from typing import Optional, Dict, List, Tuple, Union
import arviz as az

def plot_kde(
    idata: az.InferenceData,
    var_list: Tuple[str, ...] = ("fold_W", "fold_E"),
    colors: Tuple[str, ...] = ("C0", "C2"),
    figsize: Tuple[int, int] = (9, 5),
    axis_labels: Optional[List[str]] = None,  # [x_label, y_label]
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    ci_level: float = 95,
    font_size: int = 12,
    ci_shift_num: float = 0.33,
    ci_shift_factor: float = 0.51,
    ci_shift_low: float = 1.08,
    ci_shift_high: float = 1.05,
    axis_font_size: int = 12,
    legend_font_size: int = 12,
    y_ticks: Optional[List[float]] = None,
    x_ticks: Optional[List[float]] = None,
    legend_position: str = "outside_bottom",  # options: outside_bottom, outside_top, inside, left, right
    legend_labels: Optional[List[str]] = None,  # labels to show in legend (must match var_list length)
    legend_pad: float = 0.05,  # small padding to nudge legend (positive moves legend away from plot)
    legend_ncol: Optional[int] = None,
    legend_bold: bool = True,
    font_family: str = "Arial",
    legend_linewidth: Optional[float] = None,
    show_title: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    KDE plot for one figure with multiple overlayed variables and flexible legend placement.
    Returns (fig, ax).
    """
    if axis_labels is None:
        axis_labels = ["Fold Ratio", "Probability Density"]

    # helper to pull draws robustly
    def get_draws2(idata_: az.InferenceData, var: str) -> np.ndarray:
        try:
            if hasattr(idata_, "posterior") and (var in idata_.posterior):
                arr = idata_.posterior[var].values
                return arr.reshape(-1)
            if isinstance(idata_, dict) and (var in idata_):
                return np.asarray(idata_[var]).reshape(-1)
        except Exception:
            pass
        warnings.warn(f"Variable '{var}' not found in idata; skipping.", UserWarning)
        return np.array([])

    fig, ax = plt.subplots(figsize=figsize)

    kde_list = []
    peak_list = []
    # compute draws and kdes for variables that exist
    for v in var_list:
        draws = get_draws2(idata, v)
        if draws.size == 0:
            kde_list.append((v, draws, None, None, None))
            peak_list.append(0.0)
            continue
        kde = gaussian_kde(draws)
        x_min, x_max = np.min(draws), np.max(draws)
        # small padding to avoid zero-width when draws identical (guard)
        if x_max <= x_min:
            x_min -= 0.01 * max(1.0, abs(x_min))
            x_max += 0.01 * max(1.0, abs(x_max))
        x_vals = np.linspace(x_min, x_max, 500)
        y_vals = kde(x_vals)
        kde_list.append((v, draws, kde, x_vals, y_vals))
        peak_list.append(float(np.max(y_vals)))

    if all([d.size == 0 for (_, d, *_ ) in kde_list]):
        ax.text(0.5, 0.5, "no draws available", ha="center", va="center")
        return fig, ax

    max_peak_idx = int(np.argmax(peak_list)) if len(peak_list) > 0 else 0

    plot_handles = []
    plotted_labels = []

    # choose legend_ncol default: 1 if <=2 items otherwise min(3,n_items)
    n_items = sum(1 for v, d, *_ in kde_list if d.size > 0)
    if legend_ncol is None:
        legend_ncol = 1 if n_items <= 2 else min(3, n_items)

    for idx, (v, draws, kde, x_vals, y_vals) in enumerate(kde_list):
        if draws.size == 0:
            continue
        c = colors[idx % len(colors)]
        med = float(np.median(draws))
        low = float(np.percentile(draws, (100 - ci_level) / 2))
        high = float(np.percentile(draws, 100 - (100 - ci_level) / 2))
        peak = peak_list[idx]

        # plot line and fill
        line, = ax.plot(x_vals, y_vals, color=c, alpha=0.85, label=(legend_labels[idx] if legend_labels and idx < len(legend_labels) else v))
        ax.fill_between(x_vals, y_vals, color=c, alpha=0.30)

        # vertical median dashed
        ax.axvline(med, color=c, linestyle='--', linewidth=1.2)

        # choose y-level for CI horizontal:
        # use peak for non-primary curves, half-peak for primary so they don't overlap
        if idx == max_peak_idx:
            ci_y = peak * 0.5 if peak > 0 else ax.get_ylim()[1] * 0.5
        else:
            ci_y = peak if peak > 0 else ax.get_ylim()[1] * 0.8

        # enforce y_range constraints if provided
        if y_range is not None:
            ymin, ymax = y_range
            ci_y = np.clip(ci_y, ymin + 1e-12, ymax - 1e-12)

        # horizontal CI line and open circle median
        ax.hlines(ci_y, xmin=low, xmax=high, color=c, linewidth=2)
        ax.plot(med, ci_y, marker='o', markerfacecolor='white', markeredgecolor=c, markersize=7, markeredgewidth=1.4)

        # small horizontal shift for very narrow intervals
        label_shift = 0.0
        if (high - low) < ci_shift_factor:
            label_shift = (high - low) * ci_shift_num
            label_med_factor = ci_shift_low
        else:
            label_med_factor = ci_shift_high

        # number formatting
        if med > 100:
            prefix = "£"
            num_format = "{:,.0f}"
        else:
            prefix = ""
            num_format = "{:,.2f}"

        def format_val(val: float) -> str:
            return f"{prefix}{num_format.format(val)}"

        # place numeric labels on/near the horizontal CI line:
        # low/high on the horizontal line (va='bottom' so they sit slightly above the line)
        ax.text(low - label_shift, ci_y, format_val(low), color="black",
                fontsize=font_size, ha='center', va='bottom')
        ax.text(high + label_shift, ci_y, format_val(high), color="black",
                fontsize=font_size, ha='center', va='bottom')
        # median slightly above the horizontal line (and bold + black)
        ax.text(med, ci_y * label_med_factor, format_val(med),
                color="black", fontsize=font_size, ha='center', va='bottom', fontweight='bold')

        plot_handles.append(line)
        plotted_labels.append(legend_labels[idx] if legend_labels and idx < len(legend_labels) else v)

    # axes labels and ticks
    ax.set_xlabel(axis_labels[0], fontsize=axis_font_size, fontweight='bold', family=font_family)
    ax.set_ylabel(axis_labels[1], fontsize=axis_font_size, fontweight='bold', family=font_family)
    ax.tick_params(axis='both', labelsize=axis_font_size, which='both')

    if x_range is not None:
        ax.set_xlim(x_range)
    if y_range is not None:
        ax.set_ylim(y_range)

    # default y-ticks
    if y_ticks is None:
        y_max = ax.get_ylim()[1]
        raw_ticks = np.linspace(0, y_max, num=5)
        def round_tick(t):
            if y_max >= 10:
                return round(t)
            else:
                return round(t, 1)
        y_ticks = [round_tick(t) for t in raw_ticks]
    ax.set_yticks(y_ticks)

    if x_ticks is not None:
        ax.set_xticks(x_ticks)

    # --- Legend handling ---
    # create proxy handles so we can control linewidth in legend easily
    legend_handles = []
    for idx, h in enumerate(plot_handles):
        c = colors[idx % len(colors)]
        lw = legend_linewidth if legend_linewidth is not None else max(2.0, plt.rcParams.get("lines.linewidth", 1.5))
        legend_handles.append(Line2D([0], [0], color=c, lw=lw, marker=None))

    if len(legend_handles) > 0:
        font_props = font_manager.FontProperties(
            family=font_family,
            weight='bold' if legend_bold else 'normal',
            size=legend_font_size
        )

        # conservative margins so the plot is not squashed
        # these values work well for many setups; tweak if needed
        if legend_position == "outside_bottom":
            fig.subplots_adjust(bottom=0.28 + legend_pad)
            bbox = (0.5, -0.12 - legend_pad)
            legend = ax.legend(handles=legend_handles, labels=plotted_labels,
                               loc='upper center', bbox_to_anchor=bbox,
                               ncol=legend_ncol, prop=font_props, frameon=False)
        elif legend_position == "outside_top":
            fig.subplots_adjust(top=0.72 - legend_pad)
            bbox = (0.5, 1.08 + legend_pad)
            legend = ax.legend(handles=legend_handles, labels=plotted_labels,
                               loc='lower center', bbox_to_anchor=bbox,
                               ncol=legend_ncol, prop=font_props, frameon=False)
        elif legend_position == "right":
            fig.subplots_adjust(right=0.72 + legend_pad)
            bbox = (1.02 + legend_pad, 0.5)
            legend = ax.legend(handles=legend_handles, labels=plotted_labels,
                               loc='center left', bbox_to_anchor=bbox,
                               ncol=1, prop=font_props, frameon=False)
        elif legend_position == "left":
            fig.subplots_adjust(left=0.22 - legend_pad)
            bbox = (-0.02 - legend_pad, 0.5)
            legend = ax.legend(handles=legend_handles, labels=plotted_labels,
                               loc='center right', bbox_to_anchor=bbox,
                               ncol=1, prop=font_props, frameon=False)
        else:  # inside
            legend = ax.legend(handles=legend_handles, labels=plotted_labels,
                               loc='best', ncol=legend_ncol, prop=font_props, frameon=False)

        # lift legend zorder
        try:
            legend.set_zorder(10)
        except Exception:
            pass

    # optional title
    if show_title:
        ax.set_title("Posterior densities with confidence intervals",
                     fontsize=axis_font_size, fontweight='bold', pad=12, family=font_family)

    fig.tight_layout()
    #return fig, ax




def plot_loglog_ccdf(
    sim_dict: dict,
    groups_to_plot: Optional[list[str]] = None,
    marginal_label: str = "Posterior Predictive",
    figsize: tuple[int, int] = (6, 4),
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    title_fontsize: int = 12,
    axis_fontsize: int = 12,
    xtick_fontsize: int = 12,
    ytick_fontsize: int = 12,
    legend_position: str = "outside_bottom",  # options: outside_bottom, outside_top, inside, left, right
    legend_fontsize: int = 10,
    colors: Optional[list[str]] = None,
    linestyles: Optional[list[str]] = None,
    rename_legend: Optional[dict[str, str]] = None,
) -> plt.Figure:
    """
    Plot log-log CCDF curves with flexible legend placement.
    """
    if groups_to_plot is None:
        groups_to_plot = list(sim_dict.keys())

    if colors is None:
        colors = ["navy", "navy", "darkgreen", "darkgreen"]
    if linestyles is None:
        linestyles = ["-", "dotted", "-", "dotted"]

    fig, ax = plt.subplots(figsize=figsize)

    for i, group in enumerate(groups_to_plot):
        arr = np.array(sim_dict[group])
        arr = arr[np.isfinite(arr)]
        arr = arr[arr > 0]
        if arr.size == 0:
            continue

        arr_sorted = np.sort(arr)
        ccdf = 1.0 - np.arange(1, len(arr_sorted) + 1) / float(len(arr_sorted))

        label = rename_legend[group] if rename_legend and group in rename_legend else group

        ax.loglog(
            arr_sorted,
            ccdf,
            label=label,
            linewidth=2.0,
            linestyle=linestyles[i % len(linestyles)],
            color=colors[i % len(colors)],
        )

    ax.set_xlabel("Individual-level cost (£)", fontsize=axis_fontsize)
    ax.set_ylabel("Complementary CDF (1 - F(x))", fontsize=axis_fontsize)

    title = "Log–Log CCDF"
    if marginal_label:
        title += f"  ({marginal_label})"
    ax.set_title(title, fontsize=title_fontsize)

    ax.tick_params(axis='x', labelsize=xtick_fontsize)
    ax.tick_params(axis='y', labelsize=ytick_fontsize)
    ax.grid(which="major", linestyle=":", linewidth=0.7, alpha=0.7)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Legend placement
    if legend_position.startswith("outside"):
        side = legend_position.split("_")[1]
        if side in ["bottom", "top"]:
            fig.subplots_adjust(bottom=0.25 if side=="bottom" else 0.75)
            fig.legend(
                loc="lower center" if side=="bottom" else "upper center",
                bbox_to_anchor=(0.5, -0.15) if side=="bottom" else (0.5, 1.15),
                ncol=len(groups_to_plot),
                fontsize=legend_fontsize,
            )
        elif side in ["left", "right"]:
            fig.subplots_adjust(left=0.25 if side=="left" else 0.75)
            fig.legend(
                loc="center left" if side=="left" else "center right",
                bbox_to_anchor=(-0.15, 0.5) if side=="left" else (1.15, 0.5),
                ncol=1,
                fontsize=legend_fontsize,
            )
    else:
        # inside the axes
        ax.legend(loc=legend_position, fontsize=legend_fontsize)

    fig.tight_layout()
    #return fig




def _extract_draws(idata_like, var="ratio"):
    """Extract flattened draws for `var` from InferenceData or dict."""
    if idata_like is None:
        return np.array([])
    if isinstance(idata_like, dict):
        return np.asarray(idata_like.get(var, np.array([]))).reshape(-1)
    if hasattr(idata_like, "posterior") and (var in idata_like.posterior):
        return idata_like.posterior[var].values.reshape(-1)
    try:
        return np.asarray(getattr(idata_like, var)).reshape(-1)
    except Exception:
        return np.array([])


def plot_kde_grid(
    idata_list,
    var="ratio",
    figsize=(5, 12),
    xticks=None,
    yticks=None,           # NEW parameter
    x_pct_range=(0.5, 99.5),
    n_x=500,
    sharex=True,
    ci=(2.5, 97.5),
    color="C0",
    font_size=11,
    ci_font_size=9,
    bold_median=True,
    lab_fact=0.995,
    hline_pos="mid",        # "mid" or "top"
    hline_top_frac=1.0,     # 1.0 = peak, <1 = below peak
    num_color="black",      # color for CI/median numbers
    vline_at_1=False        # NEW: add red dotted vertical line at x==1
):
    """
    Stacked KDE plots (n x 1) for variable `var`, aligned x-axis ticks.

    - Same color for all plots
    - CI lines + median, annotated with numbers
    - Horizontal CI line + circle at median:
        hline_pos = "mid" → y at plot mid
        hline_pos = "top" → y at (hline_top_frac * max KDE)
    - yticks can be set manually
    - Optional red dotted line at x=1
    """
    n = len(idata_list)
    draws_list = [_extract_draws(idata, var=var) for idata in idata_list]

    # Global x-range across all methods
    all_draws = np.concatenate([d for d in draws_list if d.size > 0])
    p_lo, p_hi = np.percentile(all_draws, x_pct_range)
    width = p_hi - p_lo
    p_lo -= 0.05 * width
    p_hi += 0.05 * width
    x_vals = np.linspace(p_lo, p_hi, n_x)

    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=sharex)
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        draws = draws_list[i]
        if draws.size == 0:
            ax.text(0.5, 0.5, "no draws", ha="center", va="center")
            continue

        # KDE
        kde = gaussian_kde(draws)
        y = kde(x_vals)

       
        ax.plot(x_vals, y, color=color, linewidth=1.6)
        ax.fill_between(x_vals, 0, y, alpha=0.2, color=color)

        # CI + median
        med = np.median(draws)
        low, high = np.percentile(draws, ci)

        ax.axvline(low, color="k", linestyle=":", linewidth=1)
        ax.axvline(high, color="k", linestyle=":", linewidth=1)
        ax.axvline(med, color="k", linestyle="--", linewidth=1)

        # Optional reference line at x==1
        if vline_at_1:
            ax.axvline(1, color="red", linestyle=":", linewidth=1)

        # Horizontal CI line + open circle at median
        if hline_pos == "mid":
            y_hline = ax.get_ylim()[1] * 0.5
        elif hline_pos == "top":
            y_hline = max(y) * hline_top_frac
        else:
            raise ValueError("hline_pos must be 'mid' or 'top'")

        ax.hlines(y_hline, low, high, color="k", linestyle="-", linewidth=1)
        ax.plot(med, y_hline, "o", color="k", markersize=6,
                markerfacecolor="white")

        # Format values
        def format_val(val):
            if val > 100:
                return f"£{int(round(val))}"
            else:
                return f"{val:.2f}"

        # Labels at y_hline (median slightly above)
        ax.text(low, y_hline, format_val(low), ha="center", va="bottom",
                fontsize=ci_font_size, color=num_color)
        ax.text(high, y_hline, format_val(high), ha="center", va="bottom",
                fontsize=ci_font_size, color=num_color)

        weight = "bold" if bold_median else "normal"
        ax.text(med, y_hline * 1.05, format_val(med), ha="center", va="bottom",
                fontsize=ci_font_size, weight=weight, color="black")

        ax.set_ylabel("Density", fontsize=font_size)
        ax.tick_params(axis="both", labelsize=font_size - 1)
        ax.grid(axis="x", linestyle=":", alpha=0.6)
        ax.set_xlim(p_lo, p_hi)

        # Apply yticks if provided
        if yticks is not None:
            ax.set_yticks(yticks)

    # Shared x-axis
    if xticks is not None:
        axes[-1].set_xticks(xticks)
    axes[-1].set_xlabel(var, fontsize=font_size)
    #fig.tight_layout()
    return fig, axes

def plot_kde_grid_overlay(
    idata_list,
    var_list=("ratio",),
    figsize=(6, 12),
    xticks=None,
    yticks=None,
    x_pct_range=(0.5, 99.5),
    n_x=500,
    sharex=True,
    ci=(2.5, 97.5),
    colors=None,
    linestyles=None,
    color_alpha=0.25,
    font_size=11,
    ci_font_size=9,
    bold_median=True,
    ci_shift_factor: float = 0.51,
    ci_shift_num: float = 0.33,
    label_vpad_frac: float = 0.03,
    horiz_peak_frac: float = 0.95,
    median_marker_size: int = 7,
    format_currency_threshold: float = 100.0,
    keep_labels_level: bool = False,      # overridden by median_vshift_frac
    median_vshift_frac: float = 0.015,    # vertical lift for median label
    ci_label_color: str = "navy",      # color for CI bounds labels
    median_label_color: str = "black",    # color for median label
):
    """
    Stacked KDE plots with overlays.

    - Each subplot overlays all variables in var_list.
    - CI line + vertical markers + open circle at median.
    - Labels: low/high on same line, median shown slightly above.
    - Median label is bold and (by default) black.
    """
    if colors is None:
        colors = plt.rcParams.get("axes.prop_cycle").by_key().get("color", ["C0", "C1", "C2", "C3"])
    if linestyles is None:
        linestyles = ["-"] * max(1, len(var_list))

    # build draws matrix
    draws_matrix = [[_extract_draws(idata, var=v) for v in var_list] for idata in idata_list]

    # global x-range
    all_draws = np.concatenate([d for row in draws_matrix for d in row if d.size > 0]) \
                if any(d.size > 0 for row in draws_matrix for d in row) else np.array([0.0, 1.0])
    if all_draws.size == 0:
        p_lo, p_hi = 0.0, 1.0
    else:
        p_lo, p_hi = np.percentile(all_draws, x_pct_range)
        width = (p_hi - p_lo) if (p_hi - p_lo) > 0 else max(abs(p_hi), 1.0)
        p_lo -= 0.05 * width
        p_hi += 0.05 * width
    x_vals = np.linspace(p_lo, p_hi, n_x)

    n_plots = len(idata_list)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=sharex)
    if n_plots == 1:
        axes = [axes]

    # KDE + peaks
    kde_cache = []
    peak_cache = []
    for row in draws_matrix:
        row_kde = []
        row_peak = []
        for draws in row:
            if draws.size == 0:
                row_kde.append((None, None))
                row_peak.append(0.0)
                continue
            kde = gaussian_kde(draws)
            y = kde(x_vals)
            row_kde.append((kde, y))
            row_peak.append(float(np.max(y)))
        kde_cache.append(row_kde)
        peak_cache.append(row_peak)

    # plotting
    for i, ax in enumerate(axes):
        row_draws, row_kdes, row_peaks = draws_matrix[i], kde_cache[i], peak_cache[i]

        local_peak = max(row_peaks) if row_peaks else 1.0
        ax.set_ylim(0.0, local_peak * 1.20)

        for j, var in enumerate(var_list):
            draws = row_draws[j]
            if draws.size == 0:
                continue

            c = colors[j % len(colors)]
            ls = linestyles[j % len(linestyles)]
            kde, y_vals = row_kdes[j]
            if y_vals is None:
                y_vals = np.zeros_like(x_vals)

            # plot KDE
            ax.plot(x_vals, y_vals, color=c, linewidth=1.6, linestyle=ls)
            ax.fill_between(x_vals, 0, y_vals, alpha=color_alpha, color=c)

            # stats
            med = float(np.median(draws))
            low, high = np.percentile(draws, ci)

            # vertical markers
            ax.axvline(low, color=c, linestyle=":", linewidth=1)
            ax.axvline(high, color=c, linestyle=":", linewidth=1)
            ax.axvline(med, color=c, linestyle="--", linewidth=1)

            # horizontal CI line
            peak = row_peaks[j] if row_peaks[j] > 0 else np.max(y_vals)
            y_horiz = peak * horiz_peak_frac
            ax.hlines(y_horiz, low, high, color=c, linewidth=1.5)
            ax.plot(med, y_horiz, "o", markeredgecolor=c, markerfacecolor="white", markersize=median_marker_size)

            # shifts
            width_ci = float(high - low)
            label_shift = 0.0
            if width_ci < ci_shift_factor:
                label_shift = (width_ci * ci_shift_num) if width_ci > 0 else ( (p_hi - p_lo) * 0.005 )

            # label vertical position
            ylim = ax.get_ylim()
            vpad = label_vpad_frac * (ylim[1] - ylim[0])
            y_label_base = y_horiz + vpad
            y_label_med = y_label_base + (median_vshift_frac * (ylim[1] - ylim[0]))

            # formatting
            if abs(med) > format_currency_threshold:
                prefix, fmt = "£", "{:,.0f}"
            else:
                prefix, fmt = "", "{:,.2f}"
            fmt_val = lambda x: f"{prefix}{fmt.format(float(x))}"

            # draw labels
            ax.text(low - label_shift, y_label_base, fmt_val(low),
                    ha="center", va="bottom", fontsize=ci_font_size, color=ci_label_color)
            ax.text(high + label_shift, y_label_base, fmt_val(high),
                    ha="center", va="bottom", fontsize=ci_font_size, color=ci_label_color)
            ax.text(med, y_label_med, fmt_val(med),
                    ha="center", va="bottom", fontsize=ci_font_size,
                    color=median_label_color, weight="bold" if bold_median else "normal")

        # axes style
        ax.set_ylabel("Density", fontsize=font_size)
        ax.tick_params(axis="both", labelsize=max(8, font_size - 1))
        ax.grid(axis="x", linestyle=":", alpha=0.5)  
        
        ax.grid(axis="y", linestyle="--", alpha=0.35)  # add horizontal dashed gridlines

        ax.set_xlim(p_lo, p_hi)
        if yticks is not None:
            ax.set_yticks(yticks)

    ax.grid(axis="y", linestyle="--", alpha=0.35)  # add horizontal dashed gridlines
    ax.set_axisbelow(True)   

    if xticks is not None:
        axes[-1].set_xticks(xticks)
    axes[-1].set_xlabel(", ".join(var_list), fontsize=font_size)
    fig.tight_layout()
    return fig, axes




