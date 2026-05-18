"""
Visualise the Shenzhen vs Geneva variable comparison table.

Input  : the cross-cohort alignment .xlsx (sheet `with_overlap` by default),
         produced upstream and currently kept at
         /home/klug/temp/shenzen_gva_comp/data/gva_shenzhen_comparison.xlsx
Output : a multi-row figure (PNG + PDF) — one row per variable.

Layout
  - Continuous variables -> one subplot spanning both columns, with
    Shenzhen and Geneva boxes drawn side-by-side (boxes from median + IQR,
    no whiskers because the source has only summary stats).
  - Binary / Multi-category / Ordinal -> two subplots (Shenzhen left,
    Geneva right) with horizontal bar plots of category percentages.
  - Date and Free Text rows are dropped.

Colors
  - Shenzhen = blue; Geneva = orange. Multi-bar cells use evenly spaced
  shades of the cohort colormap.

Usage
    python visualize_gva_shenzhen_comparison.py
    python visualize_gva_shenzhen_comparison.py \
        --input /path/to/gva_shenzhen_comparison.xlsx \
        --sheet with_overlap \
        --output out/gva_shenzhen_comparison.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps, gridspec
from matplotlib.patches import Patch

from mappings import UNIT_CONVERSIONS


DEFAULT_INPUT = Path(
    "/home/klug/temp/shenzen_gva_comp/data/gva_shenzhen_comparison.xlsx"
)
DEFAULT_SHEET = "with_overlap"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "out" / "gva_shenzhen_comparison.png"

SKIP_TYPES: set[str] = {"Date", "Free Text"}
CONTINUOUS_TYPES: set[str] = {"Continuous"}
CATEGORICAL_TYPES: set[str] = {"Binary", "Multi-category", "Ordinal"}

# Per-Shenzhen-variable label rewrites (keys = shenzhen_variable_name).
# Use when the raw No/yes from the summary string is ambiguous.
SHENZHEN_LABEL_OVERRIDES: dict[str, dict[str, str]] = {
    "Wake-up Stroke": {"No": "Not wake-up", "yes": "Wake-up"},
}

SHENZHEN_BASE = "tab:blue"
GENEVA_BASE = "tab:orange"
SHENZHEN_CMAP = colormaps["Blues"]
GENEVA_CMAP = colormaps["Oranges"]

# Accept hyphen, en-dash, em-dash, and minus sign as range separators.
_DASH = r"[-‐‑‒–—−]"

_RE_CONTINUOUS = re.compile(
    rf"^\s*(-?\d+(?:\.\d+)?)\s*\(\s*(-?\d+(?:\.\d+)?)\s*{_DASH}\s*(-?\d+(?:\.\d+)?)\s*\)\s*$"
)
_RE_CAT_COLON = re.compile(r"^\s*(.+?):\s*([\d,]+)\s*\(\s*([\d.]+)\s*%?\s*\)\s*$")
_RE_CAT_SPACE = re.compile(r"^\s*(.+?)\s+([\d,]+)\s*\(\s*([\d.]+)\s*%?\s*\)\s*$")
_RE_CAT_BARE = re.compile(r"^\s*([\d,]+)\s*\(\s*([\d.]+)\s*%?\s*\)\s*$")


def parse_continuous(s: object) -> tuple[float, float, float] | None:
    """Parse `median (q1-q3)` (any dash). Return None on miss."""
    if not isinstance(s, str):
        return None
    m = _RE_CONTINUOUS.match(s)
    if not m:
        return None
    median, q1, q3 = (float(g) for g in m.groups())
    if q1 > q3:
        q1, q3 = q3, q1
    return median, q1, q3


def parse_categorical(s: object) -> list[tuple[str, float]]:
    """Parse `Cat: N (P%); Cat: N (P%); ...`.

    Returns list of (label, percentage). Tolerates Shenzhen binary shorthand
    `Label N (P%)` (no colon) and bare `N (P%)` items. Drops unparseable
    chunks (handles meta prefixes like `"6 categories; top:"`).
    """
    if not isinstance(s, str):
        return []
    out: list[tuple[str, float]] = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = _RE_CAT_COLON.match(chunk)
        if m:
            label, _count, pct = m.groups()
            out.append((label.strip(), float(pct)))
            continue
        m = _RE_CAT_SPACE.match(chunk)
        if m:
            label, _count, pct = m.groups()
            out.append((label.strip(), float(pct)))
            continue
        m = _RE_CAT_BARE.match(chunk)
        if m:
            _count, pct = m.groups()
            out.append(("", float(pct)))
            continue
    return out


def maybe_complement(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """For binary rows that ship a single category, add the complement bar."""
    if len(items) != 1:
        return items
    label, pct = items[0]
    if pct >= 100.0 or pct <= 0.0:
        return items
    other = "yes" if label.strip().lower() == "no" else (
        "no" if label.strip().lower() == "yes" else "other"
    )
    return [items[0], (other, round(100.0 - pct, 1))]


def maybe_expand_bare_binary(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """If parsed result is a single bare-label item (`N (P%)` with no label),
    treat it as a binary positive count and add the complementary `no` bar.

    Applies independently to each cohort cell, so a Multi-category row whose
    Geneva side ships only the positive count (e.g. `MedHist Smoking`: 473 (13.8%))
    still renders as yes/no on Geneva while Shenzhen keeps its multi-cat bars.
    """
    if len(items) != 1:
        return items
    label, pct = items[0]
    if label != "" or pct <= 0.0 or pct >= 100.0:
        return items
    return [("yes", pct), ("no", round(100.0 - pct, 1))]


def canonical_yesno_sort(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Stable sort that pins 'yes' first and 'no' second when present, so
    both cohorts share y-tick order on binaries/yes-no multi-categories."""
    def key(indexed):
        idx, (label, _) = indexed
        norm = label.strip().lower()
        if norm == "yes":
            return (0, idx)
        if norm == "no":
            return (1, idx)
        return (2, idx)
    return [item for _, item in sorted(enumerate(items), key=key)]


def maybe_sort_ordinal(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """If all labels are numeric (e.g. mRS 0..6), sort by numeric value."""
    if not items:
        return items
    try:
        keyed = sorted(items, key=lambda x: float(x[0]))
    except ValueError:
        return items
    return keyed


def strip_spines(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def shades(cmap, n: int) -> list:
    """n evenly spaced shades from a sequential colormap, avoiding extremes."""
    if n <= 0:
        return []
    if n == 1:
        return [cmap(0.7)]
    return [cmap(v) for v in np.linspace(0.45, 0.85, n)]


def render_boxes(ax, shen: tuple[float, float, float] | None,
                 gva: tuple[float, float, float] | None,
                 var_name: str) -> None:
    """Draw Shenzhen and Geneva box-from-summary on the same axes."""
    stats = []
    labels = []
    colors = []
    if shen is not None:
        med, q1, q3 = shen
        stats.append({"med": med, "q1": q1, "q3": q3,
                      "whislo": q1, "whishi": q3, "fliers": []})
        labels.append("Shenzhen")
        colors.append(SHENZHEN_BASE)
    if gva is not None:
        med, q1, q3 = gva
        stats.append({"med": med, "q1": q1, "q3": q3,
                      "whislo": q1, "whishi": q3, "fliers": []})
        labels.append("Geneva")
        colors.append(GENEVA_BASE)

    if not stats:
        ax.text(0.5, 0.5, "no parseable summary", ha="center", va="center",
                color="grey", style="italic", fontsize=8, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(var_name, loc="left", fontsize=9, fontweight="bold")
        strip_spines(ax)
        return

    bplot = ax.bxp(stats, positions=list(range(1, len(stats) + 1)),
                   widths=0.55, showfliers=False, patch_artist=True,
                   medianprops={"color": "black", "linewidth": 1.5})
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("black")
    for whisker in bplot["whiskers"]:
        whisker.set_visible(False)
    for cap in bplot["caps"]:
        cap.set_visible(False)
    ax.set_xticks(list(range(1, len(stats) + 1)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_title(var_name, loc="left", fontsize=9, fontweight="bold")
    strip_spines(ax)


def render_bars(ax, items: list[tuple[str, float]], cmap,
                cohort_label: str, var_name: str | None = None) -> None:
    """Horizontal bar plot of (label, pct) with shaded colors."""
    if not items:
        ax.text(0.5, 0.5, "no parseable summary", ha="center", va="center",
                color="grey", style="italic", fontsize=8, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        if var_name is not None:
            ax.set_title(f"{cohort_label} — {var_name}", loc="left",
                         fontsize=9, fontweight="bold")
        else:
            ax.set_title(cohort_label, loc="left", fontsize=8)
        strip_spines(ax)
        return

    labels = [lbl if lbl else "(unlabelled)" for lbl, _ in items]
    pcts = [p for _, p in items]
    colors = shades(cmap, len(items))
    y_pos = np.arange(len(items))[::-1]  # first item on top
    bars = ax.barh(y_pos, pcts, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0, max(100.0, max(pcts) * 1.15))
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    xmax = ax.get_xlim()[1]
    for rect, pct in zip(bars, pcts):
        ax.text(rect.get_width() + xmax * 0.01,
                rect.get_y() + rect.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=6.5, color="black")
    if var_name is not None:
        ax.set_title(f"{cohort_label} — {var_name}", loc="left",
                     fontsize=9, fontweight="bold")
    else:
        ax.set_title(cohort_label, loc="left", fontsize=8)
    strip_spines(ax)


def load_rows(input_path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(input_path, sheet_name=sheet)
    # Normalize Variable Type case so "continuous" matches "Continuous", etc.
    # Maps casefolded value -> canonical spelling used by CONTINUOUS_TYPES /
    # CATEGORICAL_TYPES / SKIP_TYPES; unknown values pass through unchanged.
    canon = {t.casefold(): t for t in SKIP_TYPES | CONTINUOUS_TYPES | CATEGORICAL_TYPES}
    df["Variable Type"] = df["Variable Type"].astype("string").str.strip().map(
        lambda v: canon.get(v.casefold(), v) if isinstance(v, str) else v
    )
    df = df[~df["Variable Type"].isin(SKIP_TYPES)].reset_index(drop=True)
    return df


CONTINUOUS_ROW_INCHES = 1.6
BAR_ROW_BASE_INCHES = 1.1
BAR_ROW_PER_BAR_INCHES = 0.35
ROW_GAP_INCHES = 0.55
FIG_WIDTH_INCHES = 14.0


def _row_label(row: pd.Series) -> str:
    shen_name = row.get("shenzhen_variable_name")
    gva_name = row.get("geneva_variable_name")
    if isinstance(shen_name, str) and isinstance(gva_name, str) \
            and shen_name.strip().lower() != gva_name.strip().lower():
        return f"{shen_name}  /  {gva_name}"
    return shen_name if isinstance(shen_name, str) else str(gva_name)


def _prepare_rows(df: pd.DataFrame) -> list[dict]:
    """Parse each row once and attach a target row-height in inches."""
    prepared: list[dict] = []
    for _, row in df.iterrows():
        vtype = row["Variable Type"]
        entry: dict = {"vtype": vtype, "var_name": _row_label(row)}

        if vtype in CONTINUOUS_TYPES:
            entry["shen"] = parse_continuous(row.get("shenzhen_summary_statistic"))
            entry["gva"] = parse_continuous(row.get("geneva_summary_statistic"))
            # Convert Geneva values into Shenzhen units when a conversion is
            # defined for this Geneva variable, so both boxes share a scale.
            gva_var = row.get("geneva_variable_name")
            if entry["gva"] is not None and isinstance(gva_var, str):
                conv = UNIT_CONVERSIONS.get(gva_var)
                if conv:
                    f = conv["factor"]
                    entry["gva"] = tuple(v * f for v in entry["gva"])
            entry["height_in"] = CONTINUOUS_ROW_INCHES
        elif vtype in CATEGORICAL_TYPES:
            shen_items = parse_categorical(row.get("shenzhen_summary_statistic"))
            gva_items = parse_categorical(row.get("geneva_summary_statistic"))
            shen_items = maybe_expand_bare_binary(shen_items)
            gva_items = maybe_expand_bare_binary(gva_items)
            if vtype == "Binary":
                # Bare "N (P%)" in a binary row means the positive count.
                shen_items = [(lbl or "yes", pct) for lbl, pct in shen_items]
                gva_items = [(lbl or "yes", pct) for lbl, pct in gva_items]
                shen_items = maybe_complement(shen_items)
                gva_items = maybe_complement(gva_items)
            if vtype in ("Binary", "Multi-category"):
                # Pin yes/no order so both cohorts share y-tick order.
                shen_items = canonical_yesno_sort(shen_items)
                gva_items = canonical_yesno_sort(gva_items)
            if vtype == "Binary":
                shen_map = SHENZHEN_LABEL_OVERRIDES.get(
                    row.get("shenzhen_variable_name")
                )
                if shen_map:
                    shen_items = [(shen_map.get(lbl, lbl), pct)
                                  for lbl, pct in shen_items]
            if vtype == "Ordinal":
                shen_items = maybe_sort_ordinal(shen_items)
                gva_items = maybe_sort_ordinal(gva_items)
            entry["shen_items"] = shen_items
            entry["gva_items"] = gva_items
            max_bars = max(len(shen_items), len(gva_items), 1)
            entry["height_in"] = max(
                CONTINUOUS_ROW_INCHES,
                BAR_ROW_BASE_INCHES + BAR_ROW_PER_BAR_INCHES * max_bars,
            )
        else:
            entry["height_in"] = CONTINUOUS_ROW_INCHES

        prepared.append(entry)
    return prepared


def build_figure(df: pd.DataFrame) -> plt.Figure:
    rows = _prepare_rows(df)
    n = len(rows)
    row_heights = [r["height_in"] for r in rows]

    total_axes_in = sum(row_heights)
    fig_h = total_axes_in + ROW_GAP_INCHES * max(n - 1, 0) + 0.8
    fig = plt.figure(figsize=(FIG_WIDTH_INCHES, fig_h))

    avg_row_in = total_axes_in / max(n, 1)
    hspace_rel = ROW_GAP_INCHES / max(avg_row_in, 0.1)
    top_margin = 1.0 - 0.4 / fig_h
    bottom_margin = 0.4 / fig_h

    gs = gridspec.GridSpec(
        n, 2, figure=fig,
        height_ratios=row_heights,
        hspace=hspace_rel, wspace=0.35,
        left=0.10, right=0.97, top=top_margin, bottom=bottom_margin,
    )

    for i, entry in enumerate(rows):
        vtype = entry["vtype"]
        var_name = entry["var_name"]
        if vtype in CONTINUOUS_TYPES:
            ax = fig.add_subplot(gs[i, :])
            render_boxes(ax, entry["shen"], entry["gva"], var_name)
        elif vtype in CATEGORICAL_TYPES:
            ax_s = fig.add_subplot(gs[i, 0])
            ax_g = fig.add_subplot(gs[i, 1])
            render_bars(ax_s, entry["shen_items"], SHENZHEN_CMAP, "Shenzhen", var_name)
            render_bars(ax_g, entry["gva_items"], GENEVA_CMAP, "Geneva")
        else:
            ax = fig.add_subplot(gs[i, :])
            ax.text(0.5, 0.5, f"unsupported type: {vtype}", ha="center",
                    va="center", color="grey", style="italic",
                    transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(var_name, loc="left", fontsize=9, fontweight="bold")
            strip_spines(ax)

    legend = [
        Patch(facecolor=SHENZHEN_BASE, alpha=0.7, edgecolor="black",
              label="Shenzhen"),
        Patch(facecolor=GENEVA_BASE, alpha=0.7, edgecolor="black",
              label="Geneva"),
    ]
    fig.legend(handles=legend, loc="upper right",
               bbox_to_anchor=(0.99, 0.999), ncol=2, fontsize=9,
               frameon=False)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise Shenzhen vs Geneva variable comparison"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"Path to comparison .xlsx (default: {DEFAULT_INPUT})")
    parser.add_argument("--sheet", default=DEFAULT_SHEET,
                        help=f"Sheet name (default: {DEFAULT_SHEET})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output PNG path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load]  {args.input}  sheet={args.sheet!r}")
    df = load_rows(args.input, args.sheet)
    print(f"[load]  {len(df)} rows after dropping {sorted(SKIP_TYPES)}")

    fig = build_figure(df)

    png_path = args.output.with_suffix(".png")
    pdf_path = args.output.with_suffix(".pdf")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {png_path}")
    print(f"[write] {pdf_path}")


if __name__ == "__main__":
    main()
