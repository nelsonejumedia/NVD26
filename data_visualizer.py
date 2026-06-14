"""
Morphometric Data Visualization & Descriptive Statistics Tool
================================================================
For research data such as brainstem / cerebellar vermis parameters
(PVD, MVD, MMAPD, MAPD, TVD, MAD, BPAPD, PMAPD, MOL, CVH, CVAPD, etc.)

Features:
- Manual data entry (add values for any number of parameters)
- CSV import (one column per parameter, header = parameter name)
- Descriptive statistics (n, mean, SD, median, min, max, range, SEM, 95% CI)
- Visualizations: bar chart, pie chart, histogram, scatter plot, line plot
- Export stats summary to CSV
- Export charts as PNG

Run:
    python morphometry_viz.py
"""

import os
import sys
import csv
import statistics as stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False


# --------------------------------------------------------------------------- #
#  DATA STORAGE
# --------------------------------------------------------------------------- #
class DataStore:
    """Holds parameter -> list of measured values."""

    def __init__(self):
        self.data = {}  # {"PVD": [12.3, 11.8, ...], "MVD": [...], ...}

    # ---------- manual entry ---------- #
    def add_value(self, parameter, value):
        parameter = parameter.strip().upper()
        self.data.setdefault(parameter, []).append(float(value))

    def add_values(self, parameter, values):
        parameter = parameter.strip().upper()
        self.data.setdefault(parameter, []).extend([float(v) for v in values])

    # ---------- CSV import ---------- #
    def load_csv(self, filepath):
        """
        Expects a CSV where each column header is a parameter name
        and each row is one subject's measurement for that parameter.
        Missing values (blank cells) are skipped.
        """
        df = pd.read_csv(filepath, dtype=object)
        skipped = []

        for col in df.columns:
            series = df[col].dropna()

            # Try robust numeric conversion (handles strings with spaces)
            numeric = pd.to_numeric(series.astype(str).str.strip(), errors="coerce")

            if numeric.notna().all():
                values = numeric.astype(float).tolist()
                if values:
                    self.data.setdefault(col.strip().upper(), []).extend(values)
                else:
                    skipped.append(col.strip().upper())
                continue

            # Column contains non-numeric values. Show unique values and
            # ask user to provide a mapping, or auto-assign integers.
            uniques = sorted({str(x).strip() for x in series.unique() if pd.notna(x)})
            print(f"\nColumn '{col}' contains non-numeric values: {uniques}")
            print("Provide mappings like M=1,F=2 (press Enter to auto-assign 1..N).")
            print("Type 'skip' to ignore this column.")

            mapping = None
            while mapping is None:
                mapping_input = input("Mapping: ").strip()

                if not mapping_input:
                    # Auto assign integers starting from 1
                    mapping = {val: float(i + 1) for i, val in enumerate(uniques)}
                    print(f"Auto-mapping applied: {mapping}")
                    break

                if mapping_input.lower() in ("skip", "s"):
                    print(f"Skipping column '{col}'.")
                    mapping = {}
                    skipped.append(col.strip().upper())
                    break

                # Try to parse user-provided mapping safely
                try:
                    tmp = {}
                    parts = [p.strip() for p in mapping_input.split(",") if p.strip()]
                    for part in parts:
                        if "=" in part:
                            k, v = part.split("=", 1)
                        elif ":" in part:
                            k, v = part.split(":", 1)
                        else:
                            toks = part.split()
                            if len(toks) >= 2:
                                k, v = toks[0], toks[1]
                            else:
                                raise ValueError(f"Invalid mapping part: '{part}'")
                        k = str(k).strip()
                        v = float(str(v).strip())
                        tmp[k] = v
                    # If parsing succeeded, accept mapping
                    mapping = tmp
                except Exception as e:
                    print(f"Invalid mapping: {e}. Try again or press Enter to auto-assign.")

            if not mapping:
                # If mapping was set to empty dict (skip), continue to next column
                continue

            # Apply mapping to the series (strip strings before mapping)
            mapped = series.astype(str).str.strip().map(lambda x: mapping.get(x, np.nan)).astype(float)
            values = mapped.dropna().tolist()
            if values:
                self.data.setdefault(col.strip().upper(), []).extend(values)
            else:
                skipped.append(col.strip().upper())
        return df, skipped

    # ---------- utilities ---------- #
    def parameters(self):
        return list(self.data.keys())

    def get(self, parameter):
        parameter = parameter.strip().upper()
        return self.data.get(parameter, [])

    def remove_parameter(self, parameter):
        parameter = parameter.strip().upper()
        self.data.pop(parameter, None)

    def is_empty(self):
        return len(self.data) == 0

    def to_dataframe(self):
        """Convert to a DataFrame (columns padded with NaN to equal length)."""
        max_len = max((len(v) for v in self.data.values()), default=0)
        padded = {
            k: v + [np.nan] * (max_len - len(v)) for k, v in self.data.items()
        }
        return pd.DataFrame(padded)


# --------------------------------------------------------------------------- #
#  DESCRIPTIVE STATISTICS
# --------------------------------------------------------------------------- #
def descriptive_stats(values):
    """Return a dict of descriptive statistics for a list of numeric values."""
    n = len(values)
    if n == 0:
        return None

    mean = stats.mean(values)
    sd = stats.stdev(values) if n > 1 else 0.0
    sem = sd / (n ** 0.5) if n > 1 else 0.0
    median = stats.median(values)
    minimum = min(values)
    maximum = max(values)
    rng = maximum - minimum

    # 95% CI using normal approximation (t recommended for small n, but
    # this gives a quick estimate; for n < 30 consider using a t-table)
    ci95 = 1.96 * sem

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "median": median,
        "min": minimum,
        "max": maximum,
        "range": rng,
        "ci95_lower": mean - ci95,
        "ci95_upper": mean + ci95,
    }


def print_stats_table(data_store):
    if data_store.is_empty():
        print("\nNo data available. Add data first.\n")
        return

    rows = []
    for param in data_store.parameters():
        values = data_store.get(param)
        s = descriptive_stats(values)
        if s:
            rows.append({
                "Parameter": param,
                "n": s["n"],
                "Mean": round(s["mean"], 3),
                "SD": round(s["sd"], 3),
                "SEM": round(s["sem"], 3),
                "Median": round(s["median"], 3),
                "Min": round(s["min"], 3),
                "Max": round(s["max"], 3),
                "Range": round(s["range"], 3),
                "95% CI": f"[{s['ci95_lower']:.3f}, {s['ci95_upper']:.3f}]",
            })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("DESCRIPTIVE STATISTICS SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")
    return df


def export_stats_csv(data_store, filepath="descriptive_stats.csv"):
    df = print_stats_table(data_store)
    if df is not None:
        df.to_csv(filepath, index=False)
        print(f"Stats exported to: {filepath}")


# --------------------------------------------------------------------------- #
#  VISUALIZATIONS
# --------------------------------------------------------------------------- #
OUTPUT_DIR = "charts"


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save_and_show(fig, filename):
    _ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()
    plt.close(fig)


def bar_chart(data_store, parameters=None, use_mean=True):
    """Bar chart comparing mean (or sum) values across selected parameters."""
    parameters = parameters or data_store.parameters()
    labels, values, errors = [], [], []

    for p in parameters:
        vals = data_store.get(p)
        if not vals:
            continue
        s = descriptive_stats(vals)
        labels.append(p)
        values.append(s["mean"] if use_mean else sum(vals))
        errors.append(s["sd"])

    if not labels:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, yerr=errors if use_mean else None,
                   capsize=5, color="#4C72B0", edgecolor="black")
    ax.set_ylabel("Mean value (± SD)" if use_mean else "Sum")
    ax.set_title("Bar Chart of Measured Parameters")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _save_and_show(fig, "bar_chart.png")


def pie_chart(data_store, parameters=None, use="mean"):
    """Pie chart showing relative proportions of parameter means/sums."""
    parameters = parameters or data_store.parameters()
    labels, values = [], []

    for p in parameters:
        vals = data_store.get(p)
        if not vals:
            continue
        s = descriptive_stats(vals)
        labels.append(p)
        values.append(s["mean"] if use == "mean" else sum(vals))

    if not labels:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90,
           wedgeprops={"edgecolor": "white"})
    ax.set_title(f"Pie Chart of Parameter {'Means' if use == 'mean' else 'Sums'}")
    plt.tight_layout()
    _save_and_show(fig, "pie_chart.png")


def histogram(data_store, parameter, bins=10):
    """Histogram of the distribution of a single parameter."""
    values = data_store.get(parameter)
    if not values:
        print(f"No data found for parameter '{parameter}'.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, color="#55A868", edgecolor="black", alpha=0.85)
    ax.set_xlabel(parameter)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram of {parameter} (n={len(values)})")

    s = descriptive_stats(values)
    ax.axvline(s["mean"], color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {s['mean']:.2f}")
    ax.axvline(s["median"], color="blue", linestyle=":", linewidth=1.5,
               label=f"Median = {s['median']:.2f}")
    ax.legend()
    plt.tight_layout()
    _save_and_show(fig, f"histogram_{parameter}.png")


def violin_plot(data_store, parameter):
    """Violin plot showing distribution of a single parameter."""
    values = data_store.get(parameter)
    if not values:
        print(f"No data found for parameter '{parameter}'.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    # matplotlib.violinplot expects a sequence of datasets
    parts = ax.violinplot([values], showmeans=False, showmedians=True)

    for pc in parts.get('bodies', []):
        pc.set_facecolor('#88CCEE')
        pc.set_edgecolor('black')
        pc.set_alpha(0.9)

    if 'cmedians' in parts:
        parts['cmedians'].set_color('black')

    ax.set_xticks([1])
    ax.set_xticklabels([parameter])
    ax.set_ylabel('Value')
    ax.set_title(f'Violin Plot of {parameter} (n={len(values)})')
    plt.tight_layout()
    _save_and_show(fig, f"violin_{parameter}.png")


def correlation_heatmap(data_store, params=None, method="pearson"):
    """Plot a correlation heatmap for selected numeric parameters.

    `params` is a list of parameter names (case-insensitive, matched to data_store keys).
    `method` can be 'pearson', 'spearman', or 'kendall'.
    """
    df = data_store.to_dataframe()

    # If params not provided, use all parameters
    if not params:
        params = list(df.columns)
    else:
        # Normalize and filter to existing columns
        params = [p.strip().upper() for p in params if p and p.strip().upper() in df.columns]

    if not params:
        print("No valid parameters selected for correlation.")
        return

    # Select and coerce to numeric, dropping columns that are all NaN
    numeric_df = df[params].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if numeric_df.shape[1] < 2:
        print("Not enough numeric parameters to compute correlations (need at least 2).")
        return

    corr = numeric_df.corr(method=method)

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title(f"Correlation Heatmap ({method})")

    # Annotate values
    for (i, j), val in np.ndenumerate(corr.values):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                color="black" if abs(val) < 0.6 else "white")

    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    plt.tight_layout()
    _save_and_show(fig, f"correlation_heatmap_{method}.png")


def pair_plot(data_store, params=None, hue=None, diag_kind="kde"):
    """Create a scatter-matrix (pair plot) for selected numeric parameters.

    If `hue` is provided (a categorical parameter name), points will be colored
    by that category. Uses seaborn's `pairplot` (with diagonal KDE) when available.
    """
    df = data_store.to_dataframe()

    if not params:
        params = list(df.columns)
    else:
        params = [p.strip().upper() for p in params if p and p.strip().upper() in df.columns]

    if not params:
        print("No valid parameters selected for pair plot.")
        return

    # Prepare numeric dataframe
    numeric_df = df[params].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if numeric_df.shape[1] < 2:
        print("Need at least two numeric parameters for a pair plot.")
        return

    if hue:
        hue_col = hue.strip().upper()
        if hue_col not in df.columns:
            print(f"Hue column '{hue}' not found; proceeding without hue.")
            hue = None
        else:
            # include hue from original df (not coerced to numeric)
            hue_series = df[hue_col].astype(str).str.strip()
            # join for seaborn
            plot_df = numeric_df.join(hue_series.rename(hue_col))
    else:
        plot_df = numeric_df

    if _HAS_SEABORN:
        try:
            g = sns.pairplot(plot_df, diag_kind=diag_kind, hue=hue if hue else None, plot_kws={"alpha": 0.6})
            g.fig.suptitle("Pair Plot (Scatter Matrix)", y=1.02)
            # Save the figure
            outname = f"pairplot_{'_'.join(numeric_df.columns)}.png"
            g.fig.tight_layout()
            g.fig.savefig(os.path.join(OUTPUT_DIR, outname), dpi=150, bbox_inches="tight")
            print(f"Saved: {os.path.join(OUTPUT_DIR, outname)}")
            plt.show()
            plt.close(g.fig)
            return
        except Exception as e:
            print(f"Seaborn pairplot failed: {e}. Falling back to basic scatter matrix.")

    # Fallback: pandas scatter_matrix (no hue, diagonal histogram)
    n = numeric_df.shape[1]
    figsize = (max(6, n * 2), max(6, n * 2))
    axes = pd.plotting.scatter_matrix(numeric_df, figsize=figsize, diagonal="hist", alpha=0.6)

    fig = plt.gcf()
    plt.suptitle("Pair Plot (Scatter Matrix)")
    plt.tight_layout()
    _save_and_show(fig, f"pairplot_{'_'.join(numeric_df.columns)}.png")


def box_plot(data_store, params=None, by=None):
    """Create box plots for selected parameters. If `by` is provided (categorical column),
    boxes will be split by category (requires seaborn for grouped boxes).
    """
    df = data_store.to_dataframe()

    if not params:
        params = list(df.columns)
    else:
        params = [p.strip().upper() for p in params if p and p.strip().upper() in df.columns]

    if not params:
        print("No valid parameters selected for box plot.")
        return

    numeric_df = df[params].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if numeric_df.shape[1] == 0:
        print("No numeric parameters available for box plot.")
        return

    # If grouping requested and seaborn available, use seaborn boxplot with hue
    if by and _HAS_SEABORN and by in df.columns:
        group_series = df[by].astype(str).str.strip()
        # Melt numeric_df to long form
        melted = numeric_df.melt(var_name="Parameter", value_name="Value")
        # Repeat group_series for each parameter (length of numeric_df.columns)
        reps = pd.concat([group_series.reset_index(drop=True)] * len(numeric_df.columns), ignore_index=True)
        melted[by] = reps

        plt.figure(figsize=(max(8, len(numeric_df.columns) * 1.2), 6))
        sns.boxplot(x="Parameter", y="Value", hue=by, data=melted)
        plt.title(f"Box Plot grouped by {by}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        outname = f"boxplot_{'_'.join(numeric_df.columns)}_by_{by}.png"
        _ensure_output_dir()
        path = os.path.join(OUTPUT_DIR, outname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
        plt.show()
        plt.close()
        return

    # Fallback: simple matplotlib boxplot for each parameter (no grouping)
    data_to_plot = [numeric_df[col].dropna().values for col in numeric_df.columns]
    if not any(len(d) for d in data_to_plot):
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(data_to_plot) * 1.2), 6))
    bp = ax.boxplot(data_to_plot, patch_artist=True)

    # Set x tick labels (boxplot positions are 1..N)
    ax.set_xticks(range(1, len(numeric_df.columns) + 1))
    ax.set_xticklabels(numeric_df.columns)

    # Style boxes
    colors = ["#4C72B0"] * len(data_to_plot)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_title("Box Plot")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _save_and_show(fig, f"boxplot_{'_'.join(numeric_df.columns)}.png")





def scatter_plot(data_store, param_x, param_y):
    """Scatter plot of two parameters against each other."""
    x = data_store.get(param_x)
    y = data_store.get(param_y)

    if not x or not y:
        print("One or both parameters have no data.")
        return

    n = min(len(x), len(y))
    if len(x) != len(y):
        print(f"Note: '{param_x}' has {len(x)} values and '{param_y}' has "
              f"{len(y)} values. Using the first {n} paired values.")
    x, y = x[:n], y[:n]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, color="#C44E52", edgecolor="black", alpha=0.8)

    # Add simple linear trend line
    if n > 1:
        coeffs = np.polyfit(x, y, 1)
        trend = np.poly1d(coeffs)
        x_sorted = sorted(x)
        ax.plot(x_sorted, trend(x_sorted), color="black", linestyle="--",
                label=f"Trend: y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}")
        # Pearson correlation
        r = np.corrcoef(x, y)[0, 1]
        ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.legend()

    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title(f"Scatter Plot: {param_x} vs {param_y}")
    plt.tight_layout()
    _save_and_show(fig, f"scatter_{param_x}_vs_{param_y}.png")


def line_plot(data_store, parameters=None):
    """Line plot showing values of one or more parameters across the sample index."""
    parameters = parameters or data_store.parameters()
    valid = [p for p in parameters if data_store.get(p)]

    if not valid:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for p in valid:
        values = data_store.get(p)
        ax.plot(range(1, len(values) + 1), values, marker="o", label=p)

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Measured value")
    ax.set_title("Line Plot of Measured Parameters")
    ax.legend()
    plt.tight_layout()
    _save_and_show(fig, "line_plot.png")


# --------------------------------------------------------------------------- #
#  INTERACTIVE MENU
# --------------------------------------------------------------------------- #
def prompt_select_parameters(data_store, allow_empty=True, prompt_text=None):
    params = data_store.parameters()
    if not params:
        print("No parameters available yet.")
        return []

    prompt_text = prompt_text or (
        "Enter parameter names separated by commas (or press Enter for ALL): "
    )
    print("Available parameters:", ", ".join(params))
    raw = input(prompt_text).strip()
    if not raw:
        return params if allow_empty else []
    chosen = [p.strip().upper() for p in raw.split(",")]
    return [p for p in chosen if p in params]


def menu_add_manual_data(ds):
    print("\n--- Manual Data Entry ---")
    print("Type a parameter name (e.g. PVD, MVD, MMAPD).")
    param = input("Parameter name: ").strip().upper()
    if not param:
        print("No parameter entered.")
        return

    print(f"Now enter values for {param}, one at a time.")
    print("Type 'done' when finished.")
    count = 0
    while True:
        raw = input(f"  {param} value #{count + 1} (or 'done'): ").strip()
        if raw.lower() == "done":
            break
        try:
            ds.add_value(param, raw)
            count += 1
        except ValueError:
            print("  Invalid number, try again.")

    print(f"Added {count} value(s) to {param}. "
          f"Total values for {param}: {len(ds.get(param))}\n")


def menu_load_csv(ds):
    print("\n--- Load CSV File ---")
    filepath = input("Enter path to CSV file (in current folder, e.g. data.csv): ").strip()
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    try:
        df, skipped = ds.load_csv(filepath)
        print(f"Loaded columns: {list(df.columns)}")
        print(f"Parameters now available: {ds.parameters()}\n")

        if not skipped:
            print("All non-numeric columns were mapped. Proceeding to visualization...")
            menu_visualize(ds)
        else:
            print(f"Columns skipped or with no values: {skipped}\n")
    except Exception as e:
        print(f"Error reading CSV: {e}")


def menu_view_data(ds):
    print("\n--- Current Data ---")
    if ds.is_empty():
        print("No data yet.\n")
        return
    for p in ds.parameters():
        vals = ds.get(p)
        print(f"{p} (n={len(vals)}): {vals}")
    print()


def menu_remove_parameter(ds):
    print("\n--- Remove Parameter ---")
    if ds.is_empty():
        print("No data yet.\n")
        return
    print("Available:", ", ".join(ds.parameters()))
    p = input("Parameter to remove: ").strip().upper()
    if p in ds.parameters():
        ds.remove_parameter(p)
        print(f"Removed {p}.\n")
    else:
        print("Parameter not found.\n")


def menu_visualize(ds):
    if ds.is_empty():
        print("\nNo data available. Add data first.\n")
        return

    print("\n--- Visualization Menu ---")
    print("1. Bar chart (mean ± SD per parameter)")
    print("2. Pie chart (relative proportions)")
    print("3. Histogram (single parameter distribution)")
    print("4. Scatter plot (two parameters)")
    print("5. Line plot (one or more parameters)")
    print("6. Violin plot (single parameter)")
    print("7. Correlation between variables")
    print("8. Pair plot (scatter matrix)")
    print("9. Box plot (one or more parameters)")
    print("0. Back")
    choice = input("Select option: ").strip()

    if choice == "1":
        chosen = prompt_select_parameters(ds)
        bar_chart(ds, chosen)
    elif choice == "2":
        chosen = prompt_select_parameters(ds)
        pie_chart(ds, chosen)
    elif choice == "3":
        chosen = prompt_select_parameters(ds, allow_empty=False,
                                           prompt_text="Enter ONE parameter name: ")
        if chosen:
            bins = input("Number of bins (default 10): ").strip()
            bins = int(bins) if bins.isdigit() else 10
            histogram(ds, chosen[0], bins=bins)
    elif choice == "4":
        print("Available parameters:", ", ".join(ds.parameters()))
        px = input("X-axis parameter: ").strip().upper()
        py = input("Y-axis parameter: ").strip().upper()
        if px in ds.parameters() and py in ds.parameters():
            scatter_plot(ds, px, py)
        else:
            print("Invalid parameter name(s).")
    elif choice == "5":
        chosen = prompt_select_parameters(ds)
        line_plot(ds, chosen)
    elif choice == "6":
        chosen = prompt_select_parameters(ds, allow_empty=False,
                                           prompt_text="Enter ONE parameter name: ")
        if chosen:
            violin_plot(ds, chosen[0])
    elif choice == "7":
        chosen = prompt_select_parameters(ds)
        if not chosen:
            print("No parameters selected for correlation.")
        else:
            method = input("Correlation method (pearson/spearman/kendall) [pearson]: ").strip().lower() or "pearson"
            if method not in ("pearson", "spearman", "kendall"):
                print("Invalid method; using 'pearson'.")
                method = "pearson"
            correlation_heatmap(ds, params=chosen, method=method)
    elif choice == "8":
        chosen = prompt_select_parameters(ds)
        if not chosen:
            print("No parameters selected for pair plot.")
        else:
            pair_plot(ds, chosen)
    elif choice == "9":
        chosen = prompt_select_parameters(ds)
        if not chosen:
            print("No parameters selected for box plot.")
        else:
            group = input("Grouping column (categorical) to split boxes (or press Enter for none): ").strip().upper()
            if group == "":
                group = None
            elif group not in ds.parameters():
                print(f"Grouping column '{group}' not found; proceeding without grouping.")
                group = None
            box_plot(ds, params=chosen, by=group)
    elif choice == "0":
        return
    else:
        print("Invalid option.")


def main():
    ds = DataStore()

    print("=" * 60)
    print(" MORPHOMETRIC DATA VISUALIZATION & STATISTICS TOOL")
    print("=" * 60)

    while True:
        print("\nMAIN MENU")
        print("1. Add data manually")
        print("2. Load data from CSV")
        print("3. View current data")
        print("4. Remove a parameter")
        print("5. Show descriptive statistics")
        print("6. Export statistics to CSV")
        print("7. Visualize data")
        print("0. Exit")

        choice = input("Select option: ").strip()

        if choice == "1":
            menu_add_manual_data(ds)
        elif choice == "2":
            menu_load_csv(ds)
        elif choice == "3":
            menu_view_data(ds)
        elif choice == "4":
            menu_remove_parameter(ds)
        elif choice == "5":
            print_stats_table(ds)
        elif choice == "6":
            export_stats_csv(ds)
        elif choice == "7":
            menu_visualize(ds)
        elif choice == "0":
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid option, try again.")


if __name__ == "__main__":
    main()
