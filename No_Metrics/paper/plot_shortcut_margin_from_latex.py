import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "all_figures"


def clean_latex_number(s: str):
    """Convert LaTeX table numeric cell to float or NaN."""
    s = s.strip()
    s = re.sub(r"\\cellcolor\{[^}]+\}", "", s)
    s = s.replace("$", "")
    s = s.replace("{", "").replace("}", "")
    s = s.replace("\\", "")
    s = s.replace("+", "")
    s = s.strip()

    if s in {"--", "", "-"}:
        return float("nan")

    try:
        return float(s)
    except ValueError:
        return float("nan")


def strip_latex_model_name(header: str) -> str:
    """Convert LaTeX multirow model header into a readable model name."""
    s = header

    # Remove LaTeX wrappers.
    s = re.sub(r"\\multirow\{[^}]+\}\{[^}]+\}\{", "", s)
    s = re.sub(r"\\makecell\[l\]\{", "", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = s.replace("\\\\", " ")
    s = s.replace("\\", "")
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", " ", s).strip()

    # Normalize labels used in the figures.
    s = s.replace("(Case B)", "").replace("(Case C)", "").strip()

    # The source-table headers also carry archive-specific boundary annotations.
    # They are useful in the numeric tables but make six-panel plot titles
    # unreadable after the figure is scaled to a two-column paper width.
    s = re.sub(r"\s*\(\$N\^\*=.*?\$\)", "", s).strip()

    replacements = {
        "GPT-5": "GPT-5",
        "Opus 4.7 (med. think)": "Opus 4.7\n(med. think)",
        "DeepSeek- Reasoner": "DeepSeek-Reasoner",
        "Qwen3-14B (think)": "Qwen3-14B\n(think)",
        "Qwen3-32B (think)": "Qwen3-32B\n(think)",
        "QwQ-32B (think)": "QwQ-32B\n(think)",
    }

    if s.startswith("Qwen3-14B"):
        return "Qwen3-14B"
    if s.startswith("Qwen3-32B"):
        return "Qwen3-32B"
    if s.startswith("QwQ-32B"):
        return "QwQ-32B"
    if s.startswith("gpt-oss-20b"):
        return "gpt-oss-20b"
    if s.startswith("Nemotron-H-47B"):
        return "Nemotron-H-47B"
    if s.startswith("Llama-Nemotron"):
        return "Llama-Nemotron-49B"

    return replacements.get(s, s)


def extract_table_block(tex: str, label: str) -> str:
    """
    Extract the table environment that contains \\label{label}.
    This is robust enough for the table* blocks used in the paper.
    """
    label_pat = rf"\\label\{{{re.escape(label)}\}}"
    m = re.search(label_pat, tex)
    if not m:
        raise ValueError(f"Could not find label: {label}")

    start = tex.rfind(r"\begin{table", 0, m.start())
    end = tex.find(r"\end{table", m.end())

    if start == -1 or end == -1:
        raise ValueError(f"Could not extract table environment for label: {label}")

    # Include the end of the table/table* environment line.
    end_line = tex.find("\n", end)
    if end_line == -1:
        end_line = len(tex)

    return tex[start:end_line]


def _header_column_index(table_tex: str):
    """
    Locate the numeric columns by reading the tabular header instead of
    assuming a fixed column count.  The source tables carry extra columns
    (MCC, G-mean, BalAcc) between the F1 pair and ADR; a fixed-offset parser
    silently plots the wrong quantities under the wrong legend labels.

    Returns a dict metric -> 0-based cell index in a data row *after* the
    leading empty cell (i.e. index 0 is N).
    """
    start = table_tex.find(r"\toprule")
    end = table_tex.find(r"\midrule", start)
    if start == -1 or end == -1:
        raise ValueError("Could not find table header (toprule/midrule).")
    header = table_tex[start:end]
    header_rows = [r for r in header.split(r"\\") if "&" in r]
    if not header_rows:
        raise ValueError("Could not find header rows.")

    def expand(row):
        cells = []
        for c in row.split("&"):
            m = re.search(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}", c, re.S)
            if m:
                n = int(m.group(1))
                cells.extend([m.group(2)] * n)
            else:
                cells.append(c)
        return [re.sub(r"\s+", " ", x).strip() for x in cells]

    row1 = expand(header_rows[0])
    row2 = expand(header_rows[1]) if len(header_rows) > 1 else [""] * len(row1)
    if len(row2) < len(row1):
        row2 = row2 + [""] * (len(row1) - len(row2))

    def norm(x):
        x = re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}\{(.*)\}", r"\1", x)
        x = x.replace("$", "").replace("\\mathrm", "").replace("\\", "")
        x = x.replace("{", "").replace("}", "").replace(" ", "")
        return x

    names = [norm(a) + "|" + norm(b) for a, b in zip(row1, row2)]

    def find(pred, what):
        hits = [i for i, n in enumerate(names) if pred(n)]
        if len(hits) != 1:
            raise ValueError(f"Header column for {what} not unique: {hits} in {names}")
        return hits[0]

    col = {}
    col["N"] = find(lambda n: n.startswith("N|"), "N")
    col["Acc"] = find(lambda n: n.startswith("Acc|"), "Acc")
    col["F1SAT"] = find(lambda n: n.startswith("F1(perclass)|F1_SAT"), "F1SAT")
    col["F1UNSAT"] = find(lambda n: n.startswith("F1(perclass)|F1_UNSAT"), "F1UNSAT")
    col["ADR"] = find(lambda n: n.startswith("ADR|"), "ADR")
    col["ADRw"] = find(lambda n: n.startswith("ADR^+w|"), "ADRw")
    col["ShortcutCeiling"] = find(lambda n: n.startswith("shortcutCeiling|"), "ShortcutCeiling")
    col["shat"] = find(lambda n: n.startswith("hats|") or n.startswith("hat{s}|"), "shat")
    col["NormalizedPoint"] = find(lambda n: n.startswith("RMC(noLCB)|nRMC"), "NormalizedPoint")
    col["NormalizedCP"] = find(lambda n: n.startswith("RMC(CP)|nRMC"), "NormalizedCP")
    col["NormalizedWilson"] = find(lambda n: n.startswith("RMC(Wilson)|nRMC"), "NormalizedWilson")

    # Data rows start with an empty cell (the multirow model cell), so the
    # header index of "Model" (0) maps to that empty cell; shift by one.
    return {k: v - 1 for k, v in col.items()}


def parse_case_table(table_tex: str):
    """
    Parse Case B / Case C table rows, locating columns from the header.

    Plotted quantities:
      Acc, F1_SAT, F1_UNSAT, ADR, ADR^{+w}, shortcut ceiling, shat,
      normalized point margin, normalized CP margin, normalized Wilson margin.
    """
    col = _header_column_index(table_tex)
    metrics = [k for k in col if k != "N"]
    min_cells = max(col.values()) + 1

    data = {}
    current_model = None

    for line in table_tex.splitlines():
        raw = line.strip()
        if not raw:
            continue

        if r"\multirow" in raw and r"\makecell" in raw:
            current_model = strip_latex_model_name(raw)
            data[current_model] = {"N": []}
            for m in metrics:
                data[current_model][m] = []
            continue

        if current_model and raw.startswith("&"):
            row = raw.split("%")[0]
            row = row.replace(r"\\", "").strip()
            cells = [c.strip() for c in row.split("&")][1:]
            if len(cells) < min_cells:
                continue
            try:
                N = int(clean_latex_number(cells[col["N"]]))
            except Exception:
                continue
            data[current_model]["N"].append(N)
            for m in metrics:
                data[current_model][m].append(clean_latex_number(cells[col[m]]))

    return {k: v for k, v in data.items() if v["N"]}


def filter_out_n(data, excluded_n_values):
    """Return a copy of parsed table data with selected N values removed."""
    excluded = set(excluded_n_values)
    filtered = {}

    for model, vals in data.items():
        keep_indices = [i for i, n_val in enumerate(vals["N"]) if n_val not in excluded]
        filtered[model] = {
            metric: [series[i] for i in keep_indices]
            for metric, series in vals.items()
        }

    return filtered


def plot_case(data, out_pdf: Path, out_png: Path, show_legend: bool):
    """
    Create one-row, three-panel figure.
    Existing metrics are dashed.
    Shortcut-aware criteria are solid.
    """
    if not data:
        raise ValueError("No data to plot.")

    figure_height = 5.6 if show_legend else 4.9
    fig, axes = plt.subplots(
        1, len(data), figsize=(15.0, figure_height), sharey=True
    )

    if len(data) == 1:
        axes = [axes]

    existing_styles = {
        "Acc": {
            "color": "#1f77b4",
            "linestyle": "--",
            "marker": "o",
            "linewidth": 2.0,
            "markersize": 5.2,
        },
        "F1SAT": {
            "color": "#ff7f0e",
            "linestyle": "--",
            "marker": "s",
            "linewidth": 2.0,
            "markersize": 5.2,
        },
        "F1UNSAT": {
            "color": "#ff7f0e",
            "linestyle": "--",
            "marker": "^",
            "linewidth": 2.0,
            "markersize": 5.2,
        },
        "ADR": {
            "color": "#2ca02c",
            "linestyle": "--",
            "marker": "D",
            "linewidth": 2.0,
            "markersize": 5.2,
        },
        "ADRw": {
            "color": "#2ca02c",
            "linestyle": "--",
            "marker": "X",
            "linewidth": 2.0,
            "markersize": 5.5,
        },
    }

    margin_styles = {
        "shat": {
            "color": "#7f7f7f",
            "linestyle": "-",
            "marker": "*",
            "linewidth": 2.2,
            "markersize": 8.0,
        },
        "ShortcutCeiling": {
            "color": "#7f7f7f",
            "linestyle": "-",
            "marker": "D",
            "linewidth": 2.2,
            "markersize": 5.6,
        },
        "NormalizedPoint": {
            "color": "#9467bd",
            "linestyle": "-",
            "marker": "o",
            "linewidth": 2.2,
            "markersize": 5.2,
        },
        "NormalizedCP": {
            "color": "#d62728",
            "linestyle": "-",
            "marker": "s",
            "linewidth": 2.2,
            "markersize": 5.2,
        },
        "NormalizedWilson": {
            "color": "#8c564b",
            "linestyle": "-",
            "marker": "^",
            "linewidth": 2.2,
            "markersize": 5.2,
        },
    }

    labels = {
        "Acc": "Acc",
        "F1SAT": r"F1$_{SAT}$",
        "F1UNSAT": r"F1$_{UNSAT}$",
        "ADR": "ADR",
        "ADRw": r"ADR$^{+w}$",
        "ShortcutCeiling": r"$C_{\mathrm{sc}}^{G}$",
        "shat": r"$\hat{s}$",
        "NormalizedPoint": "Normalized margin (point)",
        "NormalizedCP": "Normalized margin (CP)",
        "NormalizedWilson": "Normalized margin (Wilson)",
    }

    for ax, (model, vals) in zip(axes, data.items()):
        x = vals["N"]

        for metric, style in existing_styles.items():
            ax.plot(x, vals[metric], **style)

        for metric, style in margin_styles.items():
            ax.plot(x, vals[metric], **style)

        title_size = 9 if len(data) > 3 else 12
        ax.set_title(model, fontsize=title_size)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.25)
        ax.set_xticks(x if len(data) <= 3 else [3, 10, 25, 40, 60])
        ax.tick_params(axis="both", labelsize=8 if len(data) > 3 else 9)
        ax.set_xlabel(r"$N$", fontsize=11)

    axes[0].set_ylabel("Score", fontsize=11)

    existing_handles = [
        Line2D([0], [0], **existing_styles[k])
        for k in ["Acc", "F1SAT", "F1UNSAT", "ADR", "ADRw"]
    ]
    existing_labels = [
        labels[k] for k in ["Acc", "F1SAT", "F1UNSAT", "ADR", "ADRw"]
    ]

    margin_handles = [
        Line2D([0], [0], **margin_styles[k])
        for k in ["shat", "ShortcutCeiling", "NormalizedPoint", "NormalizedCP", "NormalizedWilson"]
    ]
    margin_labels = [
        labels[k] for k in ["shat", "ShortcutCeiling", "NormalizedPoint", "NormalizedCP", "NormalizedWilson"]
    ]

    if show_legend:
        leg1 = fig.legend(
            existing_handles,
            existing_labels,
            loc="lower center",
            ncol=5,
            frameon=False,
            bbox_to_anchor=(0.5, 0.105),
            fontsize=10,
            title="Existing metrics (dashed)",
        )
        plt.setp(leg1.get_title(), fontsize=10)

        leg2 = fig.legend(
            margin_handles,
            margin_labels,
            loc="lower center",
            ncol=5,
            frameon=False,
            bbox_to_anchor=(0.5, 0.005),
            fontsize=10,
            title="Shortcut-aware criteria (solid)",
        )
        plt.setp(leg2.get_title(), fontsize=10)
        fig.add_artist(leg1)
        bottom = 0.27
    else:
        bottom = 0.0

    fig.tight_layout(rect=[0.01, bottom, 1.0, 0.98], w_pad=1.2)

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python plot_caseb_from_latex.py <supplementary_or_main_tex> [output_dir]"
        )
        sys.exit(1)

    tex_path = Path(sys.argv[1])
    tex = tex_path.read_text(encoding="utf-8")
    out_dir = Path(sys.argv[2]) if len(sys.argv) >= 3 else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1 retains the original Case B trajectory exports.  Figure 2 is a
    # direct Case C-minus-Case B sensitivity comparison generated from the
    # full-precision CSV outputs by plot_shortcut_residualization_sensitivity.py.
    table_specs = [
        ("tab:com_pern_caseB", ".", "fig_3sat_caseb_byN.pdf", "fig_3sat_caseb_byN.png", [], False),
        ("tab:os_pern_caseB", ".", "fig_2sat_caseb_byN.pdf", "fig_2sat_caseb_byN.png", [20], True),
    ]

    for label, subdir_name, out_pdf, out_png, excluded_n_values, show_legend in table_specs:
        block = extract_table_block(tex, label)
        data = parse_case_table(block)
        if excluded_n_values:
            data = filter_out_n(data, excluded_n_values)

        print(f"\nParsed {label}:")
        for model, vals in data.items():
            print(f"  {model}: {len(vals['N'])} rows, N={vals['N']}")

        case_out_dir = out_dir / subdir_name
        case_out_dir.mkdir(parents=True, exist_ok=True)
        out_pdf_path = case_out_dir / out_pdf
        out_png_path = case_out_dir / out_png
        plot_case(data, out_pdf_path, out_png_path, show_legend=show_legend)
        print(f"Saved {out_pdf_path} and {out_png_path}")


if __name__ == "__main__":
    main()
