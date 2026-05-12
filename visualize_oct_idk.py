"""
Visualises the OCT-IDK (Abstain) results from oct_idk.jl.

Can also run in demo mode (no julia required) to show the concept.

Usage:
    python3 visualize_oct_idk.py               # reads oct_idk_results.json
    python3 visualize_oct_idk.py --demo        # hardcoded demo data
"""

import argparse, json, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# ── palette ───────────────────────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
TEXT    = "#c9d1d9"
DIM     = "#8b949e"
GREEN   = "#3fb950"   # benign
RED     = "#f85149"   # malignant
AMBER   = "#d29922"   # IDK
BLUE    = "#58a6ff"
PURPLE  = "#bc8cff"


# ── demo / fallback data  (D=3 tree) ─────────────────────────────────────────
DEMO_DATA = {
    "demo_beta": 0.3,
    "n": 150,
    "D": 3,
    "p": 5,
    "demo_splits": {
        "1": {"feat": "concave_pts_mean",  "thr": 0.0555},
        "2": {"feat": "area_worst",        "thr": 876.5},
        "3": {"feat": "concave_pts_worst", "thr": 0.1374},
        "4": {"feat": "radius_worst",      "thr": 16.50},
        "5": {"feat": "texture_worst",     "thr": 25.10},
        "6": {"feat": "perimeter_worst",   "thr": 105.0},
        "7": {"feat": "area_worst",        "thr": 1200.0},
    },
    "demo_leaves": {
        "8":  {"pred": "Benign",    "idk": False, "count": 68, "active": True},
        "9":  {"pred": "Malignant", "idk": False, "count": 10, "active": True},
        "10": {"pred": "IDK",       "idk": True,  "count": 10, "active": True},
        "11": {"pred": "Malignant", "idk": False, "count":  8, "active": True},
        "12": {"pred": "Benign",    "idk": False, "count":  4, "active": True},
        "13": {"pred": "IDK",       "idk": True,  "count":  6, "active": True},
        "14": {"pred": "Benign",    "idk": False, "count":  4, "active": True},
        "15": {"pred": "Malignant", "idk": False, "count": 40, "active": True},
    },
    "demo_metrics": {
        "coverage": 89.3, "acc_decided": 97.0, "acc_overall": 86.7,
        "n_idk": 16, "TP": 56, "TN": 74, "FP": 2, "FN": 2,
    },
    "sweep": [
        {"beta": 0.1, "coverage": 73.3, "acc_decided": 99.1, "acc_overall": 72.7, "n_idk": 40},
        {"beta": 0.2, "coverage": 82.0, "acc_decided": 98.8, "acc_overall": 81.1, "n_idk": 27},
        {"beta": 0.3, "coverage": 89.3, "acc_decided": 97.0, "acc_overall": 86.7, "n_idk": 16},
        {"beta": 0.5, "coverage": 95.3, "acc_decided": 97.8, "acc_overall": 93.1, "n_idk":  7},
        {"beta": 0.8, "coverage": 100.0,"acc_decided": 97.3, "acc_overall": 97.3, "n_idk":  0},
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 1 — tree diagram
# ─────────────────────────────────────────────────────────────────────────────
def _node_color(pred, idk):
    if idk or pred == "IDK":
        return AMBER
    if pred == "Malignant":
        return RED
    return GREEN


def _build_positions(D):
    """Generate (x, y) positions for every node in a depth-D binary tree."""
    T_total  = 2 ** (D + 1) - 1
    n_leaves = 2 ** D
    slot = 4  # horizontal units allocated per leaf
    pos = {}
    for t in range(1, T_total + 1):
        depth = int(math.log2(t))
        pos_in_level = t - 2 ** depth
        level_width  = 2 ** depth
        x = (pos_in_level + 0.5) * slot * n_leaves / level_width
        y = (D + 1 - depth) * 3.0
        pos[t] = (x, y)
    xlim = slot * n_leaves
    ylim = (D + 2) * 3.0
    return pos, xlim, ylim


def draw_tree(ax, splits, leaves, beta, D=2):
    ax.set_facecolor(PANEL)
    ax.axis("off")
    ax.set_title(f"Decision Tree  —  Depth {D},  β = {beta}",
                 color=TEXT, fontsize=16, pad=10, fontweight="bold")

    pos, xlim, ylim = _build_positions(D)
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)

    T_B = list(range(1, 2**D))
    T_L = list(range(2**D, 2**(D+1)))

    leaf_slot = xlim / (2 ** D)
    h = 1.4

    def has_content(t):
        """True if this node or any descendant has something to display."""
        if t in T_L:
            return leaves.get(str(t), {}).get("active", False)
        return bool(splits.get(str(t))) or has_content(t*2) or has_content(t*2+1)

    def draw_node(t, is_leaf):
        x, y = pos[t]
        depth = int(math.log2(t))
        level_slot = xlim / (2 ** depth)
        w = min(level_slot * 0.80, leaf_slot * 2.2)

        if is_leaf:
            info   = leaves.get(str(t), {})
            pred   = info.get("pred", "inactive")
            idk    = info.get("idk", False)
            cnt    = info.get("count", 0)
            active = info.get("active", False)
            if active:
                color = _node_color(pred, idk)
                label = ("IDK" if (idk or pred == "IDK") else pred) + f"\n(n={cnt})"
            else:
                color = "#3a3a4a"
                label = "empty"
            fs = 20
        else:
            sp = splits.get(str(t), {})
            if sp:
                feat   = sp.get("feat", f"node {t}")
                thr    = sp.get("thr", 0)
                feat_s = feat.replace("_", " ")
                label  = f"{feat_s}\n≤ {thr:.3f}"
                color  = BLUE
            else:
                color = "#3a3a4a"
                label = "no split"
            fs = 20

        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.04",
                             facecolor=color, edgecolor=BORDER,
                             linewidth=1.0, alpha=0.92, zorder=3)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center",
                color="white", fontsize=fs, fontweight="bold",
                zorder=4, multialignment="center")

    # Draw all edges in the full tree
    for t in list(T_B):
        px, py = pos[t]
        for child in [t * 2, t * 2 + 1]:
            if child not in pos:
                continue
            cx, cy = pos[child]
            is_left = (child % 2 == 0)
            ax.annotate("", xy=(cx, cy + h/2), xytext=(px, py - h/2),
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=3.0,
                                       mutation_scale=20),
                        zorder=2)
            mx, my = (px + cx) / 2, (py + cy) / 2
            lbl    = "≤" if is_left else ">"
            offset = 1.5 if cx < px else -1.5
            ax.text(mx + offset, my, lbl,
                    ha="center", va="center", color="black", fontsize=28,
                    fontweight="bold")

    for t in T_B:
        draw_node(t, is_leaf=False)
    for t in T_L:
        draw_node(t, is_leaf=True)

    # Legend — bottom-left of panel
    legend_items = [
        mpatches.Patch(color=RED,   label="Malignant"),
        mpatches.Patch(color=GREEN, label="Benign"),
        mpatches.Patch(color=AMBER, label="IDK (abstain)"),
        mpatches.Patch(color=BLUE,  label="Split node"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              fontsize=12, facecolor=BG, edgecolor=BORDER,
              labelcolor=TEXT, framealpha=0.9)


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 2 — β sensitivity
# ─────────────────────────────────────────────────────────────────────────────
PINK = "#f2c4ce"

def draw_sensitivity(ax, sweep, beta_demo):
    ax.set_facecolor(PINK)
    betas    = [r["beta"]      for r in sweep]
    cov      = [r["coverage"]  for r in sweep]
    acc_dec  = [r["acc_decided"] for r in sweep]
    acc_all  = [r["acc_overall"] for r in sweep]
    idk_pct  = [100*r["n_idk"]/150 for r in sweep]

    ax.plot(betas, cov,     color=BLUE,   lw=2, marker="o", ms=5, label="Coverage %")
    ax.plot(betas, acc_dec, color=GREEN,  lw=2, marker="s", ms=5, label="Accuracy* (classified)")
    ax.plot(betas, acc_all, color=PURPLE, lw=2, marker="^", ms=5, linestyle="--",
            label="Accuracy† (IDK=wrong)")
    ax.plot(betas, idk_pct, color=AMBER,  lw=2, marker="D", ms=5, label="IDK patients %")

    # Mark demo β
    ax.axvline(beta_demo, color=AMBER, lw=1, linestyle=":", alpha=0.6)
    ax.text(beta_demo + 0.01, 55, f"β={beta_demo}", color="black", fontsize=25)

    ax.set_xlabel("IDK penalty β", color="black", fontsize=25)
    ax.set_ylabel("Percentage (%)", color="black", fontsize=25)
    ax.set_title("β Sensitivity: IDK Rate vs Accuracy", color="black", fontsize=27, pad=12)
    ax.set_ylim(0, 105)
    ax.tick_params(colors="black", labelsize=25)
    ax.spines[:].set_color("black")
    ax.yaxis.grid(True, color="#cccccc", linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=25, facecolor=PINK, edgecolor="black", labelcolor="black",
              loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    # Callout arrows explaining the two extremes
    ax.annotate("Low β: model abstains\noften — high IDK rate,\nhigh accuracy on rest",
                xy=(0.1, idk_pct[0]), xytext=(0.2, 40),
                arrowprops=dict(arrowstyle="->", color="black", lw=1),
                color="black", fontsize=25, ha="center")
    ax.annotate("High β: no IDK,\nsame as original OCT",
                xy=(betas[-1], cov[-1]), xytext=(0.65, 55),
                arrowprops=dict(arrowstyle="->", color="black", lw=1),
                color="black", fontsize=25, ha="center")


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 3 — metrics bar chart (demo β)
# ─────────────────────────────────────────────────────────────────────────────
def draw_metrics(ax, metrics, beta):
    ax.set_facecolor("white")
    ax.set_title(f"Model Performance at β = {beta}", color="black", fontsize=25, pad=12)

    tp, tn   = metrics["TP"], metrics["TN"]
    fp, fn   = metrics["FP"], metrics["FN"]
    n_idk    = metrics["n_idk"]
    cov      = metrics["coverage"]
    acc_dec  = metrics["acc_decided"]
    acc_all  = metrics["acc_overall"]
    n_total  = tp + tn + fp + fn + n_idk
    sens     = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec     = 100.0 * tn / (tn + fp) if (tn + fp) > 0 else 0.0

    labels = [
        "Coverage\n(classified)",
        "Accuracy\n(classified)",
        "Accuracy\n(all patients)",
        "Sensitivity\n(malignant)",
        "Specificity\n(benign)",
    ]
    values = [cov, acc_dec, acc_all, sens, spec]
    colors = [BLUE, GREEN, PURPLE, RED, AMBER]

    bars = ax.bar(labels, values, color=colors, alpha=0.88,
                  edgecolor="black", linewidth=1.0)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom",
                color="black", fontsize=20, fontweight="bold")

    ax.set_ylim(0, 125)
    ax.set_ylabel("Percentage (%)", color="black", fontsize=20)
    ax.tick_params(colors="black", labelsize=18)
    ax.spines[:].set_color("black")
    ax.yaxis.grid(True, color="#cccccc", linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)

    # Key stats box — placed below the axes
    stats_txt = (
        f"TP={tp}   FN={fn}   FP={fp}   TN={tn}   |   IDK referred: {n_idk}/{n_total}"
    )
    ax.figure.subplots_adjust(bottom=0.22)
    ax.figure.text(0.5, 0.04, stats_txt, ha="center", va="bottom",
                   fontsize=19, color="black", fontfamily="monospace",
                   bbox=dict(facecolor="#f5f5f5", edgecolor="black",
                             boxstyle="round,pad=0.5"))


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 4 — Pareto frontier: coverage vs accuracy as β varies
# ─────────────────────────────────────────────────────────────────────────────
def draw_pareto(ax, sweep, opt_beta=None):
    ax.set_facecolor("white")
    ax.set_title("Coverage–Accuracy Trade-off  (Pareto Frontier)",
                 color="black", fontsize=25, pad=12)

    cov     = [r["coverage"]    for r in sweep]
    acc_dec = [r["acc_decided"] for r in sweep]
    acc_all = [r["acc_overall"] for r in sweep]
    betas   = [r["beta"]        for r in sweep]
    idk_pct = [100*r["n_idk"]/150 for r in sweep]

    # Colour each point by IDK rate (low=green, high=amber)
    norm_idk = [(v - min(idk_pct)) / (max(idk_pct) - min(idk_pct) + 1e-9)
                for v in idk_pct]
    colors = [
        (int(0x3f + (0xd2-0x3f)*t, ), int(0xb9 + (0x99-0xb9)*t), int(0x50 + (0x22-0x50)*t))
        for t in norm_idk
    ]
    hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]

    # Line connecting the frontier
    ax.plot(cov, acc_dec, color="#aaaaaa", lw=1.5, linestyle="--", zorder=1)

    for i, (cx, ay, hc, beta) in enumerate(zip(cov, acc_dec, hex_colors, betas)):
        sc = ax.scatter(cx, ay, s=260, color=hc, edgecolors="black",
                        linewidths=1.5, zorder=3)
        ax.annotate(f"β={beta}", (cx, ay),
                    textcoords="offset points", xytext=(8, 6),
                    fontsize=18, color="black", fontweight="bold")

    # Mark the optimal point — use opt_beta if provided, else highest acc_all
    if opt_beta is not None and opt_beta in betas:
        best_i = betas.index(opt_beta)
    else:
        best_i = int(np.argmax(acc_all))
    ax.scatter(cov[best_i], acc_dec[best_i], s=500, color="none",
               edgecolors=RED, linewidths=3, zorder=4)
    ax.annotate("optimal", (cov[best_i], acc_dec[best_i]),
                textcoords="offset points", xytext=(-70, -28),
                fontsize=18, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

    ax.set_xlabel("Coverage  (% patients classified)", color="black", fontsize=22)
    ax.set_ylabel("Accuracy on classified patients (%)", color="black", fontsize=22)
    ax.tick_params(colors="black", labelsize=20)
    ax.spines[:].set_color("black")
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.6, linestyle="--")
    ax.xaxis.grid(True, color="#dddddd", linewidth=0.6, linestyle="--")
    ax.set_axisbelow(True)

    # Colourbar legend (manual)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=GREEN,
               markeredgecolor='black', markersize=14, label='Low IDK rate (high β)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=AMBER,
               markeredgecolor='black', markersize=14, label='High IDK rate (low β)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor=RED, markeredgewidth=2.5,
               markersize=16, label='Optimal β'),
    ]
    ax.legend(handles=handles, fontsize=18, facecolor="white",
              edgecolor="black", labelcolor="black", loc="lower right")


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 5 — Confusion matrix heatmap
# ─────────────────────────────────────────────────────────────────────────────
def draw_confusion(ax, metrics, beta):
    ax.set_facecolor("white")
    ax.set_title(f"Confusion Matrix  (β = {beta},  classified patients only)",
                 color="black", fontsize=25, pad=12)

    tp, tn = metrics["TP"], metrics["TN"]
    fp, fn = metrics["FP"], metrics["FN"]
    n_idk  = metrics["n_idk"]
    total  = tp + tn + fp + fn

    cm = np.array([[tp, fn], [fp, tn]], dtype=float)
    im = ax.imshow(cm, cmap="RdYlGn", vmin=0, vmax=max(tp, tn) * 1.2, aspect="auto")

    labels = [["True Positive\n(TP)", "False Negative\n(FN — missed cancer!)"],
              ["False Positive\n(FP)", "True Negative\n(TN)"]]

    for i in range(2):
        for j in range(2):
            val  = int(cm[i, j])
            pct  = 100.0 * val / total
            txt  = f"{val}\n({pct:.1f}%)\n{labels[i][j]}"
            col  = "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=20, fontweight="bold", color=col)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nMalignant", "Predicted\nBenign"],
                       fontsize=20, color="black")
    ax.set_yticklabels(["Actual\nMalignant", "Actual\nBenign"],
                       fontsize=20, color="black")
    ax.tick_params(colors="black", length=0)
    ax.spines[:].set_color("black")

    ax.figure.text(0.5, 0.01,
                   f"IDK (referred for further testing): {n_idk} patients not shown above",
                   ha="center", fontsize=18, color=AMBER,
                   fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 6 — Cost comparison: Standard OCT vs OCT-IDK
# ─────────────────────────────────────────────────────────────────────────────
def draw_cost_comparison(ax, metrics_idk):
    """Stacked bar chart showing estimated medical cost reduction from IDK abstain."""
    ax.set_facecolor("white")

    # Cost assumptions (USD) — illustrative but clinically motivated
    COST_FN  = 50_000   # missed cancer → delayed treatment
    COST_FP  =  5_000   # unnecessary biopsy / treatment
    COST_IDK =  1_000   # specialist referral / further testing

    # Standard OCT (β=0.3, no IDK, 100% coverage, ~4 errors from sweep)
    std_fn, std_fp, std_idk = 2, 2, 0

    # OCT-IDK (β=0.2, from actual solver results)
    idk_fn  = metrics_idk["FN"]
    idk_fp  = metrics_idk["FP"]
    idk_idk = metrics_idk["n_idk"]

    std_cost_fn  = std_fn  * COST_FN
    std_cost_fp  = std_fp  * COST_FP
    std_cost_idk = std_idk * COST_IDK
    std_total    = std_cost_fn + std_cost_fp + std_cost_idk

    idk_cost_fn  = idk_fn  * COST_FN
    idk_cost_fp  = idk_fp  * COST_FP
    idk_cost_idk = idk_idk * COST_IDK
    idk_total    = idk_cost_fn + idk_cost_fp + idk_cost_idk

    savings = std_total - idk_total
    savings_pct = 100.0 * savings / std_total

    labels   = ["Standard OCT\n(β = 0.3, no IDK)", "OCT-IDK\n(β = 0.2, with abstain)"]
    fn_costs = [std_cost_fn,  idk_cost_fn]
    fp_costs = [std_cost_fp,  idk_cost_fp]
    id_costs = [std_cost_idk, idk_cost_idk]

    x = np.arange(len(labels))
    bar_w = 0.45

    b1 = ax.bar(x, fn_costs, bar_w, label=f"Missed Cancer — FN  (${COST_FN:,}/case)",
                color=RED, alpha=0.88, edgecolor="black", linewidth=1.2)
    b2 = ax.bar(x, fp_costs, bar_w, bottom=fn_costs,
                label=f"False Alarm — FP  (${COST_FP:,}/case)",
                color=AMBER, alpha=0.88, edgecolor="black", linewidth=1.2)
    b3 = ax.bar(x, id_costs, bar_w, bottom=[f+p for f,p in zip(fn_costs, fp_costs)],
                label=f"IDK Referral  (${COST_IDK:,}/patient)",
                color=BLUE, alpha=0.88, edgecolor="black", linewidth=1.2)

    # Totals above bars
    totals = [std_total, idk_total]
    for xi, tot in zip(x, totals):
        ax.text(xi, tot + 1500, f"${tot:,}", ha="center", va="bottom",
                fontsize=22, fontweight="bold", color="black")

    # Savings annotation between the two bars
    y_ann = max(std_total, idk_total) * 0.75
    ax.annotate("",
                xy=(x[1] + bar_w/2 + 0.05, idk_total),
                xytext=(x[0] - bar_w/2 - 0.05, std_total),
                arrowprops=dict(arrowstyle="<->", color="black", lw=2.0,
                                connectionstyle="arc3,rad=0.0"))
    mid_x = (x[0] + x[1]) / 2
    mid_y = (std_total + idk_total) / 2
    ax.text(mid_x, mid_y + 3000,
            f"Save ${savings:,}\n({savings_pct:.1f}% reduction)",
            ha="center", va="bottom", fontsize=22, fontweight="bold", color="black",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.4"))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=22, color="black")
    ax.set_ylabel("Estimated Medical Cost (USD)", color="black", fontsize=22)
    ax.set_title("Cost Reduction from IDK Abstain Leaves\n(illustrative cost assumptions)",
                 color="black", fontsize=24, pad=14)
    ax.set_ylim(0, std_total * 1.35)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"${int(v):,}"))
    ax.tick_params(colors="black", labelsize=18)
    ax.spines[:].set_color("black")
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.6, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=18, facecolor="white", edgecolor="black",
              labelcolor="black", loc="upper right")

    # Footnote with per-model breakdown
    breakdown = (
        f"Standard OCT:  FN={std_fn}×${COST_FN//1000}k + FP={std_fp}×${COST_FP//1000}k = ${std_total//1000}k  |  "
        f"OCT-IDK:  FN={idk_fn}×${COST_FN//1000}k + FP={idk_fp}×${COST_FP//1000}k + "
        f"IDK={idk_idk}×${COST_IDK//1000}k = ${idk_total//1000}k"
    )
    ax.figure.subplots_adjust(bottom=0.20)
    ax.figure.text(0.5, 0.04, breakdown, ha="center", fontsize=15, color="#444444",
                   fontfamily="monospace",
                   bbox=dict(facecolor="#f9f9f9", edgecolor="black",
                             boxstyle="round,pad=0.4"))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="Use built-in demo data")
    ap.add_argument("--out",  default="oct_idk_viz.png")
    args = ap.parse_args()

    json_path = os.path.join(os.path.dirname(__file__), "oct_idk_results.json")

    if args.demo or not os.path.exists(json_path):
        if not args.demo:
            print("[visualize] oct_idk_results.json not found — using demo data")
        data = DEMO_DATA
    else:
        with open(json_path) as f:
            data = json.load(f)
        print(f"[visualize] loaded results from {json_path}")

    base_dir = os.path.dirname(__file__)
    panels = [
        ("tree",        (22, 16), BG,      lambda fig: draw_tree(
            fig.add_subplot(111), data["demo_splits"], data["demo_leaves"],
            data["demo_beta"], D=data.get("D", 2))),
        ("pareto",      (16, 11), "white", lambda fig: draw_pareto(
            fig.add_subplot(111), data["sweep"], data.get("opt_beta"))),
        ("sensitivity", (24, 10), PINK,    lambda fig: draw_sensitivity(
            fig.add_subplot(111), data["sweep"], data["demo_beta"])),
        ("confusion",   (14, 10), "white", lambda fig: draw_confusion(
            fig.add_subplot(111), data["demo_metrics"], data["demo_beta"])),
        ("metrics",     (18, 10), "white", lambda fig: draw_metrics(
            fig.add_subplot(111), data["demo_metrics"], data["demo_beta"])),
        ("cost",        (16, 10), "white", lambda fig: draw_cost_comparison(
            fig.add_subplot(111), data["demo_metrics"])),
    ]

    for name, figsize, bg, draw_fn in panels:
        fig = plt.figure(figsize=figsize, facecolor=bg)
        draw_fn(fig)
        out = os.path.join(base_dir, f"oct_idk_{name}.png")
        plt.savefig(out, dpi=150, facecolor=bg, bbox_inches="tight")
        plt.close(fig)
        print(f"[visualize] saved → {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
