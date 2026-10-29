#!/usr/bin/env python3
"""Camera-ready Figure 1 for Ghost Carbon, drawn from results/emissions.csv.

Does NOT re-run the experiment (timings are not seed-deterministic), so the
paper's numbers stay frozen. Writes PNG (300 dpi), PDF and SVG.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OKABE_ITO = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "blue": "#0072B2",
             "verm": "#D55E00", "purple": "#CC79A7", "black": "#000000", "grey": "#999999"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def save(fig, stem):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(R / f"{stem}.{ext}", dpi=300)
    plt.close(fig)


def main():
    em = pd.read_csv(R / "emissions.csv")
    em[["family", "phase"]] = em["phase_model"].str.split("_", expand=True)
    fams = ["RF", "LR", "MLP"]
    mg = lambda f, p: float(em[(em.family == f) & (em.phase == p)]["emissions_kg"].iloc[0]) * 1e6  # kg -> mg
    train = np.array([mg(f, "train") for f in fams]); infer = np.array([mg(f, "infer") for f in fams])
    total = train + infer
    claim = np.array([float(em[em.family == f]["claim_kg"].sum()) * 1e6 for f in fams])
    meter = em["meter"].iloc[0]

    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    x = np.arange(len(fams)); w = 0.36
    ax.bar(x - w / 2, claim, w, color="white", edgecolor=OKABE_ITO["grey"], lw=0.8)
    for xi in x:  # zero-height claim bars: draw a visible baseline tick + "0"
        ax.plot([xi - w, xi], [0, 0], color=OKABE_ITO["grey"], lw=2.5, solid_capstyle="butt")
        ax.text(xi - w / 2, total.max() * 0.025, "0", ha="center", va="bottom", fontsize=9, color="#555555")
    ax.bar(x + w / 2, train, w, color=OKABE_ITO["blue"])
    ax.bar(x + w / 2, infer, w, bottom=train, color=OKABE_ITO["orange"], hatch="..", edgecolor="white", lw=0)
    for xi, t in zip(x, total):
        ax.text(xi + w / 2, t + total.max() * 0.02, f"{t:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["Random forest", "Logistic reg.", "Small MLP"])
    ax.set_ylabel("Train + inference emissions\n(mg CO$_2$e; 1 mg = 10$^{-6}$ kg)")
    ax.set_ylim(0, total.max() * 1.45)
    handles = [Patch(facecolor="white", edgecolor=OKABE_ITO["grey"], label="Reported in claim (= 0, unreported)"),
               Patch(facecolor=OKABE_ITO["blue"], label="Metered: train"),
               Patch(facecolor=OKABE_ITO["orange"], hatch="..", edgecolor="white", label="Metered: inference")]
    ax.legend(handles=handles, loc="upper right", frameon=False, handlelength=1.4)
    ax.text(0.52, 0.50, f"Ghost gap = 100% of metered\n(meter: {meter})", transform=ax.transAxes,
            ha="center", va="top", fontsize=9, color="#333333")
    ax.yaxis.grid(True, lw=0.4, color="#dddddd"); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "claim_vs_metered")
    print("wrote", R / "claim_vs_metered.{png,pdf,svg}")


if __name__ == "__main__":
    main()
