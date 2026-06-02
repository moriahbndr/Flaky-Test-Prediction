import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(18, 14))
ax.set_xlim(0, 18)
ax.set_ylim(0, 14)
ax.axis("off")

# colour key  #
C_DATA    = "#D0E8FF"   # light blue  — raw input data
C_STEP    = "#E8F5E9"   # light green — pipeline step
C_FEAT    = "#FFF9C4"   # light gold  — processed feature sets
C_EXP     = "#FCE4EC"   # light pink  — experiment / model
C_OUT     = "#F3E5F5"   # light purple — outputs
C_EDGE    = "#37474F"   # dark slate  — all borders / arrows
C_TITLE   = "#1A237E"   # dark navy   — section titles


def box(ax, x, y, w, h, label, sublabel=None, color=C_STEP, fontsize=9, bold=False):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        linewidth=1.2,
        edgecolor=C_EDGE,
        facecolor=color,
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    cy = y + h / 2 + (0.15 if sublabel else 0)
    ax.text(x + w / 2, cy, label, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color="#1A1A1A")
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.22, sublabel, ha="center", va="center",
                fontsize=7, color="#444444", style="italic")


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=C_EDGE, lw=1.4),
    )


def section_label(ax, x, y, text):
    ax.text(x, y, text, fontsize=8.5, fontweight="bold",
            color=C_TITLE, va="center", ha="left")


ax.text(9, 13.55, "Flaky Test Prediction — Model Pipeline",
        ha="center", va="center", fontsize=15, fontweight="bold", color=C_TITLE)

# ROW 0 — Raw input data  (y ≈ 12.3)
section_label(ax, 0.2, 12.75, "INPUT DATA")

box(ax,  1.2, 12.0, 3.8, 0.9, "flakeflagger_results.csv",
    "FlakeFlagger dataset\n(pass/fail run history + test names)",
    color=C_DATA, fontsize=8.5)
box(ax, 12.8, 12.0, 3.8, 0.9, "idFlakies_dataset.csv",
    "iDFlakies dataset\n(confirmed-flaky tests, 26 projects)",
    color=C_DATA, fontsize=8.5)

# ROW 1 — Label verification  (y ≈ 10.6)
section_label(ax, 0.2, 11.35, "STEP 0")

box(ax, 1.8, 10.7, 6.5, 0.9, "label_verification.py",
    "Checks raw FlakeFlagger labels for run-history inconsistencies",
    color=C_STEP, fontsize=8.5)

# arrow: flakeflagger CSV → label_verification
arrow(ax, 3.1, 12.0, 4.2, 11.6)

# ROW 2 — Feature extraction  (y ≈ 9.0)
section_label(ax, 0.2, 10.0, "STEPS 1–3")

# build_flakeflagger.py
box(ax, 0.3, 9.0, 5.4, 0.9, "build_flakeflagger.py  (Step 1)",
    "Static features (26) + Dynamic features (17)  →  full_features.csv",
    color=C_STEP, fontsize=8)
# build_smells.py
box(ax, 6.3, 9.0, 5.4, 0.9, "build_smells.py  (Step 2)",
    "Test-smell features (10) fetched from FF repo  →  flakeflagger_features.csv",
    color=C_STEP, fontsize=8)
# build_idflakies.py
box(ax, 12.3, 9.0, 5.4, 0.9, "build_idflakies.py  (Step 3)",
    "Static features (26) only  →  idflakies_features.csv",
    color=C_STEP, fontsize=8)

# arrows: label_verification → build_flakeflagger & build_smells
arrow(ax, 3.8, 10.7, 2.5, 9.9)
arrow(ax, 5.0, 10.7, 8.5, 9.9)
# arrow: idFlakies CSV → build_idflakies
arrow(ax, 14.7, 12.0, 15.0, 9.9)

# ROW 3 — Processed feature stores  (y ≈ 7.4)
section_label(ax, 0.2, 8.4, "PROCESSED DATA")

box(ax, 0.3, 7.4, 5.4, 0.85, "full_features.csv",
    "Project · Test · IsFlaky · 26 static + 17 dynamic cols",
    color=C_FEAT, fontsize=8)
box(ax, 6.3, 7.4, 5.4, 0.85, "flakeflagger_features.csv",
    "Project · Test · 10 smell feature cols",
    color=C_FEAT, fontsize=8)
box(ax, 12.3, 7.4, 5.4, 0.85, "idflakies_features.csv",
    "Project · Test · IsFlaky=1 · 26 static cols",
    color=C_FEAT, fontsize=8)

arrow(ax, 3.0, 9.0, 3.0, 8.25)
arrow(ax, 9.0, 9.0, 9.0, 8.25)
arrow(ax, 15.0, 9.0, 15.0, 8.25)

# ROW 4 — Model training + cross-validation header  (y ≈ 6.3)
section_label(ax, 0.2, 7.0, "STEP 4  —  MODEL TRAINING")

box(ax, 1.5, 6.0, 15.0, 0.85,
    "model_training.py  —  XGBoost (binary:logistic) · 10-fold Stratified CV",
    sublabel="scale_pos_weight balances class imbalance · threshold = 0.80 (tuned per fold for smell_only)",
    color=C_STEP, fontsize=9, bold=True)

# arrows: processed CSVs → model_training
arrow(ax, 3.0, 7.4, 4.5, 6.85)
arrow(ax, 9.0, 7.4, 9.0, 6.85)
arrow(ax, 15.0, 7.4, 13.5, 6.85)

# ROW 5 — Experiments  (y ≈ 4.5)
section_label(ax, 0.2, 5.65, "EXPERIMENTS")

EXP_Y = 4.5
EXP_H = 0.95
exp_data = [
    (0.2,  "static\n(26 features)",          "Name-structure &\nkeyword flags only"),
    (4.7,  "static_plus_dynamic\n(43 features)", "static + 17 run-history\nderived features  ★ saved"),
    (9.2,  "flakeflagger_raw\n(5 features)",  "Baseline: raw pass/\nfail counts"),
    (13.7, "smell_only\n(10 features)",        "Baseline: test-smell\ndetectors"),
]
for x0, title, sub in exp_data:
    box(ax, x0, EXP_Y, 4.1, EXP_H, title, sub, color=C_EXP, fontsize=8)

for x0 in [2.25, 6.75, 11.25, 15.75]:
    arrow(ax, x0, 6.0, x0, 5.45)

# ROW 6 — Cross-project eval  (y ≈ 3.1)
section_label(ax, 0.2, 4.2, "CROSS-PROJECT EVAL")

box(ax, 0.2, 3.1, 8.8, 0.85,
    "cross_project_eval()  —  static · static_plus_dynamic · smell_only",
    sublabel="Models trained on FlakeFlagger evaluated on iDFlakies (unseen projects) — measures generalization",
    color=C_STEP, fontsize=8.5)

# arrows: static, static_plus_dynamic, smell_only → cross_project_eval
arrow(ax, 2.25, 4.5, 2.5, 3.95)
arrow(ax, 6.75, 4.5, 5.0, 3.95)
arrow(ax, 15.75, 4.5, 8.5, 3.95)

# ROW 7 — Outputs  (y ≈ 1.5)
section_label(ax, 0.2, 2.8, "OUTPUTS")

out_items = [
    (0.2,  "model_metrics.csv",            "Precision / Recall / F1\nBrier Score per experiment"),
    (4.7,  "cross_project_metrics.csv",    "Recall + TP/FN counts\non iDFlakies per experiment"),
    (9.2,  "xgboost_model.pkl",            "Serialised static_plus_dynamic\nmodel (joblib)"),
    (13.7, "Feature importance\n& calibration plots", "Per-experiment PNGs\nin results/figures/"),
]
for x0, title, sub in out_items:
    box(ax, x0, 1.5, 4.1, 1.0, title, sub, color=C_OUT, fontsize=8)

# arrows: experiments → outputs
for src_x, dst_x in [(2.25, 2.25), (6.75, 6.75), (11.25, 11.25), (15.75, 15.75)]:
    arrow(ax, src_x, 4.5, dst_x, 2.5)

# arrow: cross_project → cross_project_metrics
arrow(ax, 4.6, 3.1, 6.75, 2.5)

# legend  #
legend_items = [
    (C_DATA, "Raw input data"),
    (C_STEP, "Pipeline step / script"),
    (C_FEAT, "Processed feature store"),
    (C_EXP,  "Experiment (feature set)"),
    (C_OUT,  "Output artifact"),
]
legend_patches = [
    mpatches.Patch(facecolor=c, edgecolor=C_EDGE, label=lbl)
    for c, lbl in legend_items
]
ax.legend(handles=legend_patches, loc="lower right",
          fontsize=8, framealpha=0.9, edgecolor=C_EDGE)

plt.tight_layout()
out_path = "results/figures/pipeline_diagram.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
