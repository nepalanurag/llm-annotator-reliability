"""Publication-quality figures for the LLM-annotator study (matplotlib)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "DejaVu Sans",
})


def confusion_matrix_fig(y_true, y_pred, labels, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
    fig.colorbar(im, ax=ax, label="Fraction of true class")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix: LLM vs PubMed topic labels (row-normalized)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def calibration_fig(bins, ece, path):
    x = [b["mean_confidence"] for b in bins]
    y = [b["accuracy"] for b in bins]
    n = [b["n"] for b in bins]
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="Perfect calibration")
    sizes = [10 + 90 * v / max(n) for v in n]
    sc = ax.scatter(x, y, s=sizes, color="#1f77b4", alpha=0.85, zorder=3,
                    label="Binned predictions")
    _calib_labels(ax, bins)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence in bin")
    ax.set_ylabel("Observed accuracy in bin")
    ax.set_title(f"Calibration curve (ECE = {ece:.3f})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _calib_labels(ax, bins):
    # Stagger labels so they never collide, even when bins crowd the top.
    for k, b in enumerate(bins):
        x, y = b["mean_confidence"], b["accuracy"]
        if x > 0.75:
            # top bins: label below the point to stay clear of the title
            ax.annotate(f"n={b['n']}", (x, y), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=8,
                        color="#333333")
        else:
            xytext = (4, 6 if k % 2 == 0 else -16)
            ax.annotate(f"n={b['n']}", (x, y), textcoords="offset points",
                        xytext=xytext, ha="left", fontsize=8, color="#333333")


def per_class_fig(metrics_dict, labels, path):
    x = np.arange(len(labels))
    prec = [metrics_dict[l]["precision"] for l in labels]
    rec = [metrics_dict[l]["recall"] for l in labels]
    f1 = [metrics_dict[l]["f1"] for l in labels]
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, prec, w, label="Precision", color="#1f77b4")
    ax.bar(x, rec, w, label="Recall", color="#ff7f0e")
    ax.bar(x + w, f1, w, label="F1", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class precision, recall, and F1 (LLM annotator)")
    ax.legend()
    for i, lab in enumerate(labels):
        ax.text(i, 1.0, f"n={metrics_dict[lab]['support']}", ha="center",
                fontsize=8, color="#555555")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cost_fig(llm_cost, human_cost, path, llm_label="LLM (Gemini Flash-Lite)",
             human_label="Human annotation (estimate)"):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.bar([llm_label, human_label], [llm_cost, human_cost],
           color=["#1f77b4", "#d62728"])
    ax.set_ylabel("Cost per 1,000 annotations (USD)")
    ax.set_title("Annotation cost: LLM vs human estimate")
    for i, v in enumerate([llm_cost, human_cost]):
        ax.text(i, v, f"${v:.2f}", ha="center", va="bottom", fontsize=11)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
