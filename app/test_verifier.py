import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# -----------------------------
# Figure
# -----------------------------
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

# -----------------------------
# Helper functions
# -----------------------------
def box(x, y, w, h, title, lines=None, fontsize=13):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1.8,
        fill=True
    )
    ax.add_patch(patch)

    ax.text(
        x + w/2, y + h - 0.35,
        title,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold"
    )

    if lines:
        ax.text(
            x + w/2, y + h/2 - 0.15,
            "\n".join(lines),
            ha="center",
            va="center",
            fontsize=10.5,
            linespacing=1.4
        )

def arrow(x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=18,
            linewidth=1.8
        )
    )

# -----------------------------
# Title
# -----------------------------
ax.text(
    8, 8.65,
    "Multimodal  Bias Correction — System Architecture",
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold"
)

# -----------------------------
# INPUT LAYER
# -----------------------------
box(
    0.5, 6.5, 2.4, 1.25,
    "MULTIMODAL INPUT",
    [
        "Image",
        "User Query"
    ]
)

# -----------------------------
# INITIAL GENERATION
# -----------------------------
box(
    3.5, 6.5, 2.5, 1.25,
    "INITIAL VLM",
    [
        "Vision-Language Model",
        "Initial Response"
    ]
)

arrow(2.9, 7.1, 3.5, 7.1)

# -----------------------------
# CLAIM EXTRACTION
# -----------------------------
box(
    6.6, 6.5, 2.6, 1.25,
    "CLAIM EXTRACTION",
    [
        "Response → Claims",
        "Observation / Inference",
        "Evidence Requirement"
    ]
)

arrow(6.0, 7.1, 6.6, 7.1)

# -----------------------------
# EVIDENCE CHECKING
# -----------------------------
box(
    9.8, 6.5, 2.6, 1.25,
    "EVIDENCE CHECKING",
    [
        "Claim ↔ Image",
        "SUPPORTED",
        "UNCERTAIN / UNSUPPORTED"
    ]
)

arrow(9.2, 7.1, 9.8, 7.1)

# -----------------------------
# BIAS MONITOR
# -----------------------------
box(
    13.0, 6.5, 2.5, 1.25,
    "BIAS MONITOR",
    [
        "Bias-Relevant?",
        "Category",
        "Severity"
    ]
)

arrow(12.4, 7.1, 13.0, 7.1)

# -----------------------------
# LOWER PIPELINE
# -----------------------------
box(
    11.0, 4.25, 3.0, 1.35,
    "SELF-CORRECTION",
    [
        "Remove unsupported claims",
        "Preserve supported information",
        "No new unsupported facts"
    ]
)

# Bias monitor → correction
arrow(14.25, 6.5, 12.5, 5.6)

# Evidence checker feedback to correction
arrow(11.0, 6.5, 11.5, 5.6)

# -----------------------------
# FINAL VERIFICATION
# -----------------------------
box(
    6.5, 4.25, 3.0, 1.35,
    "FINAL VERIFICATION",
    [
        "Re-check corrected response",
        "Residual problematic claims?",
        "Pass / Re-correct"
    ]
)

arrow(11.0, 4.9, 9.5, 4.9)

# -----------------------------
# FINAL RESPONSE
# -----------------------------
box(
    2.0, 4.25, 3.0, 1.35,
    "FINAL RESPONSE",
    [
        "Evidence-grounded",
        "Bias-aware",
        "Natural-language output"
    ]
)

arrow(6.5, 4.9, 5.0, 4.9)

# -----------------------------
# ITERATIVE LOOP
# -----------------------------
ax.add_patch(
    FancyArrowPatch(
        (8.0, 4.25),
        (8.0, 3.35),
        arrowstyle="->",
        mutation_scale=16,
        linewidth=1.5
    )
)

ax.text(
    8.2, 3.75,
    "If problematic claims remain",
    fontsize=10,
    va="center"
)

ax.add_patch(
    FancyArrowPatch(
        (8.0, 3.35),
        (12.0, 3.35),
        arrowstyle="->",
        mutation_scale=16,
        linewidth=1.5
    )
)

ax.text(
    10.0, 3.1,
    "Bounded correction loop (max 2 iterations)",
    ha="center",
    fontsize=10
)

ax.add_patch(
    FancyArrowPatch(
        (12.0, 3.35),
        (12.0, 4.25),
        arrowstyle="->",
        mutation_scale=16,
        linewidth=1.5
    )
)

# -----------------------------
# KEY PRINCIPLE
# -----------------------------
principle = (
    "KEY PRINCIPLE\n"
    "Unsupported ≠ Automatically Biased\n"
    "Correct only unsupported, bias-relevant assumptions"
)

box(
    4.5, 1.15, 7.0, 1.25,
    "DESIGN PRINCIPLE",
    [
        "Unsupported ≠ Automatically Biased",
        "Correct only unsupported, bias-relevant assumptions",
        "Preserve visually supported information"
    ],
    fontsize=13
)

# -----------------------------
# Technical components
# -----------------------------
ax.text(
    8, 0.45,
    "Inference-time only • No additional training • No model retraining",
    ha="center",
    fontsize=11,
    fontweight="bold"
)

plt.tight_layout()

# -----------------------------
# Save
# -----------------------------
output = "system_architecture.png"
plt.savefig(
    output,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Saved: {output}")