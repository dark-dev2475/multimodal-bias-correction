"""Export an evidence-graph research figure from a V2 run or scripted demo."""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


def plot(result, output):
    graph = result.get("graph")
    if graph is None or "repair" not in result:
        raise ValueError("A completed graph/repair artifact is required")
    selected = set(result["repair"]["selected_ids"])
    nodes = {c["id"]: c for c in graph["claims"]}
    levels = {}
    def level(cid):
        if cid not in levels:
            levels[cid] = 1 + max((level(p) for p in nodes[cid]["depends_on"]), default=0)
        return levels[cid]
    for cid in nodes:
        level(cid)
    grouped = {0: [o["id"] for o in graph["observations"]]}
    for cid, rank in levels.items():
        grouped.setdefault(rank, []).append(cid)
    max_level = max(grouped)
    positions = {}
    for rank, ids in grouped.items():
        for i, oid in enumerate(ids):
            positions[oid] = (rank * 3.3, (len(ids) - 1) / 2 - i)
    colors = {"observation": "#dbeafe", "retained": "#d1fae5", "removed": "#fee2e2"}
    fig = plt.figure(figsize=(17, 9), facecolor="white")
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.1], hspace=0.4, wspace=0.3)
    ax = fig.add_subplot(grid[:, :2])
    all_nodes = [(o["id"], o["text"], "observation") for o in graph["observations"]]
    all_nodes += [(c["id"], c["text"], "retained" if c["id"] in selected else "removed")
                  for c in graph["claims"]]
    for claim in graph["claims"]:
        for parent in claim["evidence_ids"] + claim["depends_on"]:
            start, end = positions[parent], positions[claim["id"]]
            color = "#047857" if claim["id"] in selected else "#b91c1c"
            ax.add_patch(FancyArrowPatch((start[0] + 1.1, start[1]), (end[0] - 1.1, end[1]),
                arrowstyle="-|>", mutation_scale=12, color=color, alpha=0.6,
                linewidth=1.5, connectionstyle="arc3,rad=0.04", zorder=1))
    for oid, text, kind in all_nodes:
        x, y = positions[oid]
        ax.add_patch(FancyBboxPatch((x - 1.1, y - 0.35), 2.2, 0.7,
                                   boxstyle="round,pad=0.08", facecolor=colors[kind],
                                   edgecolor="#94a3b8", linewidth=0.8, zorder=2))
        label = oid + "  " + "\n".join(textwrap.wrap(text, 27)[:4])
        ax.text(x, y, label, ha="center", va="center", fontsize=9, color="#0f172a",
                zorder=3, parse_math=False)
    max_rows = max(map(len, grouped.values()))
    ax.set_xlim(-1.5, max_level * 3.3 + 1.5)
    ax.set_ylim(-max_rows / 2 - 0.8, max_rows / 2 + 0.8)
    ax.set_title("Evidence paths and local graph repair", loc="left", fontsize=15, weight="bold", pad=20)
    ax.axis("off")
    ax.legend(handles=[Patch(facecolor=color, edgecolor="#94a3b8", label=name.capitalize())
                       for name, color in colors.items()], loc="lower center", ncol=3, frameon=False)
    responses = fig.add_subplot(grid[0, 2])
    responses.axis("off")
    responses.set_title("Before and after", loc="left", fontsize=15, weight="bold", pad=20)
    initial = "\n".join(textwrap.wrap(result.get("initial_response", ""), 50))
    final = "\n".join(textwrap.wrap(result["final_response"], 50))
    responses.text(0, 0.95, "INITIAL DRAFT", color="#64748b", fontsize=10, weight="bold")
    responses.text(0, 0.87, initial, va="top", fontsize=11, linespacing=1.4, parse_math=False)
    responses.text(0, 0.36, "RELEASED RESPONSE", color="#047857", fontsize=10, weight="bold")
    responses.text(0, 0.28, final, va="top", fontsize=11, linespacing=1.4, parse_math=False)
    contrasts = fig.add_subplot(grid[1, 2])
    rows = [s for probe in result.get("interventions", []) for s in probe.get("scores", [])]
    if rows:
        for offset, field, label, color in [(-0.25, "cue_drop", "Cue removal", "#ef4444"),
                                          (0, "control_drop", "Control removal", "#94a3b8"),
                                          (0.25, "dependence_score", "Contrast", "#0f766e")]:
            contrasts.bar([i + offset for i in range(len(rows))], [r[field] or 0 for r in rows],
                          width=0.23, label=label, color=color)
        contrasts.set_xticks(range(len(rows)), [r["claim_id"] for r in rows])
        contrasts.set_ylim(0, 1.18)
        contrasts.set_ylabel("Ordinal support decrease (not probability)")
        contrasts.legend(frameon=False, fontsize=9)
    else:
        contrasts.text(0.5, 0.5, "No measured intervention contrasts", ha="center", va="center")
        contrasts.set_axis_off()
    contrasts.set_title("Controlled evidence sensitivity", loc="left", fontsize=14, weight="bold", pad=18)
    synthetic = result.get("model", "").startswith("SCRIPTED")
    kind = "Scripted mechanism demonstration — not benchmark results" if synthetic else "Live run — model judgments, not independent ground truth"
    fig.suptitle("EIGR  |  Version 2", x=0.05, y=0.98, ha="left", fontsize=23, weight="bold", color="#0f172a")
    fig.text(0.05, 0.935, kind, fontsize=12, color="#475569")
    fig.text(0.05, 0.025, "Graph constraints are enforced locally. Their validity is conditional on model-extracted evidence and dependencies.",
             fontsize=10, color="#64748b")
    fig.subplots_adjust(top=0.84, bottom=0.1, left=0.05, right=0.97)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", default="cue_and_propagation")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if "cases" in payload:
        payload = next(c["result"] for c in payload["cases"]
                       if c["case_id"] == args.case_id and c["ablation"] == "full")
    plot(payload, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
