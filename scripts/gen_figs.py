"""
Paper Figure Generator v2
Generates publication-ready figures and tables from training data.
Designed for the Minecraft RL agent paper.

Figures generated:
  Fig 1: Learning curve (reward over episodes, with rolling mean + std band)
  Fig 2: Reward component breakdown (pie + time series)
  Fig 3: Tech tree progression (steps to reach each milestone)
  Fig 4: Goal completion timeline
  Fig 5: Human-like behavior radar chart
  Fig 6: Farming & construction progress over training
  Fig 7: Combat effectiveness (mobs killed per episode)
  Fig 8: Episode survival duration distribution
  Fig 9: Exploration coverage (chunks visited over time)
  Fig 10: Baseline vs agent comparison bar chart

Tables:
  Table 1: Summary statistics (LaTeX)
  Table 2: Advancement timeline (LaTeX)
  Table 3: LLM cost analysis (LaTeX)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import numpy as np
import json
from pathlib import Path
from scipy.ndimage import gaussian_filter1d


# ── STYLE ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

COLORS = {
    "primary":   "#2E86AB",
    "secondary": "#A23B72",
    "farming":   "#F18F01",
    "building":  "#C73E1D",
    "combat":    "#3B1F2B",
    "survival":  "#44BBA4",
    "goals":     "#E94F37",
    "baseline":  "#AAAAAA",
}

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── DATA LOADING ───────────────────────────────────────────────────────────────
def load_run(log_dir: str) -> dict:
    """Load all data files from a training run."""
    p = Path(log_dir)
    data = {}

    steps_path = p / "steps.csv"
    if steps_path.exists():
        data["steps"] = pd.read_csv(steps_path)
        print(f"Loaded {len(data['steps'])} steps")

    episodes_path = p / "episodes.csv"
    if episodes_path.exists():
        data["episodes"] = pd.read_csv(episodes_path)
        print(f"Loaded {len(data['episodes'])} episodes")

    events_path = p / "events.csv"
    if events_path.exists():
        data["events"] = pd.read_csv(events_path)
        print(f"Loaded {len(data['events'])} events")

    metrics_path = p / "paper_metrics_final.json"
    if not metrics_path.exists():
        metrics_path = p / "paper_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            data["metrics"] = json.load(f)
        print(f"Loaded paper metrics")

    llm_path = p / "llm_calls.csv"
    if llm_path.exists():
        data["llm"] = pd.read_csv(llm_path)
        print(f"Loaded {len(data['llm'])} LLM calls")

    return data


# ── FIGURE 1: LEARNING CURVE ───────────────────────────────────────────────────
def fig1_learning_curve(ep: pd.DataFrame, out: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    window = 20
    rewards = ep["total_reward"].values
    smoothed = gaussian_filter1d(rewards, sigma=5)
    rolling_std = pd.Series(rewards).rolling(window, min_periods=1).std().fillna(0).values

    # Main reward curve
    ax1.fill_between(ep["episode"], smoothed - rolling_std, smoothed + rolling_std,
                     alpha=0.2, color=COLORS["primary"])
    ax1.plot(ep["episode"], rewards, color=COLORS["primary"],
             alpha=0.3, linewidth=0.5, label="Raw reward")
    ax1.plot(ep["episode"], smoothed, color=COLORS["primary"],
             linewidth=2.5, label=f"Smoothed (σ=5)")
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.set_ylabel("Total Episode Reward")
    ax1.set_title("Learning Curve: Episode Reward over Training")
    ax1.legend(loc="upper left")

    # Episode duration subplot
    if "duration_seconds" in ep.columns:
        ax2.bar(ep["episode"], ep["duration_seconds"] / 60,
                color=COLORS["secondary"], alpha=0.6, width=0.8)
        ax2.set_ylabel("Duration (min)")
        ax2.set_xlabel("Episode")

    fig.tight_layout()
    fig.savefig(out / "fig1_learning_curve.pdf")
    fig.savefig(out / "fig1_learning_curve.png")
    plt.close(fig)
    print("Fig 1: Learning curve done")


# ── FIGURE 2: REWARD BREAKDOWN ─────────────────────────────────────────────────
def fig2_reward_breakdown(st: pd.DataFrame, out: Path):
    # Find reward component columns
    r_cols = [c for c in st.columns if c.startswith("r_")]

    cats = {
        "Survival":  [c for c in r_cols if "survival" in c],
        "Farming":   [c for c in r_cols if "farm" in c or "harvest" in c],
        "Building":  [c for c in r_cols if "build" in c],
        "Combat":    [c for c in r_cols if "combat" in c or "kill" in c or "hunt" in c],
        "Goals":     [c for c in r_cols if "goal" in c],
        "Exploration": [c for c in r_cols if "explor" in c or "day_" in c],
        "Penalties": [c for c in r_cols if "penalty" in c],
    }

    totals = {}
    for cat, cols in cats.items():
        existing = [c for c in cols if c in st.columns]
        if existing:
            val = st[existing].fillna(0).sum().sum()
            if abs(val) > 0.01:
                totals[cat] = val

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Pie chart
    pos_cats = {k: v for k, v in totals.items() if v > 0}
    colors = [COLORS.get(k.lower().split()[0], "#888") for k in pos_cats]
    wedges, texts, autotexts = ax1.pie(
        list(pos_cats.values()), labels=list(pos_cats.keys()),
        colors=colors, autopct='%1.1f%%', startangle=90,
        pctdistance=0.85
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax1.set_title("Reward Distribution by Category")

    # Time series
    bin_size = max(len(st) // 50, 100)
    st_copy = st.copy()
    st_copy["bin"] = (st_copy["step"] // bin_size) * bin_size

    for cat, cols in cats.items():
        existing = [c for c in cols if c in st_copy.columns]
        if existing:
            binned = st_copy.groupby("bin")[existing].sum().sum(axis=1)
            color = COLORS.get(cat.lower().split()[0], "#888")
            ax2.plot(binned.index / 1000, binned.values,
                     label=cat, linewidth=1.5, color=color)

    ax2.set_xlabel("Training Steps (thousands)")
    ax2.set_ylabel(f"Cumulative Reward (per {bin_size} steps)")
    ax2.set_title("Reward Components over Training")
    ax2.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out / "fig2_reward_breakdown.pdf")
    fig.savefig(out / "fig2_reward_breakdown.png")
    plt.close(fig)
    print("Fig 2: Reward breakdown done")


# ── FIGURE 3: TECH TREE PROGRESSION ───────────────────────────────────────────
def fig3_tech_tree(metrics: dict, out: Path):
    adv_log = metrics.get("advancement_log", {})
    if not adv_log:
        print("Fig 3: No advancement data — skipping")
        return

    # Tech tree milestones with expected step ranges
    milestones = {
        "Wood collected": metrics.get("first_harvest_step", None),
        "First harvest": metrics.get("first_harvest_step", None),
        "Iron tools": metrics.get("first_iron_step", None),
        "Diamonds": metrics.get("first_diamond_step", None),
        "First sleep": metrics.get("first_sleep_step", None),
    }

    # Add advancements
    for adv, step in adv_log.items():
        milestones[adv] = step

    # Filter out None values
    milestones = {k: v for k, v in milestones.items() if v is not None}

    if not milestones:
        print("Fig 3: No milestone data — skipping")
        return

    sorted_milestones = sorted(milestones.items(), key=lambda x: x[1])
    names = [m[0] for m in sorted_milestones]
    steps = [m[1] for m in sorted_milestones]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
    bars = ax.barh(names, steps, color=colors, edgecolor="white", height=0.6)

    for bar, step in zip(bars, steps):
        ax.text(bar.get_width() + max(steps) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"Step {step:,}", va='center', fontsize=9)

    ax.set_xlabel("Training Step (first occurrence)")
    ax.set_title("Tech Tree Milestone Progression\n(First step each milestone was achieved)")
    ax.set_xlim(0, max(steps) * 1.2)

    fig.tight_layout()
    fig.savefig(out / "fig3_tech_tree.pdf")
    fig.savefig(out / "fig3_tech_tree.png")
    plt.close(fig)
    print("Fig 3: Tech tree done")


# ── FIGURE 4: BEHAVIOR OVER TRAINING ──────────────────────────────────────────
def fig4_behavior_over_training(ep: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    metrics_to_plot = [
        ("crops_harvested", "Crops Harvested per Episode", COLORS["farming"]),
        ("blocks_placed", "Blocks Placed per Episode", COLORS["building"]),
        ("mobs_killed", "Mobs Killed per Episode", COLORS["combat"]),
        ("goals_completed", "Goals Completed per Episode", COLORS["goals"]),
    ]

    for ax, (col, title, color) in zip(axes, metrics_to_plot):
        if col not in ep.columns:
            ax.text(0.5, 0.5, f"No data: {col}",
                    ha='center', va='center', transform=ax.transAxes)
            continue

        values = ep[col].fillna(0).values
        smoothed = gaussian_filter1d(values.astype(float), sigma=3)

        ax.bar(ep["episode"], values, color=color, alpha=0.3, width=0.8)
        ax.plot(ep["episode"], smoothed, color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Count")

    fig.suptitle("Agent Behavior Metrics over Training", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out / "fig4_behavior_metrics.pdf")
    fig.savefig(out / "fig4_behavior_metrics.png")
    plt.close(fig)
    print("Fig 4: Behavior metrics done")


# ── FIGURE 5: HUMAN-LIKE BEHAVIOR RADAR ───────────────────────────────────────
def fig5_radar_chart(metrics: dict, out: Path):
    """Radar chart comparing agent capabilities — great visual for paper."""

    total_eps = max(metrics.get("total_episodes", 1), 1)

    categories = [
        "Survival\n(avg duration)",
        "Farming\n(crops/ep)",
        "Animal\nHusbandry",
        "Construction\n(blocks/ep)",
        "Combat\n(mobs/ep)",
        "Exploration\n(goals)",
    ]

    # Normalize each to 0-1 scale based on expected maximums
    values = [
        min(metrics.get("max_episode_survival_steps", 0) / 36000, 1.0),
        min(metrics.get("total_crops_harvested", 0) / max(total_eps * 5, 1), 1.0),
        min(metrics.get("total_animals_bred", 0) / max(total_eps * 2, 1), 1.0),
        min(metrics.get("total_blocks_placed", 0) / max(total_eps * 50, 1), 1.0),
        min(metrics.get("total_mobs_killed", 0) / max(total_eps * 3, 1), 1.0),
        min(metrics.get("goals_completed_total", 0) / max(total_eps * 5, 1), 1.0),
    ]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.fill(angles, values_plot, color=COLORS["primary"], alpha=0.25)
    ax.plot(angles, values_plot, color=COLORS["primary"], linewidth=2)
    ax.plot(angles, [1.0] * len(angles), color="gray",
            linewidth=0.5, linestyle="--", alpha=0.5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_title("Agent Capability Profile\n(Normalized performance across domains)",
                 pad=20, fontsize=13)

    fig.savefig(out / "fig5_radar_capability.pdf")
    fig.savefig(out / "fig5_radar_capability.png")
    plt.close(fig)
    print("Fig 5: Radar chart done")


# ── FIGURE 6: INVENTORY PROGRESSION ───────────────────────────────────────────
def fig6_inventory_progression(st: pd.DataFrame, out: Path):
    """Show when key items were acquired over training steps."""
    if "has_pickaxe" not in st.columns:
        print("Fig 6: No inventory data — skipping")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes = axes.flatten()

    items = [
        ("has_pickaxe", "Has Pickaxe", COLORS["primary"]),
        ("has_hoe", "Has Hoe (farming)", COLORS["farming"]),
        ("has_sword", "Has Sword (combat)", COLORS["combat"]),
        ("has_bed", "Has Bed (shelter)", COLORS["building"]),
    ]

    bin_size = max(len(st) // 100, 50)
    st_copy = st.copy()
    st_copy["bin"] = (st_copy["step"] // bin_size) * bin_size

    for ax, (col, label, color) in zip(axes, items):
        if col not in st_copy.columns:
            continue
        binned = st_copy.groupby("bin")[col].mean()
        ax.fill_between(binned.index / 1000, binned.values,
                        alpha=0.3, color=color)
        ax.plot(binned.index / 1000, binned.values,
                color=color, linewidth=2)
        ax.set_title(label)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.set_ylabel("% of steps with item")

    axes[-1].set_xlabel("Training Steps (thousands)")
    axes[-2].set_xlabel("Training Steps (thousands)")

    fig.suptitle("Key Item Acquisition over Training", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "fig6_inventory_progression.pdf")
    fig.savefig(out / "fig6_inventory_progression.png")
    plt.close(fig)
    print("Fig 6: Inventory progression done")


# ── FIGURE 7: SURVIVAL DURATION DISTRIBUTION ──────────────────────────────────
def fig7_survival_distribution(ep: pd.DataFrame, out: Path):
    if "duration_seconds" not in ep.columns:
        print("Fig 7: No duration data — skipping")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    durations_min = ep["duration_seconds"] / 60

    # Histogram
    ax1.hist(durations_min, bins=20, color=COLORS["survival"],
             edgecolor="white", alpha=0.8)
    ax1.axvline(durations_min.mean(), color="red", linewidth=2,
                linestyle="--", label=f"Mean: {durations_min.mean():.1f} min")
    ax1.axvline(durations_min.median(), color="orange", linewidth=2,
                linestyle="--", label=f"Median: {durations_min.median():.1f} min")
    ax1.set_xlabel("Episode Duration (minutes)")
    ax1.set_ylabel("Count")
    ax1.set_title("Survival Duration Distribution")
    ax1.legend()

    # Duration over training (learning to survive longer)
    ax2.scatter(ep["episode"], durations_min, color=COLORS["survival"],
                alpha=0.3, s=10)
    smoothed = gaussian_filter1d(durations_min.values.astype(float), sigma=5)
    ax2.plot(ep["episode"], smoothed, color=COLORS["survival"],
             linewidth=2.5, label="Trend")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Duration (minutes)")
    ax2.set_title("Survival Duration over Training")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out / "fig7_survival_distribution.pdf")
    fig.savefig(out / "fig7_survival_distribution.png")
    plt.close(fig)
    print("Fig 7: Survival distribution done")


# ── TABLE 1: SUMMARY STATISTICS (LATEX) ───────────────────────────────────────
def table1_summary(metrics: dict, out: Path):
    rows = [
        ("\\textbf{Training}", ""),
        ("Total Training Steps", f"{metrics.get('total_steps', 0):,}"),
        ("Total Episodes", f"{metrics.get('total_episodes', 0):,}"),
        ("Training Time (hours)", f"{metrics.get('total_training_time_hours', 0):.1f}"),

        ("\\textbf{Survival}", ""),
        ("Total Agent Deaths", f"{metrics.get('total_deaths', 0):,}"),
        ("Max Episode Duration (steps)", f"{metrics.get('max_episode_survival_steps', 0):,}"),

        ("\\textbf{Tech Tree}", ""),
        ("Episodes Reached Iron Tools", f"{metrics.get('episodes_reached_iron_tools', 0):,}"),
        ("Episodes Reached Diamonds", f"{metrics.get('episodes_reached_diamonds', 0):,}"),
        ("First Diamond (step)", f"{metrics.get('first_diamond_step', 'N/A')}"),

        ("\\textbf{Farming}", ""),
        ("Total Crops Harvested", f"{metrics.get('total_crops_harvested', 0):,}"),
        ("Episodes with Farming", f"{metrics.get('episodes_with_farm', 0):,}"),
        ("First Harvest (step)", f"{metrics.get('first_harvest_step', 'N/A')}"),

        ("\\textbf{Animal Husbandry}", ""),
        ("Total Animals Bred", f"{metrics.get('total_animals_bred', 0):,}"),
        ("Episodes with Breeding", f"{metrics.get('episodes_with_breeding', 0):,}"),

        ("\\textbf{Construction}", ""),
        ("Total Blocks Placed", f"{metrics.get('total_blocks_placed', 0):,}"),
        ("Episodes with Bed Placed", f"{metrics.get('episodes_bed_placed', 0):,}"),
        ("First Sleep (step)", f"{metrics.get('first_sleep_step', 'N/A')}"),

        ("\\textbf{Combat}", ""),
        ("Total Mobs Killed", f"{metrics.get('total_mobs_killed', 0):,}"),
        ("Episodes with Combat", f"{metrics.get('episodes_with_combat', 0):,}"),

        ("\\textbf{Goal System}", ""),
        ("Total Goals Completed", f"{metrics.get('goals_completed_total', 0):,}"),
        ("Max Goals in One Episode", f"{metrics.get('max_goals_in_episode', 0)}"),

        ("\\textbf{LLM Planner}", ""),
        ("Total LLM API Calls", f"{metrics.get('llm_total_calls', 0):,}"),
        ("Estimated LLM Cost (USD)", f"\\${metrics.get('llm_estimated_cost_usd', 0):.3f}"),
    ]

    latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Training Summary Statistics}\n"
        "\\label{tab:summary}\n"
        "\\begin{tabular}{lr}\n"
        "\\hline\n"
        "\\textbf{Metric} & \\textbf{Value} \\\\\n"
        "\\hline\n"
    )
    for metric, value in rows:
        if value == "":
            latex += f"\\hline\n{metric} & \\\\\n"
        else:
            latex += f"{metric} & {value} \\\\\n"

    latex += (
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )

    with open(out / "table1_summary.tex", "w") as f:
        f.write(latex)

    # Also save as JSON for easy reading
    with open(out / "table1_summary.json", "w") as f:
        json.dump({m: v for m, v in rows if v != ""}, f, indent=2)

    print("Table 1: Summary statistics done")


# ── TABLE 2: ADVANCEMENT TIMELINE (LATEX) ─────────────────────────────────────
def table2_advancements(metrics: dict, out: Path):
    adv_log = metrics.get("advancement_log", {})
    if not adv_log:
        print("Table 2: No advancement data — skipping")
        return

    total_steps = metrics.get("total_steps", 1)
    sorted_adv = sorted(adv_log.items(), key=lambda x: x[1])

    latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Minecraft Advancement Timeline}\n"
        "\\label{tab:advancements}\n"
        "\\begin{tabular}{lrr}\n"
        "\\hline\n"
        "\\textbf{Advancement} & \\textbf{Step} & \\textbf{\\% of Training} \\\\\n"
        "\\hline\n"
    )

    for adv, step in sorted_adv:
        pct = step / total_steps * 100
        latex += f"{adv} & {step:,} & {pct:.1f}\\% \\\\\n"

    latex += (
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )

    with open(out / "table2_advancements.tex", "w") as f:
        f.write(latex)

    print("Table 2: Advancement timeline done")


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main(log_dir: str, baseline_dir: str = None):
    print(f"\nGenerating paper figures from: {log_dir}")
    print("=" * 60)

    data = load_run(log_dir)
    out = OUTPUT_DIR

    ep = data.get("episodes", pd.DataFrame())
    st = data.get("steps", pd.DataFrame())
    metrics = data.get("metrics", {})

    if ep.empty:
        print("WARNING: No episode data found")
    if st.empty:
        print("WARNING: No step data found")

    # Generate all figures
    if not ep.empty:
        fig1_learning_curve(ep, out)
        fig4_behavior_over_training(ep, out)
        fig7_survival_distribution(ep, out)

    if not st.empty:
        fig2_reward_breakdown(st, out)
        fig6_inventory_progression(st, out)

    if metrics:
        fig3_tech_tree(metrics, out)
        fig5_radar_chart(metrics, out)
        table1_summary(metrics, out)
        table2_advancements(metrics, out)

    print(f"\nAll outputs saved to: {out}/")
    print("\nFiles ready for LaTeX paper:")
    for f in sorted(out.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    import sys
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/run_latest"
    main(log_dir)