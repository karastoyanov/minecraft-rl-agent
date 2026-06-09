# Minecraft RL Agent

An autonomous AI agent capable of playing Minecraft, developing farming, constructing shelters, and learning from its own progress. Built for academic research and scientific publication.

The agent combines a pre-trained Vision-Language-Action model (VPT) with an LLM-based hierarchical planner and a custom shaped reward system to produce human-like survival and building behavior.

---

## Demo

The agent autonomously progresses through the full Minecraft tech tree — from punching trees to smelting iron, building a house, placing a bed, and sleeping through the night — all without human intervention.

| Achievement | Step |
|-------------|------|
| Stone Age (stone tools) | ~2,000 |
| Getting an Upgrade (stone pickaxe) | ~3,000 |
| Acquire Hardware (iron) | ~8,000 |
| Isn't It Iron Pick | ~12,000 |
| Diamonds! | ~40,000 |
| First bed placed and slept | ~122,000 |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│              HIGH-LEVEL PLANNER                 │
│         OpenAI GPT-4o-mini (API)                │
│  Translates current goal into specific actions  │
│  Called every 200 steps — context-aware         │
└─────────────────┬───────────────────────────────┘
                  │ task instruction
┌─────────────────▼───────────────────────────────┐
│             GOAL TRACKER                        │
│  34 ordered human-like goals                    │
│  punch_trees → craft_tools → farm → build house │
│  Loops back after secure_perimeter              │
└─────────────────┬───────────────────────────────┘
                  │ current goal context
┌─────────────────▼───────────────────────────────┐
│          LOW-LEVEL CONTROLLER                   │
│     VPT (Video PreTraining) — 248.5M params     │
│  Pre-trained on 70,000 hours of YouTube         │
│  Outputs keyboard/mouse actions from pixels     │
└─────────────────┬───────────────────────────────┘
                  │ actions
┌─────────────────▼───────────────────────────────┐
│         MINECRAFT ENVIRONMENT                   │
│   MineStudio v1.1.6 — headless Minecraft 1.16  │
│   Observation: 128x128 RGB + inventory + stats  │
│   Action space: movement, crafting, interaction │
└─────────────────┬───────────────────────────────┘
                  │ observations + events
┌─────────────────▼───────────────────────────────┐
│          REWARD FUNCTION                        │
│  Shaped rewards across 6 domains:               │
│  Survival · Farming · Animals · Construction    │
│  Combat · Exploration                           │
│  34 goal completion bonuses + tech tree milest. │
└─────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Simulation framework | [MineStudio v1.1.6](https://github.com/CraftJarvis/MineStudio) |
| Low-level controller | [VPT (OpenAI)](https://github.com/openai/Video-Pre-Training) — 248.5M parameters |
| LLM planner | [OpenAI GPT-4o-mini](https://platform.openai.com) |
| Experiment tracking | [Weights & Biases](https://wandb.ai) |
| Cloud GPU training | [RunPod](https://runpod.io) — RTX A4000 16GB |
| Local development | Debian 13 VM in VirtualBox — CPU inference |
| Environment | Python 3.10, PyTorch, Conda |
| Data storage | CSV + JSON logs, LMDB trajectories |

---

## Project Structure

```
minecraft-rl-agent/
├── configs/
│   ├── experiment.yaml          # Local training config
│   ├── experiment_runpod.yaml   # Cloud GPU training config
│   └── rewards.yaml             # Reward function weights
│
├── scripts/
│   ├── train_cloud.py           # Main GPU training pipeline
│   ├── train_local.py           # CPU training for local testing
│   ├── reward_function.py       # Base + HumanLike reward functions
│   ├── data_logger.py           # Paper-ready data collection
│   ├── llm_planner.py           # GPT-4o-mini task planner
│   └── gen_figs.py              # Paper figure generation
│
├── models/
│   └── checkpoints/             # Saved model weights (.pt files)
│
├── logs/                        # Training run data
│   └── run_YYYYMMDD_HHMMSS/
│       ├── steps.csv            # Per-step metrics
│       ├── episodes.csv         # Per-episode statistics
│       ├── events.csv           # Advancements and deaths
│       ├── paper_metrics.json   # Aggregated paper metrics
│       └── videos/              # Episode recordings (.mp4)
│
├── results/
│   └── figures/                 # Generated paper figures (PDF + PNG)
│
└── analysis/
    └── generate_paper_figures.py
```

---

## Goal System

The agent follows a 34-step goal sequence that mirrors natural human Minecraft progression:

```
Stage 1 — Survival
  punch_trees → craft_crafting_table → craft_wooden_tools → collect_food

Stage 2 — Mining
  collect_stone → craft_stone_tools → find_cave → collect_iron → build_furnace → craft_iron_tools

Stage 3 — Farming
  craft_hoe → find_water_source → till_soil → plant_seeds → harvest_crops

Stage 4 — Animal Husbandry
  find_animals → build_animal_pen → breed_animals → collect_animal_products

Stage 5 — Construction
  gather_building_materials → build_foundation → build_walls → build_roof
  → add_door_windows → craft_bed → place_bed_inside → sleep_through_night

Stage 6 — Combat & Defense
  craft_weapons → craft_armor → fight_mobs → build_torches → secure_perimeter
```

Each goal has explicit completion conditions. If a goal is not completed within 3,000 steps it times out and the agent moves to the next goal. After `secure_perimeter`, the sequence loops back to `punch_trees`.

---

## Reward Function

Rewards are shaped across six domains defined in `configs/rewards.yaml`:

| Domain | Examples |
|--------|---------|
| **Survival** | +0.01/step alive, −5 starvation, −20 death |
| **Farming** | +1 plant seed, +5 harvest wheat, +25 first harvest bonus |
| **Animals** | +10 breed animals, +20 second generation born |
| **Construction** | +0.05/block placed, +20 place bed, +50 sleep through night |
| **Combat** | +8 kill zombie/skeleton/spider, +3 hunt animal |
| **Exploration** | +0.3 new chunk discovered |

Goal completion bonuses range from +10 (basic goals) to +100 (sleep through night).

Rewards are clipped to `[-10, 100]` per step to prevent gradient explosions during PPO fine-tuning.

---

## Training

### Local Development (CPU)

Used for testing and debugging the pipeline logic. Slow (~2-5 steps/second) but sufficient for validating the full system.

```bash
# Setup
conda create -n minecraft-agent --channel=conda-forge python=3.10 openjdk=8 pip -y
conda activate minecraft-agent
python -m pip install minestudio wandb openai pandas matplotlib pyyaml torch

# Set API key
export OPENAI_API_KEY="sk-proj-..."

# Start virtual display
export DISPLAY=:1
Xvfb :1 -screen 0 1280x720x24 &

# Run local training
python scripts/train_local.py --config configs/experiment.yaml
```

### Cloud GPU Training (RunPod)

Used for full training runs. ~19-22 steps/second on RTX A4000.

```bash
# On RunPod instance (RTX A4000 recommended)
cd /workspace/minecraft-rl-agent
git pull origin main

export OPENAI_API_KEY="sk-proj-..."
export DISPLAY=:1
Xvfb :1 -screen 0 1280x720x24 &

tmux new-session -d -s training \
    "/miniconda3/bin/conda run -n minecraft-agent --no-capture-output \
    env OPENAI_API_KEY='$OPENAI_API_KEY' DISPLAY=':1' \
    python scripts/train_cloud.py --config configs/experiment_runpod.yaml \
    2>&1 | tee /workspace/logs/training.log"

tail -f /workspace/logs/training.log
```

Training automatically resumes from `models/checkpoints/latest.pt` if it exists.

### Training Configuration

Key parameters in `configs/experiment_runpod.yaml`:

```yaml
training:
  total_steps: 2_000_000
  batch_size: 256
  learning_rate: 0.00003
  gamma: 0.999
  checkpoint_interval: 25000

llm:
  model: "gpt-4o-mini"
  call_interval: 200       # LLM called every 200 steps

environment:
  obs_size: [128, 128]
  max_episode_steps: 18000  # ~15 minutes per episode
```

---

## Data Collection & Metrics

All training data is automatically recorded for paper analysis.

### Files Generated Per Run

| File | Contents |
|------|---------|
| `steps.csv` | Per-step: reward, health, food, position, inventory state, reward components |
| `episodes.csv` | Per-episode: total reward, crops harvested, blocks placed, mobs killed, goals completed, duration |
| `events.csv` | Timestamped Minecraft advancements and agent deaths |
| `paper_metrics.json` | Aggregated metrics: first diamond step, total crops, beds placed, LLM cost |
| `videos/episode_N.mp4` | Recorded first-person gameplay per episode |

### W&B Metrics (real-time)

Tracked live at [wandb.ai](https://wandb.ai):

```
train/reward_mean          — rolling mean reward (last 100 steps)
train/health               — agent health
train/food_level           — agent food level
inventory/has_pickaxe      — tool acquisition tracking
inventory/has_hoe          — farming readiness
inventory/has_sword        — combat readiness
inventory/has_bed          — shelter readiness
reward_components/*        — per-domain reward breakdown
episode/total_reward       — episode-level reward
episode/crops_harvested    — farming productivity
episode/blocks_placed      — construction progress
episode/mobs_killed        — combat effectiveness
episode/goals_completed    — goal system progress
goals/total_completions    — cumulative goal completions
advancements/*             — Minecraft advancement timestamps
llm/estimated_cost_usd     — API cost tracking
```

### Generating Paper Figures

After training, generate all publication-ready figures:

```bash
python scripts/gen_figs.py logs/run_YYYYMMDD_HHMMSS
```

Outputs to `results/figures/`:

| Figure | Content |
|--------|---------|
| `fig1_learning_curve.pdf` | Episode reward over training with smoothing |
| `fig2_reward_breakdown.pdf` | Reward distribution by domain (pie + time series) |
| `fig3_tech_tree.pdf` | Milestone progression timeline |
| `fig4_behavior_metrics.pdf` | Crops, blocks, mobs, goals per episode |
| `fig5_radar_capability.pdf` | Agent capability radar chart |
| `fig6_inventory_progression.pdf` | Key item acquisition rates |
| `fig7_survival_distribution.pdf` | Episode duration histogram |
| `table1_summary.tex` | Full summary statistics (LaTeX) |
| `table2_advancements.tex` | Advancement timeline (LaTeX) |

---

## Estimated Training Cost

| Run | GPU | Steps | Time | Cost |
|-----|-----|-------|------|------|
| Baseline (random policy) | RTX A4000 | 200k | ~2.5h | ~$2 |
| Main training run | RTX A4000 | 2M | ~25h | ~$19 |
| Ablation (no LLM planner) | RTX A4000 | 500k | ~6h | ~$5 |
| LLM API (GPT-4o-mini) | — | 2M steps | — | ~$0.50 |
| **Total** | | | | **~$26** |

---

## Requirements

```bash
python -m pip install \
    minestudio \
    torch torchvision \
    openai \
    wandb \
    pandas matplotlib seaborn scipy \
    pyyaml lmdb chromadb \
    gymnasium gym \
    transformers timm \
    ray psutil
```

System requirements:
- Java 21 (system) — for Minecraft server
- Java 8 (conda) — for MineStudio
- Xvfb — virtual display for headless rendering
- CUDA 12.x — for GPU training

---

## Environment Variables

```bash
export OPENAI_API_KEY="sk-proj-..."   # Required for LLM planner
export DISPLAY=":1"                    # Required for headless Minecraft
```

