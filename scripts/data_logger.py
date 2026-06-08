"""
Enhanced Data Logger v2 — Paper-ready data collection
Improvements:
- Structured data collection for all paper sections
- Rich per-step metrics including goal progress, combat, exploration
- Episode summary with all key paper metrics
- Automatic paper-ready statistics computation
- Efficient buffered writes (avoid disk I/O bottleneck)
- LLM call tracking (cost estimation)
- Advancement/milestone event log
"""

import json
import time
import wandb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque


class AgentDataLogger:

    def __init__(self, config: dict, run_name: str = None):
        self.config = config
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = Path(f"logs/{self.run_name}")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # W&B initialization
        self.wandb_run = wandb.init(
            project=config["logging"]["wandb_project"],
            name=self.run_name,
            config=config,
            tags=["minecraft", "rl", "farming", "building", "v2"]
        )

        # ── BUFFERS (write to disk every N steps, not every step) ──────
        self.step_buffer = []
        self.episode_buffer = []
        self.event_buffer = []        # milestone events (advancements, deaths)
        self.llm_call_buffer = []     # LLM call tracking for cost analysis
        self.reward_component_buffer = []  # detailed reward breakdown

        self.BUFFER_FLUSH_SIZE = 500  # flush to CSV every 500 steps

        # ── COUNTERS ───────────────────────────────────────────────────
        self.global_step = 0
        self.current_episode = 0
        self.episode_start_time = time.time()
        self.training_start_time = time.time()

        # ── ROLLING STATISTICS (for W&B smooth curves) ─────────────────
        self.recent_rewards = deque(maxlen=100)
        self.recent_episode_rewards = deque(maxlen=20)
        self.recent_crops = deque(maxlen=20)
        self.recent_blocks = deque(maxlen=20)

        # ── LLM COST TRACKING ──────────────────────────────────────────
        self.llm_calls = 0
        self.llm_tokens_in = 0
        self.llm_tokens_out = 0
        # gpt-4o-mini pricing (per 1M tokens)
        self.PRICE_IN = 0.15 / 1_000_000
        self.PRICE_OUT = 0.60 / 1_000_000

        # ── PAPER METRICS ACCUMULATOR ──────────────────────────────────
        # These are the key numbers that go into the paper
        self.paper_metrics = {
            # Performance
            "total_steps": 0,
            "total_episodes": 0,
            "total_training_time_hours": 0,

            # Survival
            "total_deaths": 0,
            "avg_episode_survival_steps": 0,
            "max_episode_survival_steps": 0,

            # Tech tree
            "episodes_reached_stone_tools": 0,
            "episodes_reached_iron_tools": 0,
            "episodes_reached_diamonds": 0,
            "first_diamond_step": None,
            "first_iron_step": None,

            # Farming
            "total_crops_harvested": 0,
            "episodes_with_farm": 0,
            "first_harvest_step": None,

            # Animals
            "total_animals_bred": 0,
            "episodes_with_breeding": 0,

            # Construction
            "total_blocks_placed": 0,
            "episodes_with_house": 0,
            "episodes_bed_placed": 0,
            "episodes_slept": 0,
            "first_sleep_step": None,

            # Combat
            "total_mobs_killed": 0,
            "episodes_with_combat": 0,

            # Goals
            "goals_completed_total": 0,
            "max_goals_in_episode": 0,

            # LLM
            "llm_total_calls": 0,
            "llm_estimated_cost_usd": 0,
        }

        # ── ADVANCEMENT TRACKING ───────────────────────────────────────
        # Key Minecraft advancements for paper (ordered by difficulty)
        self.advancement_first_step = {}
        self.TRACKED_ADVANCEMENTS = [
            "Stone Age",
            "Getting an Upgrade",
            "Acquire Hardware",
            "Isn't It Iron Pick",
            "Hot Stuff",
            "Diamonds!",
            "We Need to Go Deeper",
        ]

        print(f"[LOGGER] Run: {self.run_name}")
        print(f"[LOGGER] Logs: {self.log_dir}")

    # ── STEP LOGGING ───────────────────────────────────────────────────────────
    def log_step(self, obs, action, reward: float,
                 reward_components: dict, info: dict, done: bool):
        """Record data for a single step — optimized for speed."""
        self.global_step += 1
        self.recent_rewards.append(reward)

        # Extract key info fields
        health = info.get("health", 20)
        food = info.get("food_level", 20)
        pos = info.get("player_pos", {})
        inv = info.get("inventory", {})
        inv_summary = {
            s["type"]: s["quantity"]
            for s in inv.values()
            if s["type"] != "none"
        }
        has_pickaxe = int(any("pickaxe" in k for k in inv_summary))
        has_hoe = int(any("hoe" in k for k in inv_summary))
        has_sword = int(any("sword" in k for k in inv_summary))
        has_bed = int(any("bed" in k for k in inv_summary))
        has_armor = int(any(k in inv_summary for k in
                           ["leather_chestplate", "iron_chestplate", "chainmail_chestplate"]))

        # Step record — lean schema, only what matters for the paper
        step_record = {
            "step": self.global_step,
            "episode": self.current_episode,
            "reward": round(reward, 4),
            "health": health,
            "food": food,
            "x": round(pos.get("x", 0), 1),
            "y": round(pos.get("y", 0), 1),
            "z": round(pos.get("z", 0), 1),
            "inv_count": len(inv_summary),
            "has_pickaxe": has_pickaxe,
            "has_hoe": has_hoe,
            "has_sword": has_sword,
            "has_bed": has_bed,
            "has_armor": has_armor,
            "current_task": info.get("current_task", "none"),
            "current_goal": info.get("current_goal", "none"),
        }

        # Add reward components (flattened)
        for k, v in reward_components.items():
            if v != 0:  # only store non-zero components (saves disk space)
                step_record[f"r_{k}"] = round(v, 4)

        self.step_buffer.append(step_record)

        # ── W&B LOGGING (every log_interval steps) ────────────────────
        log_interval = self.config["logging"]["log_interval"]
        if self.global_step % log_interval == 0:
            rolling_mean = np.mean(self.recent_rewards)
            rolling_std = np.std(self.recent_rewards)

            wandb.log({
                # Core metrics
                "train/reward_mean": rolling_mean,
                "train/reward_std": rolling_std,
                "train/reward_raw": reward,
                "train/health": health,
                "train/food_level": food,

                # Inventory state (key for paper)
                "inventory/total_items": len(inv_summary),
                "inventory/has_pickaxe": has_pickaxe,
                "inventory/has_hoe": has_hoe,
                "inventory/has_sword": has_sword,
                "inventory/has_bed": has_bed,
                "inventory/has_armor": has_armor,

                # Reward components grouped
                "reward_components/survival": sum(
                    v for k, v in reward_components.items() if "survival" in k
                ),
                "reward_components/farming": sum(
                    v for k, v in reward_components.items() if "farm" in k
                ),
                "reward_components/building": sum(
                    v for k, v in reward_components.items() if "build" in k
                ),
                "reward_components/combat": sum(
                    v for k, v in reward_components.items() if "combat" in k or "kill" in k
                ),
                "reward_components/goals": sum(
                    v for k, v in reward_components.items() if "goal" in k
                ),
                "reward_components/penalties": sum(
                    v for k, v in reward_components.items() if "penalty" in k
                ),

                # Training efficiency
                "train/steps_per_second": self.global_step / max(
                    time.time() - self.training_start_time, 1
                ),
            }, step=self.global_step)

        # ── BUFFER FLUSH ──────────────────────────────────────────────
        if len(self.step_buffer) >= self.BUFFER_FLUSH_SIZE:
            self._flush_step_buffer()

        self.paper_metrics["total_steps"] = self.global_step

    # ── EPISODE LOGGING ────────────────────────────────────────────────────────
    def log_episode_end(self, episode_stats: dict):
        """Record end-of-episode statistics."""
        duration = time.time() - self.episode_start_time
        total_reward = episode_stats.get("total_reward", 0)

        self.recent_episode_rewards.append(total_reward)
        self.recent_crops.append(episode_stats.get("crops_harvested", 0))
        self.recent_blocks.append(episode_stats.get("blocks_placed", 0))

        episode_record = {
            "episode": self.current_episode,
            "end_step": self.global_step,
            "duration_seconds": round(duration, 1),
            "total_reward": round(total_reward, 2),
            "crops_harvested": episode_stats.get("crops_harvested", 0),
            "animals_bred": episode_stats.get("animals_bred", 0),
            "blocks_placed": round(episode_stats.get("blocks_placed", 0)),
            "bed_placed": int(episode_stats.get("bed_placed", False)),
            "tools_crafted": episode_stats.get("tools_crafted", 0),
            "mobs_killed": episode_stats.get("mobs_killed", 0),
            "goals_completed": episode_stats.get("goals_completed", 0),
            "cause_of_end": episode_stats.get("cause_of_end", "timeout"),
            "final_health": episode_stats.get("final_health", 0),
            "final_food": episode_stats.get("final_food", 20),
        }

        self.episode_buffer.append(episode_record)

        # Update paper metrics
        self._update_paper_metrics(episode_stats, duration)

        # W&B episode logging
        wandb.log({
            "episode/total_reward": total_reward,
            "episode/reward_mean_20": np.mean(self.recent_episode_rewards),
            "episode/duration_seconds": duration,
            "episode/crops_harvested": episode_stats.get("crops_harvested", 0),
            "episode/blocks_placed": round(episode_stats.get("blocks_placed", 0)),
            "episode/bed_placed": int(episode_stats.get("bed_placed", False)),
            "episode/mobs_killed": episode_stats.get("mobs_killed", 0),
            "episode/goals_completed": episode_stats.get("goals_completed", 0),
            "episode/tools_crafted": episode_stats.get("tools_crafted", 0),
            "episode/mean_crops_20": np.mean(self.recent_crops),
            "episode/mean_blocks_20": np.mean(self.recent_blocks),
        }, step=self.global_step)

        self.current_episode += 1
        self.episode_start_time = time.time()

        # Flush episode buffer
        if len(self.episode_buffer) >= 10:
            self._flush_episode_buffer()

    # ── MILESTONE EVENT LOGGING ────────────────────────────────────────────────
    def log_advancement(self, advancement_name: str):
        """Log Minecraft advancement achievements — key data for paper."""
        if advancement_name not in self.advancement_first_step:
            self.advancement_first_step[advancement_name] = self.global_step

            event = {
                "step": self.global_step,
                "episode": self.current_episode,
                "type": "advancement",
                "name": advancement_name,
                "timestamp": time.time() - self.training_start_time,
            }
            self.event_buffer.append(event)

            # Update paper metrics
            if advancement_name == "Diamonds!":
                self.paper_metrics["first_diamond_step"] = self.global_step
                self.paper_metrics["episodes_reached_diamonds"] += 1
            elif advancement_name in ["Acquire Hardware", "Isn't It Iron Pick"]:
                if self.paper_metrics["first_iron_step"] is None:
                    self.paper_metrics["first_iron_step"] = self.global_step
                self.paper_metrics["episodes_reached_iron_tools"] += 1

            wandb.log({
                f"advancements/{advancement_name.replace(' ', '_')}": self.global_step,
                "advancements/total_count": len(self.advancement_first_step),
            }, step=self.global_step)

            print(f"[ADVANCEMENT] {advancement_name} at step {self.global_step}")

    def log_death(self, cause: str = "unknown"):
        """Log agent death events."""
        event = {
            "step": self.global_step,
            "episode": self.current_episode,
            "type": "death",
            "cause": cause,
            "timestamp": time.time() - self.training_start_time,
        }
        self.event_buffer.append(event)
        self.paper_metrics["total_deaths"] += 1

        wandb.log({
            "events/death": 1,
            "events/total_deaths": self.paper_metrics["total_deaths"],
        }, step=self.global_step)

    def log_llm_call(self, model: str, tokens_in: int, tokens_out: int,
                     task: str, success: bool):
        """Track LLM API usage for cost analysis in paper."""
        self.llm_calls += 1
        self.llm_tokens_in += tokens_in
        self.llm_tokens_out += tokens_out

        cost = tokens_in * self.PRICE_IN + tokens_out * self.PRICE_OUT
        self.paper_metrics["llm_total_calls"] = self.llm_calls
        self.paper_metrics["llm_estimated_cost_usd"] += cost

        self.llm_call_buffer.append({
            "step": self.global_step,
            "episode": self.current_episode,
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": round(cost, 6),
            "task": task,
            "success": int(success),
        })

        if self.llm_calls % 50 == 0:
            wandb.log({
                "llm/total_calls": self.llm_calls,
                "llm/estimated_cost_usd": self.paper_metrics["llm_estimated_cost_usd"],
                "llm/tokens_per_step": (self.llm_tokens_in + self.llm_tokens_out) / max(self.global_step, 1),
            }, step=self.global_step)

    def log_goal_progress(self, goals_completed: int, current_goal: str,
                          goal_idx: int, mobs_killed: int, chunks_visited: int,
                          max_blocks: int):
        """Log goal system progress — key metric for paper."""
        wandb.log({
            "goals/completed_count": goals_completed,
            "goals/current_goal_idx": goal_idx,
            "goals/mobs_killed_total": mobs_killed,
            "goals/chunks_explored": chunks_visited,
            "goals/max_blocks_placed": max_blocks,
        }, step=self.global_step)

    # ── PAPER METRICS UPDATE ───────────────────────────────────────────────────
    def _update_paper_metrics(self, episode_stats: dict, duration: float):
        """Update cumulative paper metrics after each episode."""
        m = self.paper_metrics

        m["total_episodes"] = self.current_episode + 1
        m["total_training_time_hours"] = (
            time.time() - self.training_start_time
        ) / 3600

        # Survival
        survival_steps = self.global_step  # approximate
        m["max_episode_survival_steps"] = max(
            m["max_episode_survival_steps"], int(duration * 20)
        )

        # Farming
        crops = episode_stats.get("crops_harvested", 0)
        m["total_crops_harvested"] += crops
        if crops > 0:
            m["episodes_with_farm"] += 1
            if m["first_harvest_step"] is None:
                m["first_harvest_step"] = self.global_step

        # Animals
        bred = episode_stats.get("animals_bred", 0)
        m["total_animals_bred"] += bred
        if bred > 0:
            m["episodes_with_breeding"] += 1

        # Construction
        blocks = episode_stats.get("blocks_placed", 0)
        m["total_blocks_placed"] += blocks
        if blocks >= 25:
            m["episodes_with_house"] += 1
        if episode_stats.get("bed_placed", False):
            m["episodes_bed_placed"] += 1

        # Combat
        mobs = episode_stats.get("mobs_killed", 0)
        m["total_mobs_killed"] += mobs
        if mobs > 0:
            m["episodes_with_combat"] += 1

        # Goals
        goals = episode_stats.get("goals_completed", 0)
        m["goals_completed_total"] += goals
        m["max_goals_in_episode"] = max(m["max_goals_in_episode"], goals)

    # ── BUFFER FLUSHES ─────────────────────────────────────────────────────────
    def _flush_step_buffer(self):
        """Write step buffer to CSV."""
        if not self.step_buffer:
            return
        path = self.log_dir / "steps.csv"
        df = pd.DataFrame(self.step_buffer)
        df.to_csv(path, mode='a', header=not path.exists(), index=False)
        self.step_buffer = []

    def _flush_episode_buffer(self):
        """Write episode buffer to CSV."""
        if not self.episode_buffer:
            return
        path = self.log_dir / "episodes.csv"
        df = pd.DataFrame(self.episode_buffer)
        df.to_csv(path, mode='a', header=not path.exists(), index=False)
        self.episode_buffer = []

    def _flush_event_buffer(self):
        """Write event buffer to CSV."""
        if not self.event_buffer:
            return
        path = self.log_dir / "events.csv"
        df = pd.DataFrame(self.event_buffer)
        df.to_csv(path, mode='a', header=not path.exists(), index=False)
        self.event_buffer = []

    def _flush_llm_buffer(self):
        """Write LLM call buffer to CSV."""
        if not self.llm_call_buffer:
            return
        path = self.log_dir / "llm_calls.csv"
        df = pd.DataFrame(self.llm_call_buffer)
        df.to_csv(path, mode='a', header=not path.exists(), index=False)
        self.llm_call_buffer = []

    # ── CHECKPOINT ─────────────────────────────────────────────────────────────
    def save_checkpoint(self, step: int):
        """Save all buffers and paper metrics summary."""
        # Flush all buffers
        self._flush_step_buffer()
        self._flush_episode_buffer()
        self._flush_event_buffer()
        self._flush_llm_buffer()

        # Update and save paper metrics
        self.paper_metrics["total_training_time_hours"] = (
            time.time() - self.training_start_time
        ) / 3600
        self.paper_metrics["advancement_log"] = self.advancement_first_step

        with open(self.log_dir / "paper_metrics.json", "w") as f:
            json.dump(self.paper_metrics, f, indent=2)

        # Summary snapshot
        summary = {
            "step": step,
            "episode": self.current_episode,
            "training_hours": round(self.paper_metrics["total_training_time_hours"], 2),
            "total_deaths": self.paper_metrics["total_deaths"],
            "total_crops": self.paper_metrics["total_crops_harvested"],
            "total_mobs_killed": self.paper_metrics["total_mobs_killed"],
            "total_blocks": self.paper_metrics["total_blocks_placed"],
            "beds_placed": self.paper_metrics["episodes_bed_placed"],
            "llm_calls": self.llm_calls,
            "llm_cost_usd": round(self.paper_metrics["llm_estimated_cost_usd"], 4),
            "advancements_reached": list(self.advancement_first_step.keys()),
        }

        with open(self.log_dir / f"summary_step_{step}.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[CHECKPOINT] Step {step} | "
              f"Episodes: {self.current_episode} | "
              f"Crops: {self.paper_metrics['total_crops_harvested']} | "
              f"Mobs: {self.paper_metrics['total_mobs_killed']} | "
              f"Blocks: {self.paper_metrics['total_blocks_placed']} | "
              f"LLM cost: ${self.paper_metrics['llm_estimated_cost_usd']:.3f}")

    def finalize(self):
        """Final flush and close."""
        self._flush_step_buffer()
        self._flush_episode_buffer()
        self._flush_event_buffer()
        self._flush_llm_buffer()

        self.paper_metrics["total_training_time_hours"] = (
            time.time() - self.training_start_time
        ) / 3600
        self.paper_metrics["advancement_log"] = self.advancement_first_step

        with open(self.log_dir / "paper_metrics_final.json", "w") as f:
            json.dump(self.paper_metrics, f, indent=2)

        wandb.finish()
        print(f"\n[DONE] All data saved to: {self.log_dir}")
        print(f"       paper_metrics_final.json ready for analysis")