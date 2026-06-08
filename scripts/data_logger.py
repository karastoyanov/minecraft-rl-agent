"""
Centralized logger for all agent data.
Uses correct MineStudio field names.
"""

import json
import time
import wandb
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict


class AgentDataLogger:

    def __init__(self, config: dict, run_name: str = None):
        self.config = config
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = Path(f"logs/{self.run_name}")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.wandb_run = wandb.init(
            project=config["logging"]["wandb_project"],
            name=self.run_name,
            config=config,
            tags=["minecraft", "rl", "farming", "building"]
        )

        self.step_data = []
        self.episode_data = []
        self.task_completion = []
        self.curriculum_progress = []

        self.current_episode = 0
        self.global_step = 0
        self.episode_start_time = time.time()

    def log_step(self, obs, action: dict, reward: float,
                 reward_components: dict, info: dict, done: bool):
        """Record data for a single step."""
        self.global_step += 1

        # Extract inventory summary
        inv = info.get("inventory", {})
        inv_summary = {
            s["type"]: s["quantity"]
            for s in inv.values()
            if s["type"] != "none"
        }

        # Player position
        pos = info.get("player_pos", {})

        step_record = {
            "global_step": self.global_step,
            "episode": self.current_episode,
            "timestamp": time.time(),
            "reward_total": reward,
            **{f"reward_{k}": v for k, v in reward_components.items()},
            "action_type": str(action.get("type", action)),
            "health": info.get("health", 0),
            "food_level": info.get("food_level", 0),  # correct field
            "position_x": pos.get("x", 0),
            "position_y": pos.get("y", 0),
            "position_z": pos.get("z", 0),
            "current_task": info.get("current_task", "none"),
            "curriculum_stage": info.get("curriculum_stage", 0),
            "inventory_count": len(inv_summary),
            "has_pickaxe": int(any("pickaxe" in k for k in inv_summary)),
            "has_hoe": int(any("hoe" in k for k in inv_summary)),
            "has_bed": int("bed" in inv_summary),
        }

        self.step_data.append(step_record)

        if self.global_step % self.config["logging"]["log_interval"] == 0:
            wandb.log({
                "step/reward_total": reward,
                "step/health": info.get("health", 0),
                "step/food_level": info.get("food_level", 0),
                "step/has_pickaxe": int(any("pickaxe" in k for k in inv_summary)),
                "step/has_hoe": int(any("hoe" in k for k in inv_summary)),
                "step/has_bed": int("bed" in inv_summary),
                "step/inventory_items": len(inv_summary),
                **{f"step/reward_{k}": v for k, v in reward_components.items()},
            }, step=self.global_step)

    def log_episode_end(self, episode_stats: dict):
        """Record statistics at end of episode."""
        duration = time.time() - self.episode_start_time

        episode_record = {
            "episode": self.current_episode,
            "global_step": self.global_step,
            "duration_seconds": duration,
            "total_reward": episode_stats.get("total_reward", 0),
            "final_health": episode_stats.get("final_health", 0),
            "crops_harvested": episode_stats.get("crops_harvested", 0),
            "animals_bred": episode_stats.get("animals_bred", 0),
            "blocks_placed": episode_stats.get("blocks_placed", 0),
            "bed_placed": episode_stats.get("bed_placed", False),
            "curriculum_stage": episode_stats.get("curriculum_stage", 0),
            "cause_of_end": episode_stats.get("cause_of_end", "timeout"),
            "tools_crafted": episode_stats.get("tools_crafted", 0),
        }

        self.episode_data.append(episode_record)

        wandb.log({
            "episode/total_reward": episode_stats.get("total_reward", 0),
            "episode/duration": duration,
            "episode/crops_harvested": episode_stats.get("crops_harvested", 0),
            "episode/blocks_placed": episode_stats.get("blocks_placed", 0),
            "episode/bed_placed": int(episode_stats.get("bed_placed", False)),
            "episode/tools_crafted": episode_stats.get("tools_crafted", 0),
        }, step=self.global_step)

        self.current_episode += 1
        self.episode_start_time = time.time()

    def log_task_attempt(self, task_name: str, success: bool,
                         duration_steps: int, details: dict = None):
        self.task_completion.append({
            "episode": self.current_episode,
            "global_step": self.global_step,
            "task": task_name,
            "success": success,
            "duration_steps": duration_steps,
            **(details or {})
        })
        wandb.log({
            f"task/{task_name}/success_rate": int(success),
            f"task/{task_name}/duration": duration_steps,
        }, step=self.global_step)

    def log_curriculum_advance(self, from_stage: str, to_stage: str,
                                trigger_metric: float):
        self.curriculum_progress.append({
            "global_step": self.global_step,
            "from_stage": from_stage,
            "to_stage": to_stage,
            "trigger_metric": trigger_metric,
            "timestamp": time.time()
        })
        wandb.log({
            "curriculum/stage_change": 1,
            "curriculum/current_stage": to_stage,
        }, step=self.global_step)
        print(f"\n[CURRICULUM] {from_stage} -> {to_stage} "
              f"(metric: {trigger_metric:.3f})")

    def save_checkpoint(self, step: int):
        if self.step_data:
            pd.DataFrame(self.step_data).to_csv(
                self.log_dir / f"steps_{step}.csv", index=False)
        if self.episode_data:
            pd.DataFrame(self.episode_data).to_csv(
                self.log_dir / f"episodes_{step}.csv", index=False)
        if self.task_completion:
            pd.DataFrame(self.task_completion).to_csv(
                self.log_dir / f"tasks_{step}.csv", index=False)

        summary = {
            "total_steps": self.global_step,
            "total_episodes": self.current_episode,
            "curriculum_advances": len(self.curriculum_progress),
            "curriculum_log": self.curriculum_progress,
        }
        with open(self.log_dir / f"summary_{step}.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[CHECKPOINT] Saved at step {step}")

    def finalize(self):
        self.save_checkpoint(self.global_step)
        wandb.finish()
        print(f"\n[DONE] Data saved to: {self.log_dir}")
