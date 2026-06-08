"""
GPU Training Pipeline v2 — Human-like Minecraft Agent
Fixed version with:
- Episode timeout (max_episode_steps enforced)
- W&B step conflict resolved (resume=never)
- Correct episode_step counter
- Auto-resume from checkpoint
"""

import yaml
import json
import torch
import os
import sys
import time
from pathlib import Path
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback
from minestudio.models import VPTPolicy

from reward_function import MinecraftRewardFunction
from data_logger_v2 import AgentDataLogger
from llm_planner import get_next_task


# ── GOAL SEQUENCE ──────────────────────────────────────────────────────────────
GOAL_SEQUENCE = [
    "punch_trees", "craft_crafting_table", "craft_wooden_tools", "collect_food",
    "find_shelter_spot", "collect_stone", "craft_stone_tools", "find_cave",
    "collect_iron", "build_furnace", "craft_iron_tools", "craft_hoe",
    "find_water_source", "till_soil", "plant_seeds", "build_farm_fence",
    "harvest_crops", "find_animals", "build_animal_pen", "breed_animals",
    "collect_animal_products", "gather_building_materials", "build_foundation",
    "build_walls", "build_roof", "add_door_windows", "craft_bed",
    "place_bed_inside", "sleep_through_night", "craft_weapons", "craft_armor",
    "fight_mobs", "build_torches", "secure_perimeter",
]


class GoalTracker:
    def __init__(self):
        self.completed = set()
        self.current_goal_idx = 0
        self.steps_on_current_goal = 0
        self.max_steps_per_goal = 3000

    def get_current_goal(self) -> str:
        if self.current_goal_idx < len(GOAL_SEQUENCE):
            return GOAL_SEQUENCE[self.current_goal_idx]
        return "explore_and_improve"

    def advance_goal(self, goal: str):
        if goal not in self.completed:
            self.completed.add(goal)
            self.current_goal_idx = min(
                self.current_goal_idx + 1, len(GOAL_SEQUENCE) - 1
            )
            self.steps_on_current_goal = 0
            print(f"[GOAL] Completed: {goal} → Next: {self.get_current_goal()}")

    def tick(self) -> bool:
        self.steps_on_current_goal += 1
        if self.steps_on_current_goal >= self.max_steps_per_goal:
            self.steps_on_current_goal = 0
            old = self.get_current_goal()
            self.current_goal_idx = min(
                self.current_goal_idx + 1, len(GOAL_SEQUENCE) - 1
            )
            print(f"[GOAL] Timeout on '{old}' → forcing next: {self.get_current_goal()}")
            return True
        return False


class HumanLikeRewardFunction(MinecraftRewardFunction):

    def __init__(self, config_path: str = "configs/rewards.yaml"):
        super().__init__(config_path)
        self.goal_tracker = GoalTracker()
        self.steps_without_new_item = 0
        self.last_inventory_count = 0
        self.visited_chunks = set()
        self.mobs_killed = 0
        self.max_blocks_placed = 0

    def _count_specific_items(self, inventory: dict, keywords: list) -> int:
        return sum(
            qty for item, qty in inventory.items()
            if any(kw in item for kw in keywords)
        )

    def _is_daytime(self, info: dict) -> bool:
        sky_light = info.get("location_stats", {}).get("sky_light_level", 0.5)
        return sky_light > 0.3

    def compute(self, obs, action, next_obs, info: dict) -> tuple:
        total_reward, components = super().compute(obs, action, next_obs, info)

        import numpy as np
        inventory = self._get_inventory(info)
        inv_count = len(inventory)
        is_day = self._is_daytime(info)
        kill_entity = dict(info.get("kill_entity", {}))
        custom = dict(info.get("custom", {}))

        current_goal = self.goal_tracker.get_current_goal()
        self.goal_tracker.tick()

        # ── GOAL COMPLETION ────────────────────────────────────────────
        goal_completed = False

        if current_goal == "punch_trees":
            wood = self._count_specific_items(inventory, ["log", "wood"])
            if wood >= 4:
                components["goal_punch_trees"] = 15.0
                goal_completed = True

        elif current_goal == "craft_crafting_table":
            if inventory.get("crafting_table", 0) > 0:
                components["goal_crafting_table"] = 10.0
                goal_completed = True

        elif current_goal == "craft_wooden_tools":
            has_axe = self._count_specific_items(inventory, ["axe"]) > 0
            has_pick = self._count_specific_items(inventory, ["pickaxe"]) > 0
            if has_axe and has_pick:
                components["goal_wooden_tools"] = 20.0
                goal_completed = True

        elif current_goal == "collect_food":
            food_items = self._count_specific_items(
                inventory, ["beef", "pork", "chicken", "bread", "apple", "carrot", "potato"]
            )
            food_level = info.get("food_level", 20)
            if food_items >= 3 or food_level >= 18:
                components["goal_collect_food"] = 15.0
                goal_completed = True

        elif current_goal == "craft_stone_tools":
            has_stone_pick = inventory.get("stone_pickaxe", 0) > 0
            has_stone_sword = inventory.get("stone_sword", 0) > 0
            if has_stone_pick and has_stone_sword:
                components["goal_stone_tools"] = 25.0
                goal_completed = True

        elif current_goal == "collect_iron":
            iron = inventory.get("iron_ore", 0) + inventory.get("iron_ingot", 0)
            if iron >= 6:
                components["goal_collect_iron"] = 20.0
                goal_completed = True

        elif current_goal == "craft_hoe":
            has_hoe = self._count_specific_items(inventory, ["hoe"]) > 0
            if has_hoe:
                components["goal_craft_hoe"] = 20.0
                goal_completed = True

        elif current_goal == "harvest_crops":
            harvested = (
                self.event_counts.get("wheat_harvested", 0) +
                self.event_counts.get("carrot_harvested", 0) +
                self.event_counts.get("potato_harvested", 0)
            )
            if harvested >= 5:
                components["goal_harvest_crops"] = 40.0
                goal_completed = True

        elif current_goal == "breed_animals":
            if self.event_counts.get("animals_bred", 0) >= 2:
                components["goal_breed_animals"] = 50.0
                goal_completed = True

        elif current_goal == "craft_bed":
            has_bed = self._count_specific_items(inventory, ["bed"]) > 0
            if has_bed:
                components["goal_craft_bed"] = 30.0
                goal_completed = True

        elif current_goal == "place_bed_inside":
            if "bed_placed" in self.achievements:
                components["goal_place_bed"] = 50.0
                goal_completed = True

        elif current_goal == "sleep_through_night":
            time_since_rest = custom.get("time_since_rest", 999)
            if time_since_rest < 10 or "slept_through_night" in self.achievements:
                components["goal_sleep"] = 100.0
                goal_completed = True

        elif current_goal == "craft_weapons":
            has_sword = self._count_specific_items(inventory, ["sword"]) > 0
            if has_sword:
                components["goal_craft_weapons"] = 20.0
                goal_completed = True

        if goal_completed:
            self.goal_tracker.advance_goal(current_goal)

        # ── COMBAT ─────────────────────────────────────────────────────
        for mob, count in kill_entity.items():
            if count > 0:
                if mob in ["zombie", "skeleton", "spider", "creeper"]:
                    components[f"combat_kill_{mob}"] = count * 8.0
                    self.mobs_killed += count
                    self.event_counts[f"killed_{mob}"] += count
                elif mob in ["cow", "pig", "sheep", "chicken"]:
                    components[f"hunt_{mob}"] = count * 3.0

        # ── ANTI-IDLE ──────────────────────────────────────────────────
        if inv_count > self.last_inventory_count:
            components["activity_new_items"] = (inv_count - self.last_inventory_count) * 0.3
            self.steps_without_new_item = 0
        else:
            self.steps_without_new_item += 1

        if self.steps_without_new_item > 1000:
            severity = min((self.steps_without_new_item - 1000) / 500, 3.0)
            components["penalty_prolonged_idle"] = -0.2 * severity

        self.last_inventory_count = inv_count

        # ── DAY/NIGHT ROUTINE ──────────────────────────────────────────
        if not is_day:
            if components.get("build_blocks", 0) > 0:
                components["night_building_bonus"] = components["build_blocks"] * 2.0
        else:
            pos = info.get("player_pos", {})
            chunk = (int(pos.get("x", 0)) // 16, int(pos.get("z", 0)) // 16)
            if chunk not in self.visited_chunks:
                self.visited_chunks.add(chunk)
                components["day_exploration"] = 0.5

        # ── BUILDING MILESTONES ────────────────────────────────────────
        total_blocks = self.event_counts.get("blocks_placed", 0)
        if total_blocks > self.max_blocks_placed:
            for threshold, bonus in [(25, 10.0), (50, 15.0), (100, 25.0), (200, 40.0)]:
                if total_blocks >= threshold and self.max_blocks_placed < threshold:
                    components[f"build_milestone_{threshold}"] = bonus
            self.max_blocks_placed = total_blocks

        # ── NORMALIZE ─────────────────────────────────────────────────
        total_reward = float(np.clip(sum(components.values()), -15.0, 100.0))

        for k, v in components.items():
            self.reward_history[k].append(v)
            self.total_by_category[k] += v

        return total_reward, components

    def get_current_goal(self) -> str:
        return self.goal_tracker.get_current_goal()

    def get_statistics(self) -> dict:
        stats = super().get_statistics()
        stats["goal_progress"] = {
            "completed_goals": list(self.goal_tracker.completed),
            "current_goal": self.goal_tracker.get_current_goal(),
            "goals_completed_count": len(self.goal_tracker.completed),
            "mobs_killed": self.mobs_killed,
            "chunks_visited": len(self.visited_chunks),
            "max_blocks_placed": self.max_blocks_placed,
        }
        return stats


def get_human_like_task(info: dict, current_goal: str,
                         history: list, tech_progress: dict) -> dict:
    import os, json
    from openai import OpenAI

    inv = info.get("inventory", {})
    inv_summary = {
        s["type"]: s["quantity"]
        for s in inv.values()
        if s["type"] != "none"
    }
    pos = info.get("player_pos", {})
    health = info.get("health", 20)
    food = info.get("food_level", 20)
    location = info.get("location_stats", {})
    sky_light = location.get("sky_light_level", 1.0)
    is_night = sky_light < 0.3
    mined = dict(info.get("mine_block", {}))
    crafted = dict(info.get("craft_item", {}))

    time_context = "NIGHT - prioritize shelter/sleep/building" if is_night \
        else "DAY - prioritize exploration/gathering/farming"

    danger = ""
    if health < 8:
        danger = "CRITICAL: Health very low! Find food or shelter NOW!"
    elif food < 5:
        danger = "URGENT: Food almost empty! Eat something immediately!"
    elif is_night:
        danger = "NIGHT: Stay safe, build shelter, place bed if possible"

    prompt = f"""You are an expert Minecraft survival guide for an AI agent.

TIME: {time_context}
CURRENT GOAL: {current_goal.replace('_', ' ').upper()}

AGENT STATE:
- Health: {health}/20, Food: {food}/20
- Position: x={pos.get('x',0):.0f} y={pos.get('y',0):.0f} z={pos.get('z',0):.0f}
- Inventory: {json.dumps(inv_summary) if inv_summary else 'empty'}
- Just mined: {mined if mined else 'nothing'}
- Just crafted: {crafted if crafted else 'nothing'}
- Recent tasks: {', '.join(history[-3:]) if history else 'none'}

{danger}

Give ONE specific action to take RIGHT NOW toward the current goal "{current_goal}".
Reply ONLY with JSON:
{{
  "task": "{current_goal}",
  "action": "specific single action to take now",
  "reasoning": "one sentence why this progresses the goal",
  "priority": 1
}}"""

    try:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            response_format={"type": "json_object"},
            timeout=8.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "task": current_goal,
            "action": current_goal.replace("_", " "),
            "reasoning": f"fallback: {str(e)[:30]}",
            "priority": 1
        }


def run_training(config_path: str = "configs/experiment_runpod.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print(f"Starting: {config['experiment']['name']} v2")
    print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    logger = AgentDataLogger(config)
    reward_fn = HumanLikeRewardFunction()

    env = MinecraftSim(
        obs_size=(128, 128),
        callbacks=[
            RecordCallback(
                record_path=f"logs/{logger.run_name}/videos",
                fps=20,
                frame_type="pov"
            )
        ]
    )

    device = "cuda"
    policy = VPTPolicy.from_pretrained(
        "CraftJarvis/MineStudio_VPT.rl_from_early_game_2x"
    ).to(device)

    # Auto-resume from checkpoint
    checkpoint_path = Path("models/checkpoints/latest.pt")
    start_step = 0
    if checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        policy.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
        meta_path = Path("models/checkpoints/latest_meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            start_step = meta.get("step", 0)
            print(f"Resuming from step {start_step}")

    param_count = sum(p.numel() for p in policy.parameters()) / 1e6
    print(f"VPT model: {param_count:.1f}M parameters on {device}\n")

    total_steps = config["training"]["total_steps"]
    llm_interval = config["llm"].get("call_interval", 200)
    checkpoint_interval = config["training"]["checkpoint_interval"]

    # ── EPISODE TIMEOUT ────────────────────────────────────────────────
    max_episode_steps = config["environment"].get("max_episode_steps", 18000)

    obs, info = env.reset()
    memory = None
    task_history = []
    current_task = GOAL_SEQUENCE[0]
    episode_step = 0  # per-episode step counter

    episode_stats = {
        "total_reward": 0,
        "tasks_completed": [],
        "crops_harvested": 0,
        "animals_bred": 0,
        "blocks_placed": 0,
        "bed_placed": False,
        "tools_crafted": 0,
        "mobs_killed": 0,
        "goals_completed": 0,
        "final_health": 20,
        "final_food": 20,
    }

    print("Training started...\n")
    episode_start = time.time()

    for step in range(start_step, total_steps):

        # ── LLM PLANNING ───────────────────────────────────────────────
        if step % llm_interval == 0:
            current_goal = reward_fn.get_current_goal()
            planned = get_human_like_task(
                info, current_goal, task_history,
                reward_fn.tech_tree_progress
            )
            current_task = planned.get("task", current_goal)
            task_history.append(current_task)
            if len(task_history) > 20:
                task_history = task_history[-20:]

        info["current_task"] = current_task
        info["current_goal"] = reward_fn.get_current_goal()

        # ── ACTION ─────────────────────────────────────────────────────
        with torch.no_grad():
            action, memory = policy.get_action(obs, memory, input_shape='*')

        next_obs, _, terminated, truncated, info = env.step(action)
        info["current_task"] = current_task
        episode_step += 1

        # ── ENFORCE EPISODE TIMEOUT ────────────────────────────────────
        if episode_step >= max_episode_steps:
            terminated = True
            episode_stats["cause_of_end"] = "timeout"

        # ── REWARD ─────────────────────────────────────────────────────
        reward, reward_components = reward_fn.compute(obs, action, next_obs, info)

        # ── EPISODE STATS UPDATE ───────────────────────────────────────
        episode_stats["total_reward"] += reward
        episode_stats["crops_harvested"] += sum(
            dict(info.get("pickup", {})).get(c, 0)
            for c in ["wheat", "carrot", "potato"]
        )
        episode_stats["animals_bred"] += int(reward_components.get("animal_breed", 0) > 0)
        episode_stats["blocks_placed"] += reward_components.get("build_blocks", 0) / 0.05
        episode_stats["tools_crafted"] += len([k for k in reward_components if "craft_" in k])
        episode_stats["mobs_killed"] += sum(
            1 for k in reward_components if "combat_kill" in k
        )
        episode_stats["goals_completed"] = len(reward_fn.goal_tracker.completed)
        episode_stats["final_health"] = info.get("health", 20)
        episode_stats["final_food"] = info.get("food_level", 20)

        if reward_components.get("build_bed", 0) > 0:
            episode_stats["bed_placed"] = True

        # ── LOGGING ────────────────────────────────────────────────────
        logger.log_step(obs, action, reward, reward_components, info,
                        terminated or truncated)

        # Goal progress W&B logging
        if step % config["logging"]["log_interval"] == 0:
            import wandb
            wandb.log({
                "goals/completed_count": len(reward_fn.goal_tracker.completed),
                "goals/current_goal_idx": reward_fn.goal_tracker.current_goal_idx,
                "goals/mobs_killed": reward_fn.mobs_killed,
                "goals/chunks_visited": len(reward_fn.visited_chunks),
                "goals/max_blocks_placed": reward_fn.max_blocks_placed,
                "goals/episode_step": episode_step,
            })

        obs = next_obs

        # ── EPISODE RESET ──────────────────────────────────────────────
        if terminated or truncated:
            episode_duration = time.time() - episode_start
            episode_stats["duration_seconds"] = episode_duration

            logger.log_episode_end(episode_stats)

            print(f"Episode {logger.current_episode} ended | "
                  f"Reward: {episode_stats['total_reward']:.1f} | "
                  f"Goals: {len(reward_fn.goal_tracker.completed)} | "
                  f"Steps: {episode_step} | "
                  f"Duration: {episode_duration:.0f}s")

            obs, info = env.reset()
            memory = None
            task_history = []
            reward_fn.reset_episode()
            episode_step = 0  # RESET episode step counter
            episode_start = time.time()

            episode_stats = {
                "total_reward": 0,
                "tasks_completed": [],
                "crops_harvested": 0,
                "animals_bred": 0,
                "blocks_placed": 0,
                "bed_placed": False,
                "tools_crafted": 0,
                "mobs_killed": 0,
                "goals_completed": 0,
                "final_health": 20,
                "final_food": 20,
                "cause_of_end": "unknown",
            }

        # ── CHECKPOINT ─────────────────────────────────────────────────
        if step % checkpoint_interval == 0 and step > start_step:
            logger.save_checkpoint(step)

            torch.save(policy.state_dict(), f"models/checkpoints/step_{step}.pt")
            torch.save(policy.state_dict(), "models/checkpoints/latest.pt")
            with open("models/checkpoints/latest_meta.json", "w") as f:
                json.dump({"step": step, "episode": logger.current_episode}, f)

            reward_stats = reward_fn.get_statistics()
            with open(f"logs/{logger.run_name}/reward_stats_{step}.json", "w") as f:
                json.dump(reward_stats, f, indent=2)

            print(f"[CHECKPOINT] Step {step}/{total_steps} | "
                  f"GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB | "
                  f"Goals: {len(reward_fn.goal_tracker.completed)}/{len(GOAL_SEQUENCE)}")

        if step % 1000 == 0 and step > start_step:
            print(f"Step {step} | Goal: {reward_fn.get_current_goal()} | "
                  f"GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # ── FINALIZE ───────────────────────────────────────────────────────
    env.close()
    logger.finalize()

    reward_stats = reward_fn.get_statistics()
    with open(f"logs/{logger.run_name}/reward_stats_final.json", "w") as f:
        json.dump(reward_stats, f, indent=2)

    torch.save(policy.state_dict(), "models/checkpoints/final_model.pt")
    print(f"\nTraining complete!")
    print(f"Goals: {len(reward_fn.goal_tracker.completed)}/{len(GOAL_SEQUENCE)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_runpod.yaml")
    args = parser.parse_args()
    run_training(args.config)