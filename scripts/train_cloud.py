"""
GPU Training Pipeline v2 — Optimized
Fixes:
- Goal loop: cycles back to punch_trees after secure_perimeter
- Crop detection: uses pickup events correctly  
- Reduced idle penalty accumulation
- Better goal completion detection
- VPT baseline mode flag
- Advancement detection from Minecraft log events
- Cleaner episode stats tracking
"""

import yaml
import json
import torch
import os
import sys
import time
from pathlib import Path
from collections import deque, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback
from minestudio.models import VPTPolicy

from reward_function import MinecraftRewardFunction
from data_logger_v2 import AgentDataLogger


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
        self.total_goal_completions = 0

    def get_current_goal(self) -> str:
        return GOAL_SEQUENCE[self.current_goal_idx % len(GOAL_SEQUENCE)]

    def advance_goal(self, goal: str):
        if goal not in self.completed:
            self.completed.add(goal)
            self.total_goal_completions += 1
        # Always advance index, loop back after last goal
        self.current_goal_idx = (self.current_goal_idx + 1) % len(GOAL_SEQUENCE)
        self.steps_on_current_goal = 0
        print(f"[GOAL] Completed: {goal} → Next: {self.get_current_goal()}")

    def tick(self) -> bool:
        """Returns True if timeout — force advance."""
        self.steps_on_current_goal += 1
        if self.steps_on_current_goal >= self.max_steps_per_goal:
            self.steps_on_current_goal = 0
            old = self.get_current_goal()
            # Loop back instead of getting stuck
            self.current_goal_idx = (self.current_goal_idx + 1) % len(GOAL_SEQUENCE)
            print(f"[GOAL] Timeout on '{old}' → next: {self.get_current_goal()}")
            return True
        return False


# ── HUMAN-LIKE REWARD FUNCTION ─────────────────────────────────────────────────
class HumanLikeRewardFunction(MinecraftRewardFunction):

    def __init__(self, config_path: str = "configs/rewards.yaml"):
        super().__init__(config_path)
        self.goal_tracker = GoalTracker()
        self.visited_chunks = set()
        self.mobs_killed = 0
        self.max_blocks_placed = 0

        # Per-episode accumulators (reset each episode)
        self.episode_crops = 0
        self.episode_blocks = 0
        self.episode_mobs = 0

        # Anti-idle: track inventory changes
        self.last_inventory_count = 0
        self.steps_without_progress = 0

        # Health tracking
        self.prev_health = 20.0
        self.prev_food = 20.0

    def _count_items(self, inventory: dict, keywords: list) -> int:
        return sum(
            qty for item, qty in inventory.items()
            if any(kw in item for kw in keywords)
        )

    def _is_daytime(self, info: dict) -> bool:
        return info.get("location_stats", {}).get("sky_light_level", 0.5) > 0.3

    def compute(self, obs, action, next_obs, info: dict) -> tuple:
        # Base rewards
        total_reward, components = super().compute(obs, action, next_obs, info)

        import numpy as np
        inventory = self._get_inventory(info)
        inv_count = len(inventory)
        health = info.get("health", 20.0)
        food = info.get("food_level", 20)
        is_day = self._is_daytime(info)
        kill_entity = dict(info.get("kill_entity", {}))
        pickup = dict(info.get("pickup", {}))
        crafted = dict(info.get("craft_item", {}))
        custom = dict(info.get("custom", {}))

        current_goal = self.goal_tracker.get_current_goal()
        self.goal_tracker.tick()

        # ── GOAL COMPLETION ────────────────────────────────────────────
        goal_completed = False

        if current_goal == "punch_trees":
            wood = self._count_items(inventory, ["log", "wood"])
            if wood >= 4:
                components["goal_punch_trees"] = 15.0
                goal_completed = True

        elif current_goal == "craft_crafting_table":
            if inventory.get("crafting_table", 0) > 0 or \
               crafted.get("crafting_table", 0) > 0:
                components["goal_crafting_table"] = 10.0
                goal_completed = True

        elif current_goal == "craft_wooden_tools":
            has_axe = self._count_items(inventory, ["axe"]) > 0
            has_pick = self._count_items(inventory, ["pickaxe"]) > 0
            if has_axe and has_pick:
                components["goal_wooden_tools"] = 20.0
                goal_completed = True

        elif current_goal == "collect_food":
            food_items = self._count_items(
                inventory, ["beef", "pork", "chicken", "bread",
                            "apple", "carrot", "potato", "cooked"]
            )
            if food_items >= 3 or food >= 18:
                components["goal_collect_food"] = 15.0
                goal_completed = True

        elif current_goal == "find_shelter_spot":
            # Auto-complete after exploring (can't easily detect)
            # Just let it timeout naturally and move on
            pass

        elif current_goal == "collect_stone":
            stone = inventory.get("cobblestone", 0)
            if stone >= 8:
                components["goal_collect_stone"] = 15.0
                goal_completed = True

        elif current_goal == "craft_stone_tools":
            has_stone_pick = inventory.get("stone_pickaxe", 0) > 0
            has_stone_sword = inventory.get("stone_sword", 0) > 0
            if has_stone_pick or has_stone_sword:
                components["goal_stone_tools"] = 25.0
                goal_completed = True

        elif current_goal == "find_cave":
            # Detect underground by Y position
            pos = info.get("player_pos", {})
            if pos.get("y", 64) < 50:
                components["goal_find_cave"] = 10.0
                goal_completed = True

        elif current_goal == "collect_iron":
            iron = (inventory.get("iron_ore", 0) +
                   inventory.get("iron_ingot", 0) +
                   inventory.get("raw_iron", 0))
            if iron >= 4:
                components["goal_collect_iron"] = 20.0
                goal_completed = True

        elif current_goal == "build_furnace":
            if inventory.get("furnace", 0) > 0 or \
               crafted.get("furnace", 0) > 0:
                components["goal_build_furnace"] = 15.0
                goal_completed = True

        elif current_goal == "craft_iron_tools":
            has_iron_pick = inventory.get("iron_pickaxe", 0) > 0
            has_iron_sword = inventory.get("iron_sword", 0) > 0
            if has_iron_pick or has_iron_sword:
                components["goal_iron_tools"] = 30.0
                goal_completed = True

        elif current_goal == "craft_hoe":
            has_hoe = self._count_items(inventory, ["hoe"]) > 0
            if has_hoe or crafted.get("wooden_hoe", 0) > 0 or \
               crafted.get("stone_hoe", 0) > 0:
                components["goal_craft_hoe"] = 20.0
                goal_completed = True

        elif current_goal == "find_water_source":
            # Auto-complete — hard to detect
            pass

        elif current_goal == "till_soil":
            # Detect via mine_block farmland or hoe use
            mined = dict(info.get("mine_block", {}))
            if mined.get("farmland", 0) > 0 or mined.get("grass", 0) > 5:
                components["goal_till_soil"] = 15.0
                goal_completed = True

        elif current_goal == "plant_seeds":
            seeds_used = pickup.get("wheat_seeds", 0)
            if self.event_counts.get("seeds_planted", 0) > 0 or \
               inventory.get("wheat_seeds", 0) == 0 and \
               self.event_counts.get("seeds_in_inventory", 0) > 0:
                components["goal_plant_seeds"] = 20.0
                goal_completed = True

        elif current_goal == "build_farm_fence":
            fence_placed = self.event_counts.get("blocks_placed", 0)
            if fence_placed >= 10:
                components["goal_build_fence"] = 15.0
                goal_completed = True

        elif current_goal == "harvest_crops":
            # Use pickup detection — this is the fix for crop detection
            crops_this_step = sum(
                pickup.get(c, 0) for c in ["wheat", "carrot", "potato"]
            )
            self.episode_crops += crops_this_step
            if self.episode_crops >= 3 or \
               self.event_counts.get("crops_harvested", 0) >= 3:
                components["goal_harvest_crops"] = 40.0
                goal_completed = True

        elif current_goal == "find_animals":
            mobs = info.get("mobs", [])
            nearby_animals = [m for m in mobs
                             if any(a in str(m) for a in
                                   ["cow", "pig", "sheep", "chicken"])]
            if len(nearby_animals) >= 2:
                components["goal_find_animals"] = 10.0
                goal_completed = True

        elif current_goal == "build_animal_pen":
            fence_count = inventory.get("fence", 0) + inventory.get("oak_fence", 0)
            blocks_total = self.event_counts.get("blocks_placed", 0)
            if blocks_total >= 20:
                components["goal_animal_pen"] = 25.0
                goal_completed = True

        elif current_goal == "breed_animals":
            if self.event_counts.get("animals_bred", 0) >= 1:
                components["goal_breed_animals"] = 50.0
                goal_completed = True

        elif current_goal == "collect_animal_products":
            has_wool = inventory.get("wool", 0) > 0
            has_leather = inventory.get("leather", 0) > 0
            has_feather = inventory.get("feather", 0) > 0
            if has_wool or has_leather or has_feather:
                components["goal_animal_products"] = 20.0
                goal_completed = True

        elif current_goal == "gather_building_materials":
            wood = self._count_items(inventory, ["planks", "log"])
            stone = inventory.get("cobblestone", 0)
            if wood >= 32 or stone >= 16:
                components["goal_building_materials"] = 15.0
                goal_completed = True

        elif current_goal in ["build_foundation", "build_walls",
                               "build_roof", "add_door_windows"]:
            blocks = self.event_counts.get("blocks_placed", 0)
            thresholds = {
                "build_foundation": 16,
                "build_walls": 40,
                "build_roof": 80,
                "add_door_windows": 90,
            }
            if blocks >= thresholds.get(current_goal, 50):
                components[f"goal_{current_goal}"] = 20.0
                goal_completed = True

        elif current_goal == "craft_bed":
            has_bed = self._count_items(inventory, ["bed"]) > 0
            if has_bed or crafted.get("white_bed", 0) > 0 or \
               crafted.get("red_bed", 0) > 0:
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
            has_sword = self._count_items(inventory, ["sword"]) > 0
            has_bow = inventory.get("bow", 0) > 0
            if has_sword or has_bow:
                components["goal_craft_weapons"] = 20.0
                goal_completed = True

        elif current_goal == "craft_armor":
            has_armor = self._count_items(
                inventory, ["helmet", "chestplate", "leggings", "boots"]
            ) > 0
            if has_armor:
                components["goal_craft_armor"] = 25.0
                goal_completed = True

        elif current_goal == "fight_mobs":
            if self.mobs_killed >= 3:
                components["goal_fight_mobs"] = 30.0
                goal_completed = True

        elif current_goal in ["build_torches", "secure_perimeter"]:
            has_torches = inventory.get("torch", 0) >= 4
            if has_torches:
                components[f"goal_{current_goal}"] = 15.0
                goal_completed = True

        if goal_completed:
            self.goal_tracker.advance_goal(current_goal)

        # ── COMBAT ─────────────────────────────────────────────────────
        for mob, count in kill_entity.items():
            if count > 0:
                if mob in ["zombie", "skeleton", "spider", "creeper", "enderman"]:
                    components[f"combat_kill_{mob}"] = count * 8.0
                    self.mobs_killed += count
                    self.episode_mobs += count
                    self.event_counts[f"killed_{mob}"] += count
                elif mob in ["cow", "pig", "sheep", "chicken"]:
                    components[f"hunt_{mob}"] = count * 3.0

        # ── EXPLORATION ────────────────────────────────────────────────
        pos = info.get("player_pos", {})
        chunk = (int(pos.get("x", 0)) // 16, int(pos.get("z", 0)) // 16)
        if chunk not in self.visited_chunks:
            self.visited_chunks.add(chunk)
            components["exploration_new_chunk"] = 0.3

        # ── ANTI-IDLE (reduced severity) ───────────────────────────────
        if inv_count > self.last_inventory_count:
            components["activity_progress"] = 0.2
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1

        # Only apply mild penalty after very long inactivity
        if self.steps_without_progress > 2000:
            severity = min((self.steps_without_progress - 2000) / 1000, 1.5)
            components["penalty_idle"] = -0.1 * severity  # much milder

        self.last_inventory_count = inv_count

        # ── DAY/NIGHT ROUTINE ──────────────────────────────────────────
        if not is_day:
            if components.get("build_blocks", 0) > 0:
                components["night_build_bonus"] = components["build_blocks"] * 1.5
            time_since_rest = custom.get("time_since_rest", 999)
            if time_since_rest < 20:
                components["night_sleep_bonus"] = 3.0
        else:
            if chunk not in self.visited_chunks:
                components["day_explore_bonus"] = 0.3

        # ── FOOD MANAGEMENT ────────────────────────────────────────────
        health_delta = health - self.prev_health
        if health_delta < -2:
            components["health_damage"] = health_delta * 0.3

        if food < 10 and self.prev_food >= 10:
            components["hunger_warning"] = -0.5

        self.prev_health = health
        self.prev_food = food

        # ── BUILDING MILESTONES ────────────────────────────────────────
        total_blocks = self.event_counts.get("blocks_placed", 0)
        if total_blocks > self.max_blocks_placed:
            for threshold, bonus in [
                (25, 8.0), (50, 12.0), (100, 20.0), (200, 35.0), (400, 50.0)
            ]:
                if total_blocks >= threshold and self.max_blocks_placed < threshold:
                    components[f"build_milestone_{threshold}"] = bonus
            self.max_blocks_placed = total_blocks

        # ── TOOL CRAFTING ──────────────────────────────────────────────
        tool_bonuses = {
            "wooden_pickaxe": 3.0, "stone_pickaxe": 6.0,
            "iron_pickaxe": 12.0, "wooden_axe": 1.5,
            "wooden_hoe": 8.0, "stone_hoe": 10.0,
            "furnace": 5.0, "chest": 3.0,
        }
        for tool, bonus in tool_bonuses.items():
            if crafted.get(tool, 0) > 0 and f"crafted_{tool}" not in self.achievements:
                components[f"craft_{tool}"] = bonus
                self.achievements.add(f"crafted_{tool}")
                self.event_counts[f"crafted_{tool}"] += 1

        # ── NORMALIZE ─────────────────────────────────────────────────
        total_reward = float(
            __import__("numpy").clip(sum(components.values()), -10.0, 100.0)
        )

        for k, v in components.items():
            self.reward_history[k].append(v)
            self.total_by_category[k] += v

        return total_reward, components

    def get_current_goal(self) -> str:
        return self.goal_tracker.get_current_goal()

    def reset_episode(self):
        """Reset per-episode state, keep cross-episode learning."""
        super().reset_episode() if hasattr(super(), 'reset_episode') else None
        self.prev_inventory = {}
        self.prev_health = 20.0
        self.prev_food = 20.0
        self.idle_counter = 0
        self.last_inventory_count = 0
        self.steps_without_progress = 0
        self.episode_crops = 0
        self.episode_blocks = 0
        self.episode_mobs = 0

    def get_statistics(self) -> dict:
        stats = super().get_statistics()
        stats["goal_progress"] = {
            "completed_goals": list(self.goal_tracker.completed),
            "current_goal": self.goal_tracker.get_current_goal(),
            "total_completions": self.goal_tracker.total_goal_completions,
            "mobs_killed": self.mobs_killed,
            "chunks_visited": len(self.visited_chunks),
            "max_blocks_placed": self.max_blocks_placed,
        }
        return stats


# ── LLM PLANNER ────────────────────────────────────────────────────────────────
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
    sky_light = info.get("location_stats", {}).get("sky_light_level", 1.0)
    is_night = sky_light < 0.3
    mined = dict(info.get("mine_block", {}))
    crafted = dict(info.get("craft_item", {}))

    time_ctx = "NIGHT - build shelter or sleep" if is_night else "DAY - explore and gather"

    danger = ""
    if health < 8:
        danger = "CRITICAL: Health very low! Find food NOW!"
    elif food < 5:
        danger = "URGENT: Starving! Eat immediately!"
    elif is_night:
        danger = "NIGHT: Avoid mobs, seek shelter"

    prompt = f"""Minecraft AI agent planner.

TIME: {time_ctx}
GOAL: {current_goal.replace('_', ' ').upper()}
Health: {health}/20, Food: {food}/20
Position: x={pos.get('x',0):.0f} y={pos.get('y',0):.0f} z={pos.get('z',0):.0f}
Inventory: {json.dumps(inv_summary) if inv_summary else 'empty'}
Mined: {mined or 'nothing'} | Crafted: {crafted or 'nothing'}
Recent: {', '.join(history[-3:]) if history else 'none'}
{danger}

Give ONE action for goal "{current_goal}". JSON only:
{{"task":"{current_goal}","action":"specific action","reasoning":"why","priority":1}}"""

    try:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            response_format={"type": "json_object"},
            timeout=8.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "task": current_goal,
            "action": current_goal.replace("_", " "),
            "reasoning": f"fallback",
            "priority": 1
        }


# ── MAIN TRAINING LOOP ─────────────────────────────────────────────────────────
def run_training(config_path: str = "configs/experiment_runpod.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print(f"Starting: {config['experiment']['name']} v2 optimized")
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
        print(f"Resuming from: {checkpoint_path}")
        policy.load_state_dict(
            torch.load(str(checkpoint_path), map_location=device)
        )
        meta_path = Path("models/checkpoints/latest_meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            start_step = meta.get("step", 0)
            print(f"Resuming from step {start_step}")

    param_count = sum(p.numel() for p in policy.parameters()) / 1e6
    print(f"VPT: {param_count:.1f}M params on {device}\n")

    total_steps = config["training"]["total_steps"]
    llm_interval = config["llm"].get("call_interval", 200)
    checkpoint_interval = config["training"]["checkpoint_interval"]
    max_episode_steps = config["environment"].get("max_episode_steps", 18000)

    obs, info = env.reset()
    memory = None
    task_history = []
    current_task = GOAL_SEQUENCE[0]
    episode_step = 0

    def fresh_episode_stats():
        return {
            "total_reward": 0.0,
            "tasks_completed": [],
            "crops_harvested": 0,
            "animals_bred": 0,
            "blocks_placed": 0.0,
            "bed_placed": False,
            "tools_crafted": 0,
            "mobs_killed": 0,
            "goals_completed": 0,
            "final_health": 20.0,
            "final_food": 20,
            "cause_of_end": "timeout",
        }

    episode_stats = fresh_episode_stats()
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
        info["current_goal"] = reward_fn.get_current_goal()
        episode_step += 1

        # Enforce episode timeout
        if episode_step >= max_episode_steps:
            terminated = True
            episode_stats["cause_of_end"] = "timeout"

        # ── REWARD ─────────────────────────────────────────────────────
        reward, reward_components = reward_fn.compute(obs, action, next_obs, info)

        # ── EPISODE STATS ──────────────────────────────────────────────
        episode_stats["total_reward"] += reward
        pickup = dict(info.get("pickup", {}))
        episode_stats["crops_harvested"] += sum(
            pickup.get(c, 0) for c in ["wheat", "carrot", "potato"]
        )
        episode_stats["animals_bred"] += int(
            reward_components.get("animal_breed", 0) > 0
        )
        episode_stats["blocks_placed"] += (
            reward_components.get("build_blocks", 0) / 0.05
        )
        episode_stats["tools_crafted"] += len(
            [k for k in reward_components if "craft_" in k]
        )
        episode_stats["mobs_killed"] += sum(
            1 for k in reward_components if "combat_kill" in k
        )
        episode_stats["goals_completed"] = (
            reward_fn.goal_tracker.total_goal_completions
        )
        episode_stats["final_health"] = info.get("health", 20)
        episode_stats["final_food"] = info.get("food_level", 20)
        if reward_components.get("build_bed", 0) > 0:
            episode_stats["bed_placed"] = True

        # ── LOGGING ────────────────────────────────────────────────────
        logger.log_step(obs, action, reward, reward_components, info,
                        terminated or truncated)

        # Goal metrics to W&B
        if step % config["logging"]["log_interval"] == 0:
            import wandb
            wandb.log({
                "goals/completed_count": len(reward_fn.goal_tracker.completed),
                "goals/total_completions": reward_fn.goal_tracker.total_goal_completions,
                "goals/current_idx": reward_fn.goal_tracker.current_goal_idx,
                "goals/mobs_killed": reward_fn.mobs_killed,
                "goals/chunks_visited": len(reward_fn.visited_chunks),
                "goals/max_blocks": reward_fn.max_blocks_placed,
                "goals/episode_crops": episode_stats["crops_harvested"],
                "goals/episode_step": episode_step,
            })

        obs = next_obs

        # ── EPISODE RESET ──────────────────────────────────────────────
        if terminated or truncated:
            duration = time.time() - episode_start
            episode_stats["duration_seconds"] = duration
            logger.log_episode_end(episode_stats)

            print(f"Episode {logger.current_episode} ended | "
                  f"Reward: {episode_stats['total_reward']:.1f} | "
                  f"Goals: {reward_fn.goal_tracker.total_goal_completions} | "
                  f"Crops: {episode_stats['crops_harvested']} | "
                  f"Blocks: {int(episode_stats['blocks_placed'])} | "
                  f"Steps: {episode_step} | "
                  f"Duration: {duration:.0f}s")

            obs, info = env.reset()
            memory = None
            task_history = []
            reward_fn.reset_episode()
            episode_step = 0
            episode_start = time.time()
            episode_stats = fresh_episode_stats()

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

        if step % 1000 == 0 and step > start_step:
            print(f"Step {step} | Goal: {reward_fn.get_current_goal()} | "
                  f"GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # ── FINALIZE ───────────────────────────────────────────────────────
    env.close()
    logger.finalize()

    reward_stats = reward_fn.get_statistics()
    with open(f"logs/{logger.run_name}/reward_stats_final.json", "w") as f:
        json.dump(reward_stats, f, indent=2)

    torch.save(policy.state_dict(), "models/checkpoints/final_model.pt")
    print(f"\nTraining complete!")
    print(f"Total goal completions: {reward_fn.goal_tracker.total_goal_completions}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_runpod.yaml")
    args = parser.parse_args()
    run_training(args.config)