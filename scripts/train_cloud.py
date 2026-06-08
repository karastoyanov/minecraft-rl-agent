"""
GPU Training Pipeline v2 — Human-like Minecraft Agent
Improvements over v1:
- Structured goal system (survival → tools → farm → shelter → combat)
- Anti-idle enforcement with forced exploration
- Human-like daily routine (day=gather, night=build/sleep)
- Weapon crafting and mob combat rewards
- Storage/chest management
- House building with bed as primary goal
- PPO-ready reward shaping
- OOM protection (reduced memory footprint)
- Auto-resume from checkpoint
"""

import yaml
import json
import torch
import os
import sys
import time
import random
from pathlib import Path
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback
from minestudio.models import VPTPolicy

from reward_function import MinecraftRewardFunction
from data_logger import AgentDataLogger
from llm_planner import get_next_task


# ─── GOAL SYSTEM ──────────────────────────────────────────────────────────────
# Ordered list of human-like goals the agent should pursue
GOAL_SEQUENCE = [
    # Stage 1: Basic survival
    "punch_trees",          # collect wood
    "craft_crafting_table", # make crafting table
    "craft_wooden_tools",   # axe, pickaxe
    "collect_food",         # kill animals, collect apples
    "find_shelter_spot",    # find a good location

    # Stage 2: Tools and mining
    "collect_stone",        # mine cobblestone
    "craft_stone_tools",    # stone pickaxe, axe, sword
    "find_cave",            # explore underground
    "collect_iron",         # mine iron ore
    "build_furnace",        # smelt iron
    "craft_iron_tools",     # iron pickaxe, sword, axe

    # Stage 3: Farming
    "craft_hoe",            # wooden or stone hoe
    "find_water_source",    # locate water for irrigation
    "till_soil",            # prepare farmland
    "plant_seeds",          # plant wheat/carrots/potatoes
    "build_farm_fence",     # protect crops
    "harvest_crops",        # collect first harvest

    # Stage 4: Animal husbandry
    "find_animals",         # locate cows/pigs/sheep
    "build_animal_pen",     # fenced enclosure
    "breed_animals",        # feed and breed
    "collect_animal_products", # milk, wool, meat

    # Stage 5: House and shelter
    "gather_building_materials", # wood, stone, glass
    "build_foundation",     # lay the floor
    "build_walls",          # erect walls
    "build_roof",           # close the roof
    "add_door_windows",     # door and glass panes
    "craft_bed",            # craft a bed (wool + planks)
    "place_bed_inside",     # place bed in house
    "sleep_through_night",  # MAIN GOAL

    # Stage 6: Combat and defense
    "craft_weapons",        # sword, bow
    "craft_armor",          # leather/iron armor
    "fight_mobs",           # kill zombies/skeletons
    "build_torches",        # light up area
    "secure_perimeter",     # light up around house
]


class GoalTracker:
    """Tracks which goals have been achieved and what to pursue next."""

    def __init__(self):
        self.completed = set()
        self.current_goal_idx = 0
        self.steps_on_current_goal = 0
        self.max_steps_per_goal = 3000  # force move on if stuck

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
        """Returns True if goal timeout reached — force advance."""
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


# ─── HUMAN-LIKE BEHAVIOR REWARDS ──────────────────────────────────────────────
class HumanLikeRewardFunction(MinecraftRewardFunction):
    """
    Extended reward function with human-like behavior incentives.
    Builds on top of the base reward function.
    """

    def __init__(self, config_path: str = "configs/rewards.yaml"):
        super().__init__(config_path)

        self.goal_tracker = GoalTracker()
        self.steps_without_new_item = 0
        self.last_inventory_count = 0
        self.recent_rewards = deque(maxlen=200)
        self.time_of_day_history = deque(maxlen=10)

        # Track combat
        self.mobs_killed = 0
        self.damage_taken = 0

        # Track building progress
        self.max_blocks_placed = 0
        self.building_started = False

        # Exploration grid (16x16 chunks)
        self.visited_chunks = set()

    def _is_daytime(self, info: dict) -> bool:
        """Estimate daytime from sky light level."""
        sky_light = info.get("location_stats", {}).get("sky_light_level", 0.5)
        return sky_light > 0.3

    def _count_specific_items(self, inventory: dict, keywords: list) -> int:
        """Count items matching any keyword."""
        return sum(
            qty for item, qty in inventory.items()
            if any(kw in item for kw in keywords)
        )

    def compute(self, obs, action, next_obs, info: dict) -> tuple:
        """Extended compute with human-like behavior rewards."""

        # Get base rewards
        total_reward, components = super().compute(obs, action, next_obs, info)

        inventory = self._get_inventory(info)
        inv_count = len(inventory)
        health = info.get("health", 20)
        food = info.get("food_level", 20)
        custom = dict(info.get("custom", {}))
        kill_entity = dict(info.get("kill_entity", {}))
        damage_dealt = dict(info.get("damage_dealt", {}))
        is_day = self._is_daytime(info)

        # ── GOAL PROGRESS REWARDS ──────────────────────────────────────
        current_goal = self.goal_tracker.get_current_goal()
        goal_timeout = self.goal_tracker.tick()

        # Check goal completion conditions
        goal_completed = False

        if current_goal == "punch_trees":
            wood = self._count_specific_items(inventory, ["log", "wood"])
            if wood >= 4:
                components["goal_punch_trees"] = 15.0
                goal_completed = True

        elif current_goal == "craft_crafting_table":
            if inventory.get("crafting_table", 0) > 0 or "crafting_table" in self.achievements:
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
            if food_items >= 3 or food >= 18:
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
            if self.event_counts.get("wheat_harvested", 0) + \
               self.event_counts.get("carrot_harvested", 0) + \
               self.event_counts.get("potato_harvested", 0) >= 5:
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
            if "slept_through_night" in self.achievements:
                components["goal_sleep"] = 100.0
                goal_completed = True

        elif current_goal == "craft_weapons":
            has_sword = self._count_specific_items(inventory, ["sword"]) > 0
            if has_sword:
                components["goal_craft_weapons"] = 20.0
                goal_completed = True

        if goal_completed:
            self.goal_tracker.advance_goal(current_goal)

        # ── COMBAT REWARDS ─────────────────────────────────────────────
        for mob, count in kill_entity.items():
            if count > 0:
                if mob in ["zombie", "skeleton", "spider", "creeper"]:
                    components[f"combat_kill_{mob}"] = count * 8.0
                    self.mobs_killed += count
                    self.event_counts[f"killed_{mob}"] += count
                elif mob in ["cow", "pig", "sheep", "chicken"]:
                    # Killing animals for food is okay but less rewarded
                    components[f"hunt_{mob}"] = count * 3.0

        # Bonus for surviving combat (took damage but survived)
        total_damage = sum(damage_dealt.values())
        if total_damage > 0:
            components["combat_dealt_damage"] = min(total_damage * 0.5, 5.0)

        # ── ANTI-IDLE: HUMAN-LIKE ACTIVITY REWARDS ─────────────────────
        # Reward for gaining new items (active resource collection)
        if inv_count > self.last_inventory_count:
            new_items = inv_count - self.last_inventory_count
            components["activity_new_items"] = new_items * 0.3
            self.steps_without_new_item = 0
        else:
            self.steps_without_new_item += 1

        # Increasing penalty for prolonged inactivity
        if self.steps_without_new_item > 1000:
            severity = min((self.steps_without_new_item - 1000) / 500, 3.0)
            components["penalty_prolonged_idle"] = -0.2 * severity

        self.last_inventory_count = inv_count

        # ── DAY/NIGHT ROUTINE ──────────────────────────────────────────
        # During night: reward building and sleeping, penalize being outside
        if not is_day:
            # Reward placing blocks at night (building shelter)
            if components.get("build_blocks", 0) > 0:
                components["night_building_bonus"] = components["build_blocks"] * 2.0

            # Reward sleep attempt at night
            time_since_rest = custom.get("time_since_rest", 999)
            if time_since_rest < 20:
                components["night_sleep_attempt"] = 5.0

        # During day: reward exploration and resource gathering
        else:
            pos = info.get("player_pos", {})
            chunk = (int(pos.get("x", 0)) // 16, int(pos.get("z", 0)) // 16)
            if chunk not in self.visited_chunks:
                self.visited_chunks.add(chunk)
                components["day_exploration"] = 0.5

        # ── FOOD MANAGEMENT ────────────────────────────────────────────
        # Reward eating when hungry (human-like behavior)
        if food < 14 and self.prev_food < food:
            components["food_management_ate"] = 2.0

        # Reward having food in inventory
        food_in_inv = self._count_specific_items(
            inventory, ["bread", "beef", "pork", "chicken", "carrot",
                        "potato", "apple", "steak", "cooked"]
        )
        if food_in_inv >= 5:
            components["food_stockpile"] = 0.5

        # ── BUILDING PROGRESS ──────────────────────────────────────────
        # Track maximum blocks placed (proxy for house size)
        total_blocks = self.event_counts.get("blocks_placed", 0)
        if total_blocks > self.max_blocks_placed:
            if total_blocks >= 25 and self.max_blocks_placed < 25:
                components["build_milestone_25"] = 10.0
            if total_blocks >= 50 and self.max_blocks_placed < 50:
                components["build_milestone_50"] = 15.0
            if total_blocks >= 100 and self.max_blocks_placed < 100:
                components["build_milestone_100"] = 25.0
            if total_blocks >= 200 and self.max_blocks_placed < 200:
                components["build_milestone_200"] = 40.0
            self.max_blocks_placed = total_blocks

        # ── STORAGE / CHEST REWARD ─────────────────────────────────────
        # Reward for crafting and using chest (human-like storage behavior)
        if inventory.get("chest", 0) > 0 and "crafted_chest" not in self.achievements:
            components["storage_chest_crafted"] = 5.0
            self.achievements.add("crafted_chest")

        # ── NORMALIZE ─────────────────────────────────────────────────
        import numpy as np
        total_reward = sum(components.values())
        total_reward = float(np.clip(total_reward, -15.0, 100.0))

        self.recent_rewards.append(total_reward)

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


# ─── IMPROVED LLM PLANNER ─────────────────────────────────────────────────────
def get_human_like_task(info: dict, current_goal: str,
                         history: list, tech_progress: dict) -> dict:
    """Context-aware LLM planner focused on current goal."""
    from llm_planner import client

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

PRIORITY GUIDE for current goal "{current_goal}":
- punch_trees: Find and punch oak/birch trees, collect at least 4 logs
- craft_crafting_table: Use logs to make planks, craft crafting table
- craft_wooden_tools: Make wooden pickaxe and axe at crafting table
- collect_food: Kill animals or collect apples/berries for food
- collect_stone: Mine cobblestone with pickaxe
- craft_stone_tools: Make stone pickaxe and stone sword
- collect_iron: Find iron ore in caves or hillsides, mine it
- build_furnace: Craft furnace with 8 cobblestone, smelt iron
- craft_hoe: Make hoe with sticks and planks/stone
- till_soil: Find flat land near water, use hoe on dirt
- plant_seeds: Plant wheat seeds or carrots on tilled soil
- harvest_crops: Wait for crops, harvest when grown
- find_animals: Explore for cows, pigs, sheep
- build_animal_pen: Place fence blocks to enclose 2+ animals
- breed_animals: Feed wheat to cows, carrots to pigs
- gather_building_materials: Collect 64+ wood, 32+ stone, 4+ glass
- build_foundation: Place floor blocks in 5x5 or larger area
- build_walls: Stack 3 blocks high around perimeter
- build_roof: Close the top with slabs or planks
- craft_bed: Get 3 wool from sheep, craft bed with 3 planks
- place_bed_inside: Place bed inside enclosed structure
- sleep_through_night: Right-click bed at night to sleep
- craft_weapons: Make sword, optionally bow and arrows
- fight_mobs: Attack zombies/skeletons with sword

Give ONE specific action to take RIGHT NOW toward the current goal.
Reply ONLY with JSON:
{{
  "task": "{current_goal}",
  "action": "specific single action to take now",
  "reasoning": "one sentence why this progresses the goal",
  "priority": 1
}}"""

    try:
        from openai import OpenAI
        import os
        client_obj = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client_obj.chat.completions.create(
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


# ─── MAIN TRAINING LOOP ───────────────────────────────────────────────────────
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

    # ── ENVIRONMENT (reduced memory footprint) ─────────────────────────
    env = MinecraftSim(
        obs_size=(128, 128),   # keep small for memory
        callbacks=[
            RecordCallback(
                record_path=f"logs/{logger.run_name}/videos",
                fps=20,
                frame_type="pov"
            )
        ]
    )

    # ── MODEL ──────────────────────────────────────────────────────────
    device = "cuda"
    policy = VPTPolicy.from_pretrained(
        "CraftJarvis/MineStudio_VPT.rl_from_early_game_2x"
    ).to(device)

    # Auto-resume from latest checkpoint
    checkpoint_path = Path("models/checkpoints/latest.pt")
    start_step = 0
    if checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        policy.load_state_dict(
            torch.load(str(checkpoint_path), map_location=device)
        )
        # Try to read saved step
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

    obs, info = env.reset()
    memory = None
    task_history = []
    current_task = GOAL_SEQUENCE[0]

    episode_stats = {
        "total_reward": 0,
        "tasks_completed": [],
        "crops_harvested": 0,
        "animals_bred": 0,
        "blocks_placed": 0,
        "bed_placed": False,
        "tools_crafted": 0,
        "mobs_killed": 0,
    }

    print("Training started...\n")
    episode_start = time.time()

    for step in range(start_step, total_steps):

        # ── LLM PLANNING ───────────────────────────────────────────────
        if step % llm_interval == 0:
            current_goal = reward_fn.get_current_goal()
            planned = get_human_like_task(
                info,
                current_goal,
                task_history,
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

        # ── REWARD ─────────────────────────────────────────────────────
        reward, reward_components = reward_fn.compute(obs, action, next_obs, info)

        # ── EPISODE STATS ──────────────────────────────────────────────
        episode_stats["total_reward"] += reward
        episode_stats["crops_harvested"] += sum(
            dict(info.get("pickup", {})).get(c, 0)
            for c in ["wheat", "carrot", "potato"]
        )
        episode_stats["animals_bred"] += reward_components.get("animal_breed", 0) > 0
        episode_stats["blocks_placed"] += reward_components.get("build_blocks", 0) / 0.05
        episode_stats["tools_crafted"] += len(
            [k for k in reward_components if "craft_" in k]
        )
        episode_stats["mobs_killed"] += sum(
            1 for k in reward_components if "combat_kill" in k
        )
        if reward_components.get("build_bed", 0) > 0:
            episode_stats["bed_placed"] = True

        # ── LOGGING ────────────────────────────────────────────────────
        logger.log_step(obs, action, reward, reward_components, info,
                        terminated or truncated)

        # Extra W&B metrics for goal tracking
        if step % config["logging"]["log_interval"] == 0:
            import wandb
            wandb.log({
                "goal/current_goal_idx": reward_fn.goal_tracker.current_goal_idx,
                "goal/goals_completed": len(reward_fn.goal_tracker.completed),
                "goal/mobs_killed": reward_fn.mobs_killed,
                "goal/chunks_visited": len(reward_fn.visited_chunks),
                "goal/max_blocks_placed": reward_fn.max_blocks_placed,
                "goal/steps_without_new_item": reward_fn.steps_without_new_item,
            }, step=step)

        obs = next_obs

        # ── EPISODE RESET ──────────────────────────────────────────────
        if terminated or truncated:
            episode_duration = time.time() - episode_start
            episode_stats["duration_seconds"] = episode_duration
            logger.log_episode_end(episode_stats)

            print(f"Episode {logger.current_episode} ended | "
                  f"Reward: {episode_stats['total_reward']:.1f} | "
                  f"Goals: {len(reward_fn.goal_tracker.completed)} | "
                  f"Duration: {episode_duration:.0f}s")

            obs, info = env.reset()
            memory = None
            task_history = []
            reward_fn.reset_episode()
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
            }

        # ── CHECKPOINT ─────────────────────────────────────────────────
        if step % checkpoint_interval == 0 and step > start_step:
            logger.save_checkpoint(step)

            # Save model
            step_path = f"models/checkpoints/step_{step}.pt"
            torch.save(policy.state_dict(), step_path)

            # Save latest (for resume)
            torch.save(policy.state_dict(), "models/checkpoints/latest.pt")
            with open("models/checkpoints/latest_meta.json", "w") as f:
                json.dump({"step": step, "episode": logger.current_episode}, f)

            # Save reward statistics
            reward_stats = reward_fn.get_statistics()
            with open(f"logs/{logger.run_name}/reward_statistics_{step}.json", "w") as f:
                json.dump(reward_stats, f, indent=2)

            print(f"[CHECKPOINT] Step {step}/{total_steps} | "
                  f"GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB | "
                  f"Goals: {len(reward_fn.goal_tracker.completed)}/{len(GOAL_SEQUENCE)}")

        # ── PROGRESS PRINT ─────────────────────────────────────────────
        if step % 1000 == 0 and step > start_step:
            print(f"Step {step} | Goal: {reward_fn.get_current_goal()} | "
                  f"GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # ── FINALIZE ───────────────────────────────────────────────────────
    env.close()
    logger.finalize()

    reward_stats = reward_fn.get_statistics()
    with open(f"logs/{logger.run_name}/reward_statistics_final.json", "w") as f:
        json.dump(reward_stats, f, indent=2)

    torch.save(policy.state_dict(), "models/checkpoints/final_model.pt")

    print("\nTraining complete!")
    print(f"Goals completed: {len(reward_fn.goal_tracker.completed)}/{len(GOAL_SEQUENCE)}")
    print(f"Final model: models/checkpoints/final_model.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_runpod.yaml")
    args = parser.parse_args()
    run_training(args.config)