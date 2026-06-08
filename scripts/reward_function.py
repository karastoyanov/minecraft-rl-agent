"""
Improved reward function with:
- Potential-based shaping (dense rewards)
- Progress tracking toward goals
- Correct MineStudio field names
- Normalized reward scale
"""

import yaml
import numpy as np
from collections import defaultdict


class MinecraftRewardFunction:

    def __init__(self, config_path: str = "configs/rewards.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.reward_history = defaultdict(list)
        self.total_by_category = defaultdict(float)
        self.event_counts = defaultdict(int)
        self.idle_counter = 0

        # State tracking for potential-based shaping
        self.prev_inventory = {}
        self.prev_health = 20.0
        self.prev_food = 20.0
        self.visited_positions = set()
        self.steps_since_new_area = 0

        # Progress tracking for goals
        self.tech_tree_progress = {
            "has_wood": False,
            "has_planks": False,
            "has_crafting_table": False,
            "has_wooden_pickaxe": False,
            "has_stone": False,
            "has_stone_pickaxe": False,
            "has_iron_ore": False,
            "has_iron_ingot": False,
            "has_iron_pickaxe": False,
            "has_hoe": False,
            "has_seeds": False,
            "farm_tilled": False,
            "crops_planted": False,
            "first_harvest": False,
            "has_fence": False,
            "pen_built": False,
            "has_bed": False,
            "bed_placed_in_shelter": False,
            "slept_through_night": False,
        }

        # One-time achievement bonuses (prevent double counting)
        self.achievements = set()

        # Statistics
        self.total_steps = 0
        self.episode_rewards = []

    def _get_inventory(self, info: dict) -> dict:
        """Extract {item: quantity} from info."""
        inv = info.get("inventory", {})
        result = defaultdict(int)
        for slot in inv.values():
            if slot["type"] != "none":
                result[slot["type"]] += slot["quantity"]
        return dict(result)

    def _get_position_cell(self, info: dict) -> tuple:
        """Get coarse position for exploration tracking."""
        pos = info.get("player_pos", {})
        # 16x16 chunk-level granularity
        return (
            int(pos.get("x", 0)) // 16,
            int(pos.get("z", 0)) // 16,
        )

    def _check_tech_tree(self, inventory: dict) -> dict:
        """Check tech tree progress and return newly achieved milestones."""
        newly_achieved = {}

        checks = {
            "has_wood": lambda i: i.get("log", 0) + i.get("oak_log", 0) > 0,
            "has_planks": lambda i: i.get("planks", 0) + i.get("oak_planks", 0) > 0,
            "has_crafting_table": lambda i: i.get("crafting_table", 0) > 0,
            "has_wooden_pickaxe": lambda i: i.get("wooden_pickaxe", 0) > 0,
            "has_stone": lambda i: i.get("cobblestone", 0) > 3,
            "has_stone_pickaxe": lambda i: i.get("stone_pickaxe", 0) > 0,
            "has_iron_ore": lambda i: i.get("iron_ore", 0) > 0,
            "has_iron_ingot": lambda i: i.get("iron_ingot", 0) > 0,
            "has_iron_pickaxe": lambda i: i.get("iron_pickaxe", 0) > 0,
            "has_hoe": lambda i: any("hoe" in k for k in i),
            "has_seeds": lambda i: i.get("wheat_seeds", 0) > 0 or i.get("seeds", 0) > 0,
            "has_fence": lambda i: i.get("fence", 0) > 4,
            "has_bed": lambda i: any("bed" in k for k in i),
        }

        for milestone, check_fn in checks.items():
            if not self.tech_tree_progress[milestone]:
                try:
                    if check_fn(inventory):
                        self.tech_tree_progress[milestone] = True
                        newly_achieved[milestone] = True
                except Exception:
                    pass

        return newly_achieved

    def compute(self, obs, action, next_obs, info: dict) -> tuple:
        """
        Compute reward for a single step.
        Returns: (total_reward, components_dict)
        """
        components = {}
        cfg = self.cfg
        self.total_steps += 1

        health = info.get("health", 20.0)
        food = info.get("food_level", 20)
        inventory = self._get_inventory(info)
        pos_cell = self._get_position_cell(info)

        # -- SURVIVAL --------------------------------------------------
        # Small constant reward for staying alive
        components["survival_alive"] = cfg["survival"]["stay_alive_per_step"]

        # Health change penalty (being hurt)
        health_delta = health - self.prev_health
        if health_delta < -1:
            components["survival_damage"] = health_delta * 0.5  # negative
            self.event_counts["times_damaged"] += 1

        # Critical health penalty
        if health < 5:
            components["survival_critical"] = cfg["survival"]["health_below_5_penalty"]

        # Food level reward shaping — encourage eating before starving
        if food < 8 and self.prev_food >= 8:
            components["survival_hunger_warning"] = -1.0
        if food <= 0:
            components["survival_starve"] = cfg["survival"]["starve_penalty"]

        # -- EXPLORATION -----------------------------------------------
        if pos_cell not in self.visited_positions:
            self.visited_positions.add(pos_cell)
            components["exploration_new_area"] = 0.3
            self.steps_since_new_area = 0
            self.event_counts["areas_explored"] += 1
        else:
            self.steps_since_new_area += 1

        # Anti-stuck penalty — penalize if not exploring for too long
        if self.steps_since_new_area > 500:
            components["exploration_stuck"] = -0.1

        # -- TECH TREE PROGRESS ----------------------------------------
        # One-time bonuses for reaching milestones (dense progress signal)
        tech_bonuses = {
            "has_wood": 2.0,
            "has_planks": 1.0,
            "has_crafting_table": 3.0,
            "has_wooden_pickaxe": 5.0,
            "has_stone": 2.0,
            "has_stone_pickaxe": 8.0,
            "has_iron_ore": 5.0,
            "has_iron_ingot": 8.0,
            "has_iron_pickaxe": 15.0,
            "has_hoe": 10.0,      # farming prerequisite
            "has_seeds": 5.0,
            "has_fence": 5.0,
            "has_bed": 20.0,
        }

        newly_achieved = self._check_tech_tree(inventory)
        for milestone, _ in newly_achieved.items():
            if milestone not in self.achievements:
                bonus = tech_bonuses.get(milestone, 1.0)
                components[f"tech_{milestone}"] = bonus
                self.achievements.add(milestone)
                self.event_counts[f"achieved_{milestone}"] += 1

        # -- FARMING ---------------------------------------------------
        picked_up = dict(info.get("pickup", {}))
        crafted = dict(info.get("craft_item", {}))
        mined = dict(info.get("mine_block", {}))

        # Crop harvesting
        for crop, reward_key in [
            ("wheat", "harvest_wheat"),
            ("carrot", "harvest_carrot"),
            ("potato", "harvest_potato"),
        ]:
            qty = picked_up.get(crop, 0)
            if qty > 0:
                components[f"farm_harvest_{crop}"] = qty * cfg["farming"][reward_key]
                self.event_counts[f"{crop}_harvested"] += qty
                if "first_harvest" not in self.achievements:
                    components["farm_first_harvest_bonus"] = 25.0
                    self.achievements.add("first_harvest")
                    self.tech_tree_progress["first_harvest"] = True

        # Seeds obtained
        seeds_picked = picked_up.get("wheat_seeds", 0) + picked_up.get("seeds", 0)
        if seeds_picked > 0 and not self.tech_tree_progress["crops_planted"]:
            components["farm_seeds_obtained"] = seeds_picked * 0.5

        # -- ANIMALS ---------------------------------------------------
        # Detect breeding via custom stats or entity kills
        # Baby animals spawned = positive signal
        if "animal_baby_spawned" in dict(info.get("custom", {})):
            components["animal_breed"] = cfg["animals"]["breed_animals"]
            self.event_counts["animals_bred"] += 1

        # -- CONSTRUCTION ----------------------------------------------
        # Track blocks placed via inventory decrease
        blocks_placed_this_step = 0
        building_blocks = [
            "planks", "oak_planks", "cobblestone", "stone",
            "oak_log", "log", "dirt", "glass", "fence",
            "oak_fence", "spruce_planks", "birch_planks"
        ]
        for block in building_blocks:
            prev_qty = self.prev_inventory.get(block, 0)
            curr_qty = inventory.get(block, 0)
            if curr_qty < prev_qty:
                blocks_placed_this_step += prev_qty - curr_qty

        if blocks_placed_this_step > 0:
            components["build_blocks"] = (
                blocks_placed_this_step * cfg["construction"]["place_block"]
            )
            self.event_counts["blocks_placed"] += blocks_placed_this_step

        # Bed placement detection
        had_bed = any("bed" in k for k in self.prev_inventory)
        has_bed = any("bed" in k for k in inventory)
        if had_bed and not has_bed and "bed_placed" not in self.achievements:
            components["build_bed"] = cfg["construction"]["place_bed"]
            self.achievements.add("bed_placed")
            self.tech_tree_progress["bed_placed_in_shelter"] = True

        # Sleep detection via time_since_rest resetting
        custom = dict(info.get("custom", {}))
        time_since_rest = custom.get("time_since_rest", 999)
        if time_since_rest < 10 and "slept_through_night" not in self.achievements:
            components["build_sleep"] = cfg["construction"]["sleep_through_night"]
            self.achievements.add("slept_through_night")
            self.tech_tree_progress["slept_through_night"] = True

        # -- TOOL CRAFTING BONUSES ------------------------------------
        tool_bonuses = {
            "wooden_pickaxe": 3.0,
            "stone_pickaxe": 6.0,
            "iron_pickaxe": 12.0,
            "wooden_axe": 1.5,
            "stone_axe": 3.0,
            "wooden_hoe": 8.0,   # hoe = farming intent
            "stone_hoe": 10.0,
            "furnace": 5.0,      # smelting capability
        }
        for tool, bonus in tool_bonuses.items():
            qty = crafted.get(tool, 0)
            if qty > 0 and f"crafted_{tool}" not in self.achievements:
                components[f"craft_{tool}"] = bonus
                self.achievements.add(f"crafted_{tool}")
                self.event_counts[f"crafted_{tool}"] += 1

        # -- IDLE PENALTY ---------------------------------------------
        # Detect near-zero movement as idle
        is_idle = (
            action.get("forward", 0) == 0 and
            action.get("back", 0) == 0 and
            action.get("left", 0) == 0 and
            action.get("right", 0) == 0 and
            action.get("attack", 0) == 0 and
            action.get("use", 0) == 0
        )
        if is_idle:
            self.idle_counter += 1
            if self.idle_counter >= 100:
                components["penalty_idle"] = cfg["penalties"]["idle_steps_per_5s"]
                self.idle_counter = 0
        else:
            self.idle_counter = max(0, self.idle_counter - 5)

        # -- UPDATE STATE ---------------------------------------------
        self.prev_health = health
        self.prev_food = food
        self.prev_inventory = inventory

        # -- NORMALIZE & CLIP -----------------------------------------
        total_reward = sum(components.values())
        # Clip to prevent extreme values destabilizing training
        total_reward = float(np.clip(total_reward, -10.0, 50.0))

        for k, v in components.items():
            self.reward_history[k].append(v)
            self.total_by_category[k] += v

        return total_reward, components

    def reset_episode(self):
        """Call at episode start to reset per-episode tracking."""
        self.prev_inventory = {}
        self.prev_health = 20.0
        self.prev_food = 20.0
        self.idle_counter = 0
        # Keep visited_positions and tech_tree_progress across episodes
        # (agent remembers what it has learned)

    def get_statistics(self) -> dict:
        """Return statistics for paper analysis."""
        stats = {}
        for category, values in self.reward_history.items():
            if values:
                stats[category] = {
                    "count": len([v for v in values if v != 0]),
                    "total": sum(values),
                    "mean": sum(values) / len(values),
                    "max": max(values),
                }
        stats["event_counts"] = dict(self.event_counts)
        stats["achievements"] = list(self.achievements)
        stats["tech_tree_progress"] = dict(self.tech_tree_progress)
        stats["total_steps"] = self.total_steps
        return stats
