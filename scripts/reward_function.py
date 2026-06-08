"""
Reward function for Minecraft agent.
Uses correct MineStudio field names from info dict.
"""

import yaml
from collections import defaultdict


class MinecraftRewardFunction:

    def __init__(self, config_path: str = "configs/rewards.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.reward_history = defaultdict(list)
        self.total_by_category = defaultdict(float)
        self.event_counts = defaultdict(int)
        self.idle_counter = 0
        self.prev_inventory = {}

        self.achievements = {
            "farm_created": False,
            "animal_pen_built": False,
            "house_walls_up": False,
            "bed_placed": False,
            "slept_in_bed": False,
        }

    def _get_inventory_items(self, info: dict) -> dict:
        """Extract inventory as {item_type: quantity} dict."""
        inv = info.get("inventory", {})
        result = defaultdict(int)
        for slot in inv.values():
            if slot["type"] != "none":
                result[slot["type"]] += slot["quantity"]
        return dict(result)

    def _get_mined_blocks(self, info: dict) -> dict:
        """Extract mined blocks this step."""
        return dict(info.get("mine_block", {}))

    def _get_crafted_items(self, info: dict) -> dict:
        """Extract crafted items this step."""
        return dict(info.get("craft_item", {}))

    def _get_picked_up(self, info: dict) -> dict:
        """Extract picked up items this step."""
        return dict(info.get("pickup", {}))

    def compute(self, obs: dict, action: dict, next_obs: dict,
                info: dict) -> tuple:
        """
        Compute reward for a single step.
        Returns: (total_reward, components_dict)
        """
        components = {}
        cfg = self.cfg

        # -- SURVIVAL --------------------------------------------------
        components["survival_alive"] = cfg["survival"]["stay_alive_per_step"]

        health = info.get("health", 20)
        food = info.get("food_level", 20)  # correct field name

        if health < 5:
            components["survival_low_health"] = cfg["survival"]["health_below_5_penalty"]
        if food <= 0:
            components["survival_starve"] = cfg["survival"]["starve_penalty"]

        # -- FARMING ---------------------------------------------------
        mined = self._get_mined_blocks(info)
        crafted = self._get_crafted_items(info)
        picked_up = self._get_picked_up(info)
        inventory = self._get_inventory_items(info)

        # Detect wheat/carrot/potato harvesting via pickup
        for crop, reward_key in [
            ("wheat", "harvest_wheat"),
            ("carrot", "harvest_carrot"),
            ("potato", "harvest_potato"),
        ]:
            if crop in picked_up and picked_up[crop] > 0:
                components[f"farm_harvest_{crop}"] = (
                    picked_up[crop] * cfg["farming"][reward_key]
                )
                self.event_counts[f"{crop}_harvested"] += picked_up[crop]

        # Seeds planted — detect farmland via mine_block or inventory change
        if "wheat_seeds" in picked_up:
            self.event_counts["seeds_in_inventory"] += picked_up["wheat_seeds"]

        # -- ANIMALS ---------------------------------------------------
        # Detect breeding via baby mob spawns (kill_entity inverse)
        # Approximate: if cow/pig/sheep in nearby mobs increases
        for animal in ["cow", "pig", "sheep", "chicken"]:
            if animal in picked_up:
                # picked up meat = killed animal, not breeding
                pass

        # -- CONSTRUCTION ----------------------------------------------
        # Detect blocks placed by comparing inventory decreases
        current_inv = inventory
        blocks_placed = 0
        for item, prev_qty in self.prev_inventory.items():
            curr_qty = current_inv.get(item, 0)
            if curr_qty < prev_qty:
                # inventory decreased = likely placed blocks
                blocks_placed += prev_qty - curr_qty

        if blocks_placed > 0:
            components["build_blocks"] = (
                blocks_placed * cfg["construction"]["place_block"]
            )
            self.event_counts["blocks_placed"] += blocks_placed

        # Detect bed placement
        if "bed" in self.prev_inventory and "bed" not in current_inv:
            if not self.achievements["bed_placed"]:
                components["build_bed"] = cfg["construction"]["place_bed"]
                self.achievements["bed_placed"] = True

        # Detect sleep via custom stats
        custom = info.get("custom", {})
        time_since_rest = custom.get("time_since_rest", 999)
        if time_since_rest < 5 and not self.achievements["slept_in_bed"]:
            components["build_sleep"] = cfg["construction"]["sleep_through_night"]
            self.achievements["slept_in_bed"] = True

        self.prev_inventory = current_inv

        # -- TOOL CRAFTING BONUS ---------------------------------------
        for tool, bonus in [
            ("wooden_pickaxe", 2.0),
            ("stone_pickaxe", 5.0),
            ("iron_pickaxe", 10.0),
            ("wooden_axe", 1.0),
            ("wooden_hoe", 3.0),   # hoe = farming intent
        ]:
            if tool in crafted and crafted[tool] > 0:
                components[f"craft_{tool}"] = bonus
                self.event_counts[f"crafted_{tool}"] += 1

        # -- PENALTIES -------------------------------------------------
        if action.get("type") == "idle" or action.get("noop", False):
            self.idle_counter += 1
            if self.idle_counter >= 100:
                components["penalty_idle"] = cfg["penalties"]["idle_steps_per_5s"]
                self.idle_counter = 0
        else:
            self.idle_counter = 0

        # -- FINALIZE --------------------------------------------------
        total_reward = sum(components.values())

        for k, v in components.items():
            self.reward_history[k].append(v)
            self.total_by_category[k] += v

        return total_reward, components

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
        stats["achievements"] = dict(self.achievements)
        return stats
