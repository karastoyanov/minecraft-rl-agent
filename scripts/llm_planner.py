"""
Improved LLM planner with:
- Tech tree awareness
- Longer context window
- Structured output with subtasks
- Fallback on API error
"""

import os
import json
import time
from openai import OpenAI



# Fallback task sequence if API fails
FALLBACK_SEQUENCE = [
    "collect_wood", "craft_tools", "collect_stone",
    "craft_stone_pickaxe", "collect_iron", "smelt_iron",
    "craft_iron_pickaxe", "craft_hoe", "find_flat_land",
    "till_soil", "plant_seeds", "build_shelter", "place_bed"
]
_fallback_index = 0


def get_next_task(info: dict, history: list = None,
                  tech_progress: dict = None) -> dict:
    """
    Uses GPT-4o-mini to plan the next task.
    Falls back to hardcoded sequence on API error.
    """
    
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    global _fallback_index

    # Extract state
    inv = info.get("inventory", {})
    inv_summary = {
        s["type"]: s["quantity"]
        for s in inv.values()
        if s["type"] != "none"
    }

    pos = info.get("player_pos", {})
    health = info.get("health", 20)
    food = info.get("food_level", 20)
    mined = dict(info.get("mine_block", {}))
    crafted = dict(info.get("craft_item", {}))
    location = info.get("location_stats", {})
    biome = location.get("biome_id", "unknown")
    is_raining = location.get("is_raining", False)
    light_level = location.get("light_level", 15)

    history_text = ""
    if history:
        history_text = f"Recent tasks (last 5): {', '.join(history[-5:])}"

    tech_text = ""
    if tech_progress:
        achieved = [k for k, v in tech_progress.items() if v]
        not_achieved = [k for k, v in tech_progress.items() if not v]
        tech_text = f"""
Tech tree achieved: {', '.join(achieved) if achieved else 'none'}
Next milestones needed: {', '.join(not_achieved[:5])}"""

    # Danger assessment
    danger = ""
    if health < 10:
        danger = "URGENT: Health is low! Find food or shelter immediately."
    elif food < 6:
        danger = "WARNING: Food is low! Find and eat food."
    elif light_level < 8:
        danger = "WARNING: Low light - mobs may spawn. Find/make shelter or torches."
    elif is_raining:
        danger = "NOTE: It is raining - visibility reduced."

    prompt = f"""You are an expert Minecraft AI planner. Guide an RL agent step by step.

CURRENT STATE:
- Health: {health}/20, Food: {food}/20
- Position: x={pos.get('x',0):.0f} y={pos.get('y',0):.0f} z={pos.get('z',0):.0f}
- Biome: {biome}, Light level: {light_level}
- Inventory: {json.dumps(inv_summary, indent=2) if inv_summary else 'empty'}
- Just mined: {mined if mined else 'nothing'}
- Just crafted: {crafted if crafted else 'nothing'}
{tech_text}
{history_text}
{danger}

LONG-TERM GOALS (in order):
1. Survive (health>10, food>10, shelter at night)
2. Farm (craft hoe, till soil near water, plant wheat/carrots, harvest)
3. Animal pen (find 2+ cows/pigs, build fenced enclosure, breed them)
4. House with bed (5x5 minimum, walls, roof, door, place bed, sleep)

Reply ONLY with this JSON (no markdown, no explanation):
{{
  "task": "short task name",
  "action": "single specific action to do right now",
  "reasoning": "one sentence why",
  "priority": 1,
  "subtasks": ["step1", "step2", "step3"]
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            response_format={"type": "json_object"},
            timeout=10.0   # don't block training for too long
        )
        result = json.loads(response.choices[0].message.content)
        _fallback_index = 0  # reset fallback on success
        return result

    except Exception as e:
        # On any error, use fallback sequence
        task = FALLBACK_SEQUENCE[_fallback_index % len(FALLBACK_SEQUENCE)]
        _fallback_index += 1
        return {
            "task": task,
            "action": task.replace("_", " "),
            "reasoning": f"API error fallback: {str(e)[:50]}",
            "priority": 1,
            "subtasks": []
        }
