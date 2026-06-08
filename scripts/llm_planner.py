"""
LLM planner using OpenAI GPT-4o-mini.
Uses correct MineStudio observation field names.
"""

import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def get_next_task(info: dict, history: list = None) -> dict:
    """
    Uses GPT-4o-mini to plan the next task.
    """
    # Extract readable inventory
    inv = info.get("inventory", {})
    inv_summary = {
        s["type"]: s["quantity"]
        for s in inv.values()
        if s["type"] != "none"
    }

    # Player position
    pos = info.get("player_pos", {})

    history_text = ""
    if history:
        history_text = f"\nRecent tasks: {', '.join(history[-5:])}"

    prompt = f"""You are an AI planner for a Minecraft agent.
Current agent state:
- Inventory: {inv_summary}
- Health: {info.get('health', 20)}/20
- Food: {info.get('food_level', 20)}/20
- Position: x={pos.get('x',0):.0f} y={pos.get('y',0):.0f} z={pos.get('z',0):.0f}
- Biome: {info.get('location_stats', {}).get('biome_id', 'unknown')}
- Recently mined: {dict(info.get('mine_block', {}))}
- Recently crafted: {dict(info.get('craft_item', {}))}
{history_text}

Goals by priority:
1. Survival (health > 10, food > 10) — eat food, avoid danger
2. Farming — craft hoe, find water, till soil, plant seeds, harvest crops
3. Animal husbandry — find animals, build fence pen, breed them
4. Construction — gather wood/stone, build 5x5 house, place bed, sleep

Reply ONLY with JSON (no markdown):
{{
  "task": "task name",
  "action": "specific action to take right now",
  "reasoning": "why this is the priority",
  "priority": 1
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)
