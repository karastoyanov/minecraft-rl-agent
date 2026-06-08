"""
Local training pipeline.
CPU mode: slow but sufficient for testing and debugging logic.
For full training use train_cloud.py with GPU.
"""

import yaml
import json
import torch
from pathlib import Path
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback
from minestudio.models import VPTPolicy

from reward_function import MinecraftRewardFunction
from data_logger import AgentDataLogger
from llm_planner import get_next_task


def run_training(config_path: str = "configs/experiment.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print(f"Starting: {config['experiment']['name']}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"LLM planner: OpenAI {config['llm']['model']}")
    print("=" * 60)

    logger = AgentDataLogger(config)
    reward_fn = MinecraftRewardFunction()

    env = MinecraftSim(
        obs_size=tuple(config["environment"]["obs_size"]),
        callbacks=[
            RecordCallback(
                record_path=f"logs/{logger.run_name}/videos",
                fps=20,
                frame_type="pov"
            )
        ]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = VPTPolicy.from_pretrained(
        "CraftJarvis/MineStudio_VPT.rl_from_early_game_2x"
    ).to(device)

    param_count = sum(p.numel() for p in policy.parameters()) / 1e6
    print(f"\nVPT model loaded on: {device} ({param_count:.1f}M parameters)\n")

    total_steps = config["training"]["total_steps"]
    obs, info = env.reset()
    memory = None
    task_history = []
    current_task = None

    episode_stats = {
        "total_reward": 0,
        "tasks_completed": [],
        "crops_harvested": 0,
        "animals_bred": 0,
        "blocks_placed": 0,
        "bed_placed": False,
        "tools_crafted": 0,
    }

    print("Training started...\n")

    for step in range(total_steps):

        # GPT-4o plans a new task every 200 steps
        if step % 200 == 0:
            planned = get_next_task(info, task_history)
            current_task = planned.get("task", "explore")
            task_history.append(current_task)
            if len(task_history) > 20:
                task_history = task_history[-20:]

        # VPT executes the action
        with torch.no_grad():
            action, memory = policy.get_action(obs, memory, input_shape='*')

        next_obs, _, terminated, truncated, info = env.step(action)
        info["current_task"] = current_task

        reward, reward_components = reward_fn.compute(obs, action, next_obs, info)

        episode_stats["total_reward"] += reward
        episode_stats["crops_harvested"] += sum(dict(info.get("pickup", {})).get(c, 0)
                                         for c in ["wheat", "carrot", "potato"])
        episode_stats["animals_bred"] += info.get("animals_bred", 0)
        episode_stats["blocks_placed"] += reward_components.get("build_blocks", 0) / 0.05
        episode_stats["tools_crafted"] += len([k for k in reward_components if "craft_" in k])

        if reward_components.get("build_bed", 0) > 0:
            episode_stats["bed_placed"] = True

        logger.log_step(obs, action, reward, reward_components, info,
                        terminated or truncated)
        obs = next_obs

        # End of episode
        if terminated or truncated:
            logger.log_episode_end(episode_stats)
            obs, info = env.reset()
            memory = None
            task_history = []
            episode_stats = {
                "total_reward": 0,
                "tasks_completed": [],
                "crops_harvested": 0,
                "animals_bred": 0,
                "blocks_placed": 0,
                "bed_placed": False,
                "tools_crafted": 0,
            }

        # Checkpoint
        if step % config["training"]["checkpoint_interval"] == 0 and step > 0:
            logger.save_checkpoint(step)
            print(f"Step {step}/{total_steps}")

    env.close()
    logger.finalize()

    # Save final reward statistics
    reward_stats = reward_fn.get_statistics()
    with open(f"logs/{logger.run_name}/reward_statistics.json", "w") as f:
        json.dump(reward_stats, f, indent=2)

    print("\nTraining complete!")


if __name__ == "__main__":
    run_training()
