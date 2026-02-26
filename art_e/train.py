import art
import asyncio
import os
from dotenv import load_dotenv
from typing import List
from art_e.rollout import rollout
from art_e.data.query_iterators import load_synthetic_queries
from art_e.data.types_enron import SyntheticQuery
from art_e.data.local_email_db import generate_database
from art.utils import iterate_dataset
from art_e.project_types import ProjectPolicyConfig, TrainingConfig

load_dotenv()

# Best model config from ART-E (agent_008)
agent_008 = art.TrainableModel(
    name="email-agent-008",
    project="email_agent",
    base_model="Qwen/Qwen2.5-14B-Instruct",
    config=ProjectPolicyConfig(
        max_turns=10,
        log_to_openpipe=False,
        use_tools=True,
        training_config=TrainingConfig(
            trajectories_per_group=4,
            groups_per_step=12,
            learning_rate=1.2e-5,
            eval_steps=30,
            val_set_size=100,
            training_dataset_size=2510,  # Full Vince Kaminski train set
            num_epochs=3,
        ),
    ),
)


async def run_training(model: art.TrainableModel):
    # Step 1: Generate SQLite email database (downloads from HF if needed)
    generate_database()

    assert isinstance(model.config, ProjectPolicyConfig)
    if model.config.training_config is None:
        raise ValueError("Training config is not set")

    # Step 2: Initialize local API (starts vLLM server)
    api = art.LocalAPI()
    await model.register(api)

    # Step 3: Load training data from local Arrow dataset
    print("Loading training data...")
    train_scenarios: List[SyntheticQuery] = load_synthetic_queries(
        split="train", limit=model.config.training_config.training_dataset_size
    )
    print(f"Training data size: {len(train_scenarios)}")

    # Step 4: Training loop (GRPO)
    train_iterator = iterate_dataset(
        train_scenarios,
        groups_per_step=model.config.training_config.groups_per_step,
        num_epochs=model.config.training_config.num_epochs,
        initial_step=await model.get_step(),
    )

    for batch, epoch, global_step, epoch_step in train_iterator:
        print(f"\n--- Step {global_step} (Epoch {epoch}, Step {epoch_step}) ---")

        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    (
                        rollout(model, scenario)
                        for _ in range(
                            model.config.training_config.trajectories_per_group
                        )
                    )
                )
                for scenario in batch
            )
        )

        await model.train(
            groups,
            config=art.TrainConfig(
                learning_rate=model.config.training_config.learning_rate
            ),
        )

    print("Training finished.")


if __name__ == "__main__":
    asyncio.run(run_training(agent_008))
