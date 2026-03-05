import art
import asyncio
from typing import List

from art.dev import EngineArgs, InternalModelConfig, TrainerArgs
from dotenv import load_dotenv

from art.utils import iterate_dataset
from art_e.data.local_email_db import generate_database
from art_e.data.query_iterators import load_synthetic_queries
from art_e.data.types_enron import SyntheticQuery
from art_e.evaluate.benchmark import benchmark_model
from art_e.project_types import ProjectPolicyConfig, TrainingConfig
from art_e.rollout import rollout

load_dotenv()


# Qwen3-14B without thinking
agent_qwen3_14b = art.TrainableModel(
    name="email-agent-qwen3-14b",
    project="email_agent",
    base_model="Qwen/Qwen3-14B",
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
            training_dataset_size=2510,
            num_epochs=3,
            keep_top_k=3,
            max_steps=300,
        ),
    ),
    _internal_config=InternalModelConfig(
        engine_args=EngineArgs(max_model_len=32768, gpu_memory_utilization=0.7),
        trainer_args=TrainerArgs(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
        ),
    ),
)

# Qwen3-14B with thinking enabled
agent_qwen3_14b_thinking = art.TrainableModel(
    name="email-agent-qwen3-14b-thinking",
    project="email_agent",
    base_model="Qwen/Qwen3-14B",
    config=ProjectPolicyConfig(
        max_turns=10,
        max_tokens=4096,
        log_to_openpipe=False,
        use_tools=True,
        enable_thinking=True,
        training_config=TrainingConfig(
            trajectories_per_group=4,
            groups_per_step=12,
            learning_rate=1.2e-5,
            eval_steps=30,
            val_set_size=100,
            training_dataset_size=2510,
            num_epochs=3,
            keep_top_k=3,
            max_steps=300,
        ),
    ),
    _internal_config=InternalModelConfig(
        engine_args=EngineArgs(max_model_len=32768, gpu_memory_utilization=0.7),
        trainer_args=TrainerArgs(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
        ),
    ),
)

# Qwen2.5-14B-Instruct (resume from email-agent-008)
agent_qwen25_14b = art.TrainableModel(
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
            training_dataset_size=2510,
            num_epochs=3,
            keep_top_k=3,
            max_steps=300,
        ),
    ),
    _internal_config=InternalModelConfig(
        engine_args=EngineArgs(max_model_len=32768, gpu_memory_utilization=0.6),
        trainer_args=TrainerArgs(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
        ),
    ),
)

MODELS = {
    "qwen3-14b": agent_qwen3_14b,
    "qwen3-14b-thinking": agent_qwen3_14b_thinking,
    "qwen25-14b": agent_qwen25_14b,
}


async def run_training(model: art.TrainableModel):
    # Step 1: Generate SQLite email database (downloads from HF if needed)
    generate_database()

    assert isinstance(model.config, ProjectPolicyConfig)
    if model.config.training_config is None:
        raise ValueError("Training config is not set")

    tc = model.config.training_config

    # Step 2: Initialize local backend (starts vLLM server)
    from art.local import LocalBackend

    backend = LocalBackend()

    # Build OpenAI server config for thinking mode
    openai_config = None
    if isinstance(model.config, ProjectPolicyConfig) and model.config.enable_thinking:
        from art.dev.openai_server import OpenAIServerConfig

        openai_config = OpenAIServerConfig(
            server_args={
                "reasoning_parser": "qwen3",
            }
        )

    await model.register(backend, _openai_client_config=openai_config)

    # Step 3: Load training data from local Arrow dataset
    print("Loading training data...")
    train_scenarios: List[SyntheticQuery] = load_synthetic_queries(
        split="train", limit=tc.training_dataset_size
    )
    print(f"Training data size: {len(train_scenarios)}")

    # Step 4: Training loop (GRPO)
    train_iterator = iterate_dataset(
        train_scenarios,
        groups_per_step=tc.groups_per_step,
        num_epochs=tc.num_epochs,
        initial_step=await model.get_step(),
    )

    for dataset_batch in train_iterator:
        batch = dataset_batch.items
        epoch = dataset_batch.epoch
        global_step = dataset_batch.step
        epoch_step = dataset_batch.epoch_step
        if tc.max_steps is not None and global_step >= tc.max_steps:
            print(f"\nReached max_steps={tc.max_steps}, stopping training.")
            break
        print(f"\n--- Step {global_step} (Epoch {epoch}, Step {epoch_step}) ---")

        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    (rollout(model, scenario) for _ in range(tc.trajectories_per_group))
                )
                for scenario in batch
            )
        )

        await model.train(
            groups,
            config=art.TrainConfig(learning_rate=tc.learning_rate),
        )

        # Step 5: Periodic validation & checkpoint pruning
        if global_step > 0 and global_step % tc.eval_steps == 0:
            print(f"\n=== Validation at step {global_step} ===")
            avg_metrics = await benchmark_model(model, limit=tc.val_set_size)
            print(avg_metrics)
            await model.delete_checkpoints()

    # Final validation
    print("\n=== Final validation ===")
    avg_metrics = await benchmark_model(model, limit=tc.val_set_size)
    print(avg_metrics)
    await model.delete_checkpoints()

    print("Training finished.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="qwen3-14b",
        help="Model config to train",
    )
    args = parser.parse_args()
    asyncio.run(run_training(MODELS[args.model]))
