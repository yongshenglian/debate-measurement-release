import logging
from tqdm import tqdm
import asyncio
from asyncio import Semaphore
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from dotenv import load_dotenv
from typing import List, Type, Dict, Any, Tuple, Union, Optional, Callable, Literal
import os
import shutil
import json

from src.agents.base_agent import AgentConfig
from src.agents.debate.basic_debate_agent import BasicDebateAgent
from src.llm_api import PromptConfig, LLMConfig
from src.environments.debate.utils import (
    get_question_data,
    load_and_prepare_data,
    save_results,
    log_to_wandb,
    log_single_debate_result,
)
from src.environments.debate.adts import (
    DebateResult,
    DebateRound,
    DebateProcedureResult,
)
from src.environments.debate.checkpointing import (
    ensure_checkpoint_dir,
    checkpoint_exists,
    save_config,
    save_checkpoint,
    load_checkpoint,
)

# Configure loggers
loggers = ["LiteLLM Proxy", "LiteLLM Router", "LiteLLM", "httpx"]
for logger_name in loggers:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL + 1)


def create_agent_configs(agent_types):
    """Create a list of agent configs based on agent types and their counts."""
    agent_configs = []
    for agent_type in agent_types:
        for i in range(agent_type.count):
            agent_configs.append(
                {
                    "name": f"{agent_type.name}_{i}",
                    "prompts": agent_type.prompts,
                    "llm_config": agent_type.llm_config,
                }
            )
    return agent_configs


def create_agents(agent_types, num_agents, task_name, debug):
    """Create and initialize agents based on agent types."""
    agent_configs = create_agent_configs(agent_types)
    agents = [
        BasicDebateAgent(
            config=AgentConfig(
                prompt_config=PromptConfig(
                    system_prompt=agent_configs[i]["prompts"]["system_prompt"],
                    partials={
                        **(agent_configs[i]["prompts"]["partials"] or {}),
                        **(agent_configs[i]["prompts"]["additional_partials"] or {}),
                    },
                ),
                llm_config=LLMConfig.from_hydra_config(agent_configs[i]["llm_config"]),
                name=agent_configs[i]["name"],
            ),
            num_agents=num_agents,
            domain=task_name,
            debug=debug,
        )
        for i in range(num_agents)
    ]

    # Check if number of unique agent names matches expected num_agents
    unique_agent_names = set(agent.config.name for agent in agents)
    if len(unique_agent_names) != num_agents:
        raise ValueError(
            f"Expected {num_agents} unique agent names, but got {len(unique_agent_names)}. "
            f"Please check agent configurations for duplicate names: {unique_agent_names}"
        )

    # Build and initialize agents
    for agent in agents:
        agent.build()
        agent.reset()

    return agents


async def run_debate_procedure(
    agents: List[BasicDebateAgent],
    question: str,
    question_prompt: Optional[str],
    num_rounds: int,
    summarize: bool,
    task_name: Literal["mmlu", "math", "commonsense_qa", "gsm8k"],
    semaphore: Optional[Semaphore] = None,
) -> DebateProcedureResult:
    """Run a complete debate round for a given question."""
    # placeholder to store the debate result
    debate_procedure_result = DebateProcedureResult()

    # Set instruction for all agents
    for agent in agents:
        agent.set_instruction(question)

    # Debate rounds
    for debate in range(num_rounds + 1):
        # Create a new round object
        current_round = DebateRound(round_number=debate)

        if debate != 0:
            # Get responses with semaphore control
            async def add_discussion_with_semaphore(agent):
                async with semaphore:
                    return await agent.add_discussion_with_other_agents_in_context(
                        [a.answer_history[-1] for a in agents if a != agent],
                        summarize=summarize,
                        additional_context=question_prompt
                        if task_name == "math" or task_name == "gsm8k"
                        else None,
                    )

            # Create and run all discussion tasks
            discussion_tasks = [
                add_discussion_with_semaphore(agent) for agent in agents
            ]
            await asyncio.gather(*discussion_tasks)

        # Generate answers with semaphore control
        async def generate_answer_with_semaphore(agent):
            async with semaphore:
                return await agent.generate_answer()

        answer_tasks = [generate_answer_with_semaphore(agent) for agent in agents]
        await asyncio.gather(*answer_tasks)

        # Record the result here
        for agent in agents:
            # current_round.add_query(agent.name, agent.latest_query()) # can add query here
            current_round.add_response(agent.name, agent.latest_response())

        debate_procedure_result.add_round(current_round)

    return debate_procedure_result


async def process_question(
    idx: int,
    question: str,
    answer: Union[str, int],
    question_prompt: Optional[str],
    agent_types,
    task_name: Literal["mmlu", "math", "commonsense_qa", "gsm8k"],
    num_rounds: int,
    summarize: bool,
    debug: bool = False,
    semaphore: Optional[Semaphore] = None,
    log_single_results: bool = True,
) -> DebateResult:
    # Create new agent instances for this specific question
    num_agents = sum(agent_type.count for agent_type in agent_types)
    agents = create_agents(agent_types, num_agents, task_name, debug)

    # Run debate rounds
    debate_procedure_result = await run_debate_procedure(
        agents, question, question_prompt, num_rounds, summarize, task_name, semaphore
    )

    # Update with question-specific metadata
    debate_result = DebateResult(
        question_id=idx,
        question=question,
        correct_answer=str(answer),
        question_prompt=question_prompt,
        debate_session=debate_procedure_result,
    )

    # Log this debate result to wandb if enabled
    if log_single_results:
        log_single_debate_result(debate_result, task_name, num_rounds)

    return debate_result


async def run_experiment(cfg: DictConfig):
    # Display config parameters
    # print("\nExperiment Configuration:")
    # print(OmegaConf.to_yaml(OmegaConf.to_container(cfg, resolve=True)))
    # print("\n" + "=" * 50 + "\n")

    # Checkpointing configuration - extract these first
    checkpoint_dir = os.path.join(cfg.exp_dir, "checkpoint")
    resume = cfg.checkpoint.get("resume", False)

    # Ensure checkpoint directory exists
    ensure_checkpoint_dir(checkpoint_dir)

    # If resuming, load the checkpoint and use its config
    if resume and checkpoint_exists(checkpoint_dir):
        print(f"\nResuming experiment from checkpoint: {checkpoint_dir}")
        try:
            simulated_debates, start_idx, loaded_config = load_checkpoint(
                checkpoint_dir
            )

            print(f"Resumed from question index {start_idx}")
            print(f"Loaded {len(simulated_debates)} previously completed debates")

            # Use the loaded config for the experiment parameters
            # This ensures we're using the same parameters as the original run
            cfg = loaded_config
            print("\nUsing loaded configuration from checkpoint:")
            print(OmegaConf.to_yaml(OmegaConf.to_container(cfg, resolve=True)))
            print("\n" + "=" * 50 + "\n")

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            user_response = input("Start from beginning instead? (yes/no): ").lower()
            if user_response not in ["yes", "y"]:
                print("Experiment cancelled.")
                return
            # Reset to start from beginning
            start_idx = 0
            simulated_debates = []
            resume = False
    else:
        # Not resuming, start fresh
        start_idx = 0
        simulated_debates = []

        # Save initial config
        save_config(checkpoint_dir, cfg)

    task_name = cfg.task.name
    assert task_name in [
        "mmlu",
        "math",
        "commonsense_qa",
        "gsm8k",
    ], "Task domain must be mmlu, math, commonsense_qa, or gsm8k"

    # Get experiment parameters from config
    num_rounds = cfg.experiment.num_rounds
    num_questions = cfg.experiment.num_questions
    summarize = cfg.experiment.summarize
    output_dir = cfg.exp_dir
    log_single_results = cfg.experiment.get(
        "log_single_results", True
    )  # Default to True if not specified

    # Re-extract checkpointing parameters from the config (might have changed if loaded)
    checkpoint_frequency = cfg.checkpoint.get("frequency", 10)

    # Calculate total number of agents from agent types
    num_agents = sum(agent_type.count for agent_type in cfg.agent_types)
    print(f"Total number of agents: {num_agents}")

    # Load and prepare data using shared utility
    questions = load_and_prepare_data(
        task_name=task_name,
        data_path=cfg.task.data_path,
        num_questions=num_questions,
        seed=cfg.seed,
    )

    # Assert that enough questions are loaded
    assert (
        len(questions) >= num_questions
    ), f"Loaded only {len(questions)} questions, but num_questions is set to {num_questions}."

    # Cost checking logic
    if cfg.cost_check:
        print("\nRunning cost check with one question...")

        # Create agents for cost check
        agents = create_agents(cfg.agent_types, num_agents, task_name, cfg.debug)

        # Get first question
        question, answer, question_prompt = get_question_data(questions[0], task_name)

        # Run one complete debate round
        await run_debate_procedure(
            agents,
            question,
            question_prompt,
            num_rounds,
            summarize,
            task_name,
            semaphore=asyncio.Semaphore(
                1
            ),  # use a sequential semaphore for cost estimation
        )

        # Calculate total cost
        total_cost = sum(agent.get_total_cost() for agent in agents)
        estimated_total_cost = total_cost * num_questions

        print(f"\nCost check results for {task_name} task:")
        print(f"Cost for one question: ${total_cost:.4f}")
        print(
            f"Estimated total cost for {num_questions} questions: ${estimated_total_cost:.4f}"
        )

        # Ask for confirmation
        while True:
            response = input(
                "\nDo you want to proceed with the experiment? (yes/no): "
            ).lower()
            if response in ["yes", "y"]:
                break
            elif response in ["no", "n"]:
                print("Experiment cancelled.")
                return
            else:
                print("Please enter 'yes' or 'no'.")

    # Get processing mode, default to concurrent
    processing_mode = cfg.experiment.get("processing_mode", "concurrent")
    # Set key variables based on processing mode
    if processing_mode == "sequential":
        max_concurrent = 1
        semaphore_permits = 1
        concurrent_questions = 1
        print("\nProcessing Strategy: SEQUENTIAL (one at a time)")
    else:  # concurrent mode
        max_concurrent = cfg.experiment.max_concurrent_calls
        semaphore_permits = num_agents
        concurrent_questions = max(1, max_concurrent // semaphore_permits)
        print(
            f"\nProcessing Strategy: CONCURRENT ({concurrent_questions} questions at a time)"
        )

    remaining_questions = num_questions - start_idx
    questions_pbar = tqdm(
        total=remaining_questions, desc="Questions processed", position=0
    )

    # Process questions in batches
    for batch_start in range(start_idx, num_questions, concurrent_questions):
        batch_end = min(batch_start + concurrent_questions, num_questions)
        batch = range(batch_start, batch_end)

        batch_tasks = []
        for idx in batch:
            question, answer, question_prompt = get_question_data(
                questions[idx], task_name
            )
            question_semaphore = asyncio.Semaphore(semaphore_permits)

            task = process_question(
                idx=idx,
                question=question,
                answer=answer,
                question_prompt=question_prompt,
                agent_types=cfg.agent_types,
                task_name=task_name,
                num_rounds=num_rounds,
                summarize=summarize,
                debug=cfg.debug,
                semaphore=question_semaphore,
                log_single_results=log_single_results,
            )
            batch_tasks.append(task)

        batch_results = await asyncio.gather(*batch_tasks)
        simulated_debates.extend(batch_results)
        questions_pbar.update(len(batch_results))

        # Save checkpoint after processing this batch
        # Find the last index in this batch that should trigger a checkpoint
        last_checkpoint_idx = None
        for checkpoint_idx in range(batch_start, batch_end):
            if (
                checkpoint_idx > 0  # Skip the first question (index 0)
                and (
                    checkpoint_idx % checkpoint_frequency == 0
                    or checkpoint_idx == num_questions - 1
                )
            ):
                last_checkpoint_idx = checkpoint_idx

        # Save at the latest checkpoint opportunity in this batch
        if last_checkpoint_idx is not None:
            next_question_idx = last_checkpoint_idx + 1
            save_checkpoint(
                simulated_debates,
                next_question_idx,
                cfg,
                checkpoint_dir,
            )
            print(f"Checkpoint saved at question {next_question_idx}/{num_questions}")

    questions_pbar.close()

    # Save results and get file paths using shared utility
    output_file, performance_file = save_results(
        generated_results=simulated_debates,
        output_dir=output_dir,
        task_name=task_name,
        num_rounds=num_rounds,
        strict=cfg.get("strict", False),
    )

    # Log to wandb using shared utility
    log_to_wandb(
        generated_results=simulated_debates,
        task_name=task_name,
        num_rounds=num_rounds,
        output_file=output_file,
        performance_file=performance_file,
        strict=cfg.get("strict", False),
    )

    # Copy final results to checkpoint directory for completeness
    shutil.copy(output_file, os.path.join(checkpoint_dir, f"{task_name}_result.json"))
    shutil.copy(
        performance_file,
        os.path.join(checkpoint_dir, f"{task_name}_performance.json"),
    )

    print(f"\nExperiment completed and saved to checkpoint: {checkpoint_dir}")


load_dotenv()


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="basic_debate",
)
def main(cfg: DictConfig):
    # Check if resuming from checkpoint
    checkpoint_dir = os.path.join(cfg.exp_dir, "checkpoint")
    resume = cfg.checkpoint.get("resume", False)

    # Initialize wandb run ID from checkpoint if resuming
    wandb_run_id = None
    if resume and checkpoint_exists(checkpoint_dir):
        # Try to load wandb ID from checkpoint
        try:
            wandb_metadata_file = os.path.join(checkpoint_dir, "wandb_metadata.json")
            if os.path.exists(wandb_metadata_file):
                with open(wandb_metadata_file, "r") as f:
                    wandb_metadata = json.load(f)
                    wandb_run_id = wandb_metadata.get("run_id")
                    print(f"Resuming wandb run: {wandb_run_id}")
        except Exception as e:
            print(f"Error loading wandb metadata: {e}")

    # Create a run identifier for grouping (not used as name but as a tag)
    model_parts = []

    # Build model_count strings dynamically
    llm_keys = [key for key in cfg.keys() if key.startswith("llm")]
    llm_keys.sort()  # Sort to ensure consistent ordering

    for i, llm_key in enumerate(llm_keys):
        if i < len(cfg.agent_counts) and cfg.agent_counts[i] > 0:
            model_name = cfg[llm_key].language_models[0].model_name
            count = cfg.agent_counts[i]
            model_parts.append(f"{model_name}_{count}")

    # Join model parts with hyphens
    models_str = "-".join(model_parts)

    # Create experiment identifier (for tags, not name)
    experiment_id = (
        f"{cfg.task.name}_r{cfg.experiment.num_rounds}_{models_str}_s{cfg.seed}"
    )

    # Generate tags for wandb
    tags = [
        f"task-{cfg.task.name}",
        f"rounds-{cfg.experiment.num_rounds}",
        f"seed-{cfg.seed}",
        # f"exp-{experiment_id}",  # Add the full experiment identifier as a tag
        f"name-{cfg.experiment.name}",
    ]

    # Add model tags dynamically
    for i, llm_key in enumerate(llm_keys):
        if i < len(cfg.agent_counts) and cfg.agent_counts[i] > 0:
            model_name = cfg[llm_key].language_models[0].model_name
            count = cfg.agent_counts[i]
            tags.append(f"{model_name}-{count}")

    # Initialize wandb with the run_id if resuming
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.team,
        mode=cfg.wandb.mode,
        config=dict(cfg),
        id=wandb_run_id,  # Use the existing run ID if resuming
        resume="must" if wandb_run_id else None,  # Force resume if we have an ID
        tags=tags,
    )

    # Save the wandb run ID to checkpoint for future resuming
    if not resume or not wandb_run_id:
        # Only save if we're not resuming or if we didn't find a previous ID
        wandb_metadata_file = os.path.join(checkpoint_dir, "wandb_metadata.json")
        os.makedirs(os.path.dirname(wandb_metadata_file), exist_ok=True)
        with open(wandb_metadata_file, "w") as f:
            json.dump({"run_id": wandb.run.id}, f)

    asyncio.run(run_experiment(cfg))

    wandb.finish()


if __name__ == "__main__":
    main()
