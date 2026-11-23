# Talk Ain't Cheap

ArXiv link: coming soon!

## Table of Contents

- [Setup](#setup)
  - [Environment Setup](#environment-setup)
  - [API Keys](#api-keys)
  - [Dataset Setup](#dataset-setup)
- [Running Experiments](#running-experiments)

## Setup

### Environment-setup

Setup a virtual environment and install the dependencies.

```zsh
python -m venv venv
source venv/bin/activate
pip install -r requirements.in
pip install -e .
```

### API Keys

Keep the API keys in the environment variable or in the `.env` file. For eg,

```zsh
OPENAI_API_KEY=<your_openai_api_key>
```

### Dataset setup

- Get the MMLU dataset from [here](https://github.com/hendrycks/test), and put it in the `data/mmlu` directory.
- Get the CommonSenseQA dataset from [here](https://www.tau-nlp.sites.tau.ac.il/commonsenseqa), and put it in the `data/commonsense_qa` directory.
- Get the GSM8K dataset from [here](https://github.com/openai/grade-school-math), and put it in the `GSM8k/gsm8k_test.jsonl`.

## Running the experiments

The basic debate procedure can be run via using the following command:

```zsh
python src/experiments/basic_debate.py task=gsm8k debug=True experiment.num_questions=1 experiment.num_rounds=2
```

The experiment parameters are specified in the `conf/experiments/basic_debate.yaml` file.

To run a single-agent baseline, use the basic debate with only 1 agent and `num_rounds = 0`, i.e.,

```zsh
python src/experiments/basic_debate.py task=gsm8k debug=False experiment.num_questions=30 experiment.num_rounds=2 agent_counts.0=1 agent_counts.1=0
```

### Wandb logging

You can add the wandb authentication on the command line via `wandb login` and copy the API key to the terminal. Alternatively, you can run the experiments without wandb logging via passing `wandb.mode = disabled` while running the experiments or modifying the configs.

```zsh
python src/experiments/basic_debate.py task=gsm8k debug=False experiment.num_questions=1 experiment.num_rounds=2 agent_counts.0=1 agent_counts.1=1 wandb.mode=disabled
```

### Cost tracking

In the experiment configs `src/conf/basic_debate.yaml` there is a flag `cost_check = True` that first estimates the cost for proprietry LLMs for one QA datapoint and then uses that to estimate the cost of the experiment. It then displays that cost to the user and asks for confirmation to proceed or not.

Although NOT RECOMMENDED, you can ovveride it by command line (for instance when you have verified the costs but want to do multiple runs) by following format:

```zsh
python src/experiments/basic_debate.py ....  cost_check=False
```

### Multi-run via Hydra

```zsh
python src/experiments/basic_debate.py -m task=gsm8k,math experiment.num_questions=1,5
```

### Checkpointing

Easiest way is to override the run directory to resume for experiment:

```zsh
python src/experiments/basic_debate.py task=... checkpoint.resume=True hydra.run.dir=/path/to/previous/run
```

# Running the experiments

## Single agent

Use the following command to benchmark various single agent models on the different tasks:

```zsh
python src/experiments/basic_debate.py -m task=gsm8k,mmlu experiment.num_questions=100 debug=False cost_check=False experiment.num_rounds=0 checkpoint.frequency=25 experiment.max_concurrent_calls=20 agent_counts.0=1 agent_counts.1=0 seed=0,1,2,3,4  "llm_conf@llm1=llama_3_1_8B"
```

# References

- [MMLU](https://github.com/hendrycks/test).
- [Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://github.com/composable-models/llm_multiagent_debate/tree/main).
- [LLM-Agora](https://github.com/gauss5930/LLM-Agora).
