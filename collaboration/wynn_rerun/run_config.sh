#!/bin/bash
# Run one Wynn configuration over seeds with the authors' basic_debate.py.
# Usage: run_config.sh <name> <n_llama> <n_mistral> <num_questions> <seeds...>
cd /home/lian/swarmworld/research
PT=pytorch/pytorch@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2
R=$PWD/reports/wynn-rerun-20261120-a; name=$1 nl=$2 nm=$3 nq=$4; shift 4
mkdir -p $R/out/$name
docker run --rm --name wynn-client-$name --network wynn-net --cpus 4 --memory 16g --user $(id -u):$(id -g) -e HOME=/tmp \
  -e OPENAI_API_KEY=sk-local-placeholder -e WANDB_MODE=disabled -e WANDB_SILENT=true -e PYTHONPATH=/w -e PYTHONDONTWRITEBYTECODE=1 \
  -v $PWD/envs/wynn-py311:/env:ro -v $PWD/collaboration/wynn_rerun/talk-aint-cheap:/w:ro -v $R/out/$name:/out \
  --workdir /w --entrypoint bash $PT -c "
  for s in $*; do
    timeout 7200s /env/bin/python src/experiments/basic_debate.py task=gsm8k experiment.name=$name experiment.num_questions=$nq \
      experiment.num_rounds=2 agent_counts.0=$nl agent_counts.1=$nm seed=\$s wandb.mode=disabled cost_check=False \
      experiment.max_concurrent_calls=60 hydra.run.dir=/out/seed\$s > /out/seed\$s.log 2>&1; echo \"seed \$s exit \$?\" >> /out/status.txt
  done"
