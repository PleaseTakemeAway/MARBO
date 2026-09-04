# UMONGUS

This repository contains the code for collecting Among Us belief trajectories,
building belief-aware KTO/SFT datasets, and training Gemma policies with KTO plus
auxiliary SFT(CE) supervision.

Generated logs, checkpoints, cache files, and experiment outputs are not required
for the code release.

## Layout

```text
collect_belief_data.py               # data-collection entrypoint
run/collect_belief_data.sh           # vLLM/OpenRouter data-collection launcher
among-agents/amongagents/            # Among Us environment and LLM agents
among-agents/rewards_with_belief/     # reward and dataset construction code
among-agents/training/                # KTO + auxiliary SFT(CE) training code
```

## Setup

```bash
pip install -r requirements.txt
pip install -e among-agents
```

For hosted models, set `OPENROUTER_API_KEY`. Local open-source models are served
through vLLM by `run/collect_belief_data.sh`.

## Collect Trajectories

```bash
NUM_GAMES=20 RATE_LIMIT=4 bash run/collect_belief_data.sh
```

The default belief-action models are:

```text
google/gemma-4-E2B-it
google/gemma-4-E4B-it
```

Logs are written to:

```text
expt-logs/belief_shift_collection/<run_name>/
```

## Build Datasets

```bash
cd among-agents
python -m rewards_with_belief.build_all_datasets \
  --log_root ../expt-logs/belief_shift_collection \
  --output_root rewards_with_belief/data/gemma \
  --include_models gemma \
  --overwrite
```

Add `--llm_verifier` to use the API-backed speech fact verifier. Without it,
speech validity uses the rule-based verifier.

## Train

```bash
cd among-agents
CUDA_DEVICES=0,1 MODELS="E2B E4B" USE_WANDB=false \
  bash training/run_gemma_kto_with_state_recon.sh
```

The training script reads:

```text
rewards_with_belief/data/gemma/kto_dataset
rewards_with_belief/data/gemma/role_prediction_sft_dataset
```

and writes checkpoints under:

```text
among-agents/models/gemma-kto-sft-ce/
```

Common overrides:

```bash
CUDA_DEVICES=0
MODELS="E2B"
OUTPUT_ROOT=...
KTO_DATASET_PATH=...
AUX_SFT_DATASET_PATH=...
```
