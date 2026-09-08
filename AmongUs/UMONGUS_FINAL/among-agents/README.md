# among-agents

This package contains the Among Us environment, LLM agents, dataset builders,
and Gemma training scripts used by the submission code.

Use the repository-level `collect_belief_data.py` entrypoint for data
collection. It assigns required Gemma belief-action models to
`BeliefStateLLMAgent` and all opponent/measurement LLM agents to
`BeliefPredictOnlyLLMAgent`, then records meeting-start and post-speech
listener belief sweeps.

Main subpackages:

```text
amongagents/              # environment, players, actions, and agent classes
rewards_with_belief/      # KTO/SFT dataset construction with listener belief-shift speech reward
training/                 # Gemma KTO + state-reconstruction training
```

See the root `README.md` for the full collection, reward, and training
commands.
