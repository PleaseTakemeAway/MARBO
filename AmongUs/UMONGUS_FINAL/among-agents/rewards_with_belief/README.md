# rewards_with_belief

`rewards_with_belief` builds preference datasets from belief-enabled Among Us
logs. The package assumes that action logs contain separate role-prediction
entries:

```text
[Relational Belief]
- Player 1: Role=Crewmate
- Player 2: Role=unknown
- Player 3: Role=Impostor
```

Labels follow the KTO convention:

| Label | Meaning |
| --- | --- |
| `True` | desirable completion |
| `False` | undesirable completion |
| `None` | skip / insufficient evidence |

## Data Sources

Belief-action agents use `BeliefStateLLMAgent`: they predict a relational belief
and condition their action on it. Opponent/measurement agents use
`BeliefPredictOnlyLLMAgent`: they predict and log beliefs but act and speak
without seeing those predictions.

The game loop logs listener belief updates at meeting start and immediately
after each `SPEAK` action. These post-speech role-prediction rows are the
primary signal for speech-induced belief-shift feedback.

## Reward Priority

`BeliefAwareGameReward` dispatches by phase/action:

1. skip non-clean belief-action entries;
2. task phase: observable hard-rule reward;
3. meeting vote: target true-role plus actor-belief consistency reward;
4. meeting speech: validity gate;
5. meeting speech: logged listener belief-shift reward;
6. ambiguous or missing belief-shift cases: skip.

## Task Actions

Task-phase rewards reuse the observable hard rules vendored into this package.

| Role | Desirable | Undesirable |
| --- | --- | --- |
| Crewmate | complete task, report body, call meeting with hard evidence | unsupported meeting call, ignore visible body, repeatedly ignore available task |
| Impostor | kill Crewmate with no witnesses, move/vent after own kill | witnessed kill, vent in front of Crewmate, repeatedly avoid available safe kill |

## Meeting Votes

Vote labels use true target roles and the actor's categorical relational belief:

| Voter | Desirable | Undesirable |
| --- | --- | --- |
| Crewmate | vote for a true Impostor that the actor believes is an Impostor | vote for a Crewmate; ignore a believed Impostor and vote elsewhere |
| Impostor | vote for a Crewmate | self-vote |

Unsupported true-Impostor guesses, teammate-distancing votes, and unknown target
roles are skipped.

## Meeting Speech

Speech reward is intentionally split into validity and belief-shift feedback.

### 1. Validity Gate

The default rule-based `FactVerifier`, or optional API-backed `LLMFactVerifier`,
checks only speaker-visible context. A fact-conflicting or unsupported speech is
labelled `False`. A valid speech is not automatically positive; it proceeds to
belief-shift scoring.

The LLM verifier can flag objective contradictions, unsupported direct witness
claims, role-certainty claims without visible support, game-rule errors, and
inappropriate/self-contradictory reasoning. It does not reconstruct listener
belief states.

### 2. Listener Belief Shift

For each valid speech, the reward compares relevant listener belief maps before
and after the speech. The game loop logs these measurements as paired
`pre_speech` and `post_speech` role-prediction rows with a shared
`Speech Event Id`; `pre_speech` rows are measurement-only snapshots, while
`post_speech` rows are committed as the listener's updated belief state.

Relevant listeners are Crewmates:

- for Crewmate speakers, the desired effect is to make Crewmate listeners more
  accurate about true roles;
- for Impostor speakers, the desired effect is to make Crewmate listeners less
  accurate or move them into a wrong decided role.

Each changed role prediction receives a score:

| Speaker | Favorable shift | Unfavorable shift |
| --- | --- | --- |
| Crewmate | listener moves toward the true role | listener moves away from the true role or into a wrong role |
| Impostor | listener moves away from the true role or into a wrong role | listener moves toward the true role |

Preference labels are assigned only from paired measurements for the same
`Speech Event Id`. Missing pre/post pairs give `None`. If a pair exists but no
role prediction changes, that listener contributes zero, so a zero or negative
aggregate shift gives `False`; only a positive aggregate shift gives `True`.

## Role-Prediction Labels

The auxiliary role-prediction dataset uses conservative target labels before
there is enough discussion context. Until at least two discussion rounds have
been observed in a game, labels include only the actor's own role, privately
known Impostor teammates, and observed `KILL`/`VENT` actors as `Impostor`; all
other players remain `unknown`. Once the discussion threshold is reached, later
role-prediction rows use oracle true-role labels for the full roster.

## Build Command

From `among-agents/`:

```bash
python -m rewards_with_belief.build_all_datasets \
  --log_root ../expt-logs/belief_shift_collection \
  --output_root rewards_with_belief/data/gemma \
  --include_models gemma \
  --overwrite
```

Optional LLM verifier:

```bash
python -m rewards_with_belief.build_all_datasets \
  --log_root ../expt-logs/belief_shift_collection \
  --output_root rewards_with_belief/data/gemma \
  --include_models gemma \
  --llm_verifier \
  --overwrite
```
