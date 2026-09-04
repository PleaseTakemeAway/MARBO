"""
Dataset sampling knobs for rewards_with_belief.

Edit this file directly when you want to change the final dataset size or
bucket ratios. These settings are intentionally not argparse flags so repeated
dataset builds use one auditable configuration.
"""

from typing import Optional


# Shared deterministic seed for all ratio sampling.
SAMPLING_SEED = 7

# None means "use the largest feasible size under the configured ratios".
# Set an integer, e.g. 50000, to force the final sampled KTO dataset size.
KTO_TOTAL_SIZE: Optional[int] = 2500

# None means "use the largest feasible size under the configured ratios".
# Set an integer, e.g. 50000, to force the final sampled SFT dataset size.
SFT_TOTAL_SIZE: Optional[int] = 500

# ---------------------------------------------------------------------------
# KTO action/vote/speech dataset sampling
# ---------------------------------------------------------------------------

KTO_SAMPLING_ENABLED = True
KTO_MODEL_RATIOS = None


# Overall True:False preference-label ratio. Default is 6:4.
KTO_LABEL_RATIOS = {
    True: 0.6,
    False: 0.4,
}

# Within each True/False label bucket, split examples across final KTO tasks.
# Supported KTO bucket names:
#   - task_action
#   - meeting_vote
#   - meeting_speech
KTO_DETAIL_RATIOS = {
    True: {
        "task_action": 0.35,
        "meeting_vote": 0.3,
        "meeting_speech": 0.35,
    },
    False: {
        "task_action": 0.35,
        "meeting_vote": 0.3,
        "meeting_speech": 0.35,
    },
}

# Optional actor-role ratio inside each KTO label/detail bucket.
#
# Supported values:
#   - None or {}: do not control Crewmate/Impostor ratios.
#   - dict: split each configured KTO_DETAIL_RATIOS bucket by actor role, e.g.
#       {"Crewmate": 0.7, "Impostor": 0.3}
#   - dict keyed by label: use different role ratios per True/False label, e.g.
#       {
#           True: {"Crewmate": 0.8, "Impostor": 0.2},
#           False: {"Crewmate": 0.5, "Impostor": 0.5},
#       }
#
# Role ratios are multiplied into KTO_LABEL_RATIOS and KTO_DETAIL_RATIOS.
# KTO_ROLE_RATIOS = None

KTO_ROLE_RATIOS = {
          True: {"Crewmate": 0.5, "Impostor": 0.5},
          False: {"Crewmate": 0.5, "Impostor": 0.5},
      }


# KTO_ROLE_RATIOS = None

# If False, scarce buckets are capped by available data. If True, scarce
# buckets are sampled with replacement to hit the exact requested count.
KTO_ALLOW_OVERSAMPLE = False

# If True and oversampling is disabled, missing samples from scarce buckets are
# filled from configured buckets that still have unused examples.
KTO_FILL_SHORTAGE = True

# Full KTO builds use the same task-action-only construction below for their
# task_action rows, then keep the normally sampled vote/speech rows unchanged.
KTO_REPLACE_TASK_ACTION_WITH_TASK_ONLY = True
KTO_REPLACE_TASK_ACTION_INCLUDE_REPAIR_NEGATIVES = True

# When --llm_verifier_after_sampling is used, exact speech buckets can be too
# sparse after LLM verification. If enabled, remaining meeting_speech shortage
# is filled from LLM-verified speech candidates.
KTO_SPEECH_FILL_SHORTAGE_FROM_VERIFIED = True

# Preserve the requested True/False label totals when filling speech shortage.
# Role ratios are relaxed first; label ratios are relaxed only if this is set
# to False.
KTO_SPEECH_FILL_SHORTAGE_KEEP_LABEL = True

# Number of exact-bucket LLM speech rounds to try before using the fallback.
KTO_SPEECH_FALLBACK_AFTER_ROUNDS = 1

# Rebalance labels inside Task-phase action types after the normal KTO sampler.
#
# This keeps the sampled task_action total and action-type totals. If an action
# type has both True and False source rows, it targets KTO_TASK_ACTION_LABEL_RATIOS
# and uses as many minority-label rows as available without oversampling.
KTO_TASK_ACTION_BALANCE_ENABLED = True
KTO_TASK_ACTION_LABEL_RATIOS = {
    True: 0.6,
    False: 0.4,
}

# Task-action-only dataset knobs. These are used only when build_dataset.py is
# run with --task_actions_only. The ratios intentionally downsample positive
# COMPLETE TASK rows and reserve negative mass for invalid repair attempts such
# as bare "[Action] COMPLETE TASK".
KTO_TASK_ONLY_SAMPLING_ENABLED = True
KTO_TASK_ONLY_TOTAL_SIZE: Optional[int] = 6000
KTO_TASK_ONLY_MODEL_RATIOS = None
KTO_TASK_ONLY_ROLE_RATIOS = None
KTO_TASK_ONLY_LABEL_RATIOS = {
    True: 0.6,
    False: 0.4,
}
KTO_TASK_ONLY_TYPE_RATIOS = {
    True: {
        "complete_task": 0.25,
        "movement": 0.45,
        "impostor_action": 0.20,
        "meeting_trigger": 0.08,
        "monitor": 0.02,
    },
    False: {
        "complete_task": 0.25,
        "movement": 0.45,
        "impostor_action": 0.20,
        "meeting_trigger": 0.08,
        "monitor": 0.02,
    },
}
KTO_TASK_ONLY_ALLOW_OVERSAMPLE = False
KTO_TASK_ONLY_FILL_SHORTAGE = True
KTO_TASK_ONLY_TYPE_LABEL_BALANCE_ENABLED = True
KTO_TASK_ONLY_TYPE_LABEL_RATIOS = {
    True: 0.6,
    False: 0.4,
}


# ---------------------------------------------------------------------------
# State reconstruction / role-prediction SFT dataset sampling
# ---------------------------------------------------------------------------

SFT_SAMPLING_ENABLED = True
SFT_MODEL_RATIOS = None


# Which SFT row fields control state-reconstruction sampling.
#
# Default uses a combined bucket:
#   (state_point, state_timing)
#
# This keeps both label-policy balance and temporal balance visible. If you
# want a single-axis ablation, set this to ("state_point",) or
# ("state_timing",) and update SFT_STATE_POINT_RATIOS keys accordingly.
SFT_RATIO_FIELDS = ("state_point", "state_timing")

# Ratio by SFT reconstruction bucket. The default keys below are
# (state_point, state_timing) tuples.
#
# Supported default state point names:
#   - task_crewmate_rule
#   - task_impostor_rule
#   - meeting_crewmate_rule
#   - meeting_impostor_rule
#   - meeting_crewmate_oracle
#   - meeting_impostor_oracle
#
# Supported default state timing names:
#   - task_phase
#   - meeting_start
#   - meeting_after_speech
#   - meeting_after_vote
#   - meeting_after_other
SFT_STATE_POINT_RATIOS = {
    ("task_crewmate_rule", "task_phase"): 0.25,
    ("task_impostor_rule", "task_phase"): 0.10,
    ("meeting_crewmate_rule", "meeting_start"): 0.05,
    ("meeting_crewmate_rule", "meeting_after_speech"): 0.15,
    ("meeting_impostor_rule", "meeting_start"): 0.03,
    ("meeting_impostor_rule", "meeting_after_speech"): 0.07,
    ("meeting_crewmate_oracle", "meeting_after_speech"): 0.22,
    ("meeting_crewmate_oracle", "meeting_after_vote"): 0.05,
    ("meeting_impostor_oracle", "meeting_after_speech"): 0.06,
    ("meeting_impostor_oracle", "meeting_after_vote"): 0.02,
}

SFT_ALLOW_OVERSAMPLE = False
SFT_FILL_SHORTAGE = True



# Example:
#
# python -m rewards_with_belief.build_all_datasets \
#   --log_root ../expt-logs/belief_shift_collection \
#   --output_root rewards_with_belief/data/gemma \
#   --include_models gemma \
#   --llm_verifier \
#   --overwrite
