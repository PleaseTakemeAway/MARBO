"""
Compatibility wrapper.

Dataset construction now lives in rewards_with_belief so training code can load a
prebuilt dataset instead of building labels during KTO training.
"""

from rewards_with_belief.data_builder import *  # noqa: F401,F403
