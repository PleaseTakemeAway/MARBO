from dataclasses import dataclass
from typing import Dict, Optional


BeliefMap = Dict[str, str]


@dataclass(frozen=True)
class VerifierTransition:
    """
    Optional precomputed state-reconstruction verifier output.

    The reward package does not call an LLM during dataset construction. If a
    separate verifier pass is run later, store observer-visible before/after
    belief maps externally and join them into reward analysis with a stable
    event key. Current speech rewards prefer logged opponent belief transitions
    from the actual game traces over verifier-generated transitions.
    """

    before: BeliefMap
    after: BeliefMap
    verifier_model: str
    notes: str = ""


class PrecomputedStateReconstructionVerifier:
    """Small lookup wrapper for offline LLM-verifier annotations."""

    def __init__(self, transitions: Optional[Dict[str, VerifierTransition]] = None) -> None:
        self.transitions = transitions or {}

    def get(self, event_key: str) -> Optional[VerifierTransition]:
        return self.transitions.get(event_key)
