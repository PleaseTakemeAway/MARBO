from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class RewardFunction:
    """Abstract interface for KTO labeling."""

    def label(
        self,
        entry: Dict[str, Any],
        game_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[bool]:
        raise NotImplementedError


@dataclass(frozen=True)
class RewardDecision:
    """Inspectable reward output used by notebooks and tests."""

    label: Optional[bool]
    source: str
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
