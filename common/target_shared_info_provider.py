from collections.abc import Hashable
from typing import Any, Dict, List

from rlgym.api import SharedInfoProvider
from rlgym.rocket_league.api import GameState

TARGET_HEADER = "targets"


class TargetSharedInfoProvider(SharedInfoProvider[Hashable, GameState]):
    def create(self, shared_info: Dict[str, Any]) -> Dict[str, Any]:
        shared_info[TARGET_HEADER] = {}

        return shared_info

    def set_state(
        self,
        agents: List[Hashable],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        for agent in agents:
            shared_info[TARGET_HEADER][agent] = {}
        return shared_info

    def step(
        self, agents: List[Hashable], state: GameState, shared_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        return shared_info
