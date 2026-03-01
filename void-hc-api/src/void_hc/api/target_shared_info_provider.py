from typing import Any, Dict, Generic, List

from rlgym.api import SharedInfoProvider, AgentID, StateType

TARGET_HEADER = "targets"


class TargetSharedInfoProvider(
    Generic[AgentID, StateType], SharedInfoProvider[AgentID, StateType]
):
    def create(self, shared_info: Dict[str, Any]) -> Dict[str, Any]:
        shared_info[TARGET_HEADER] = {}

        return shared_info

    def set_state(
        self,
        agents: List[AgentID],
        initial_state: StateType,
        shared_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        for agent in agents:
            shared_info[TARGET_HEADER][agent] = {}
        return shared_info

    def step(
        self, agents: List[AgentID], state: StateType, shared_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        return shared_info
