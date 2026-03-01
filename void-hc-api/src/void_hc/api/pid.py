from abc import abstractmethod
from typing import Any, Generic

from rlgym.api import StateType, AgentID


class PID(Generic[AgentID, StateType]):
    def __init__(self, p: float = 0, i: float = 0, d: float = 0) -> None:
        self.p, self.i, self.d = p, i, d
        self.p_error = {}
        self.i_error = {}
        self.d_error = {}
        self._raw_error = {}
        self._computed_error = {}

    @abstractmethod
    def get_targets(
        self, agents: list[AgentID], state: StateType, shared_info: dict[str, Any]
    ) -> dict[AgentID, Any]:
        pass

    @abstractmethod
    def reset(
        self,
        agents: list[AgentID],
        initial_state: StateType,
        shared_info: dict[str, Any],
    ):
        pass

    @abstractmethod
    def update_error(
        self, agents: list[AgentID], state: StateType, shared_info: dict[str, Any]
    ):
        pass

    @abstractmethod
    def get_output(
        self, agents: list[AgentID], state: StateType, shared_info: dict[str, Any]
    ) -> dict[AgentID, Any]:
        pass

    def apply_error(self, agent: AgentID, ticks_passed: int, error: Any) -> None:
        self.p_error[agent] = error * self.p
        self._raw_error[agent] = error

        if agent not in self.i_error:
            self.i_error[agent] = error * ticks_passed
        else:
            self.i_error[agent] += error * ticks_passed

        self.i_error[agent] *= self.i

        if agent not in self.d_error:
            self.d_error[agent] = error / ticks_passed
        else:
            self.d_error[agent] = (error - self.d_error[agent]) / ticks_passed

        self.d_error[agent] *= self.d

        self._computed_error[agent] = (
            self.p_error[agent] + self.i_error[agent] + self.d_error[agent]
        )
