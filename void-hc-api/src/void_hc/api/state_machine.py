"""The module for the base state machine"""

from abc import abstractmethod
from typing import Any, Generic

from rlgym.api import AgentID, StateType

from void_hc.api.hc_typing import MachineActionType, MachineStateType


class StateMachine(Generic[AgentID, MachineStateType, MachineActionType, StateType]):
    """A state machine is an object that updates its internal state based on an external action

    This can be represented as a graph, you start at a given node and
    an action allows you to move to another node, and so on
    """

    def __init__(self) -> None:
        self.states: dict[AgentID, MachineStateType] = {}

    @abstractmethod
    def reset(
        self, agents: list[AgentID], state: StateType, shared_info: dict[str, Any]
    ):
        """Resets the state machine to its initial state

        :param agents: The agents to reset the state of
        :type agents: list[AgentID]
        :param state: The state to use to compute stuff
        :type state: StateType
        :param shared_info: The shared info of the environment
        :type shared_info: dict[str, Any]
        """

    @abstractmethod
    def step(
        self,
        actions: dict[AgentID, MachineActionType],
        state: StateType,
        shared_info: dict[str, Any],
    ):
        """Steps the state machine

        :param actions: The actions to use to step the machine
        :type actions: dict[AgentID, MachineActionType]
        :param state: The state to compute stuff
        :type state: StateType
        :param shared_info: The shared info of the environment
        :type shared_info: dict[str, Any]
        """
