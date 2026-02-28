"""The module for the base state machine"""

from abc import abstractmethod
from collections.abc import Hashable
from typing import Any, Generic

from rlgym.rocket_league.api import GameState

from common.hc_typing import MachineActionType, MachineStateType


class StateMachine(Generic[MachineStateType, MachineActionType]):
    """A state machine is an object that updates its internal state based on an external action

    This can be represented as a graph, you start at a given node and
    an action allows you to move to another node, and so on
    """

    def __init__(self) -> None:
        self.states: dict[Hashable, MachineStateType] = {}

    @abstractmethod
    def reset(
        self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]
    ):
        """Resets the state machine to its initial state

        :param agents: The agents to reset the state of
        :type agents: list[Hashable]
        :param state: The state to use to compute stuff
        :type state: GameState
        :param shared_info: The shared info of the environment
        :type shared_info: dict[str, Any]
        """

    @abstractmethod
    def step(
        self,
        actions: dict[Hashable, MachineActionType],
        state: GameState,
        shared_info: dict[str, Any],
    ):
        """Steps the state machine

        :param actions: The actions to use to step the machine
        :type actions: dict[Hashable, MachineActionType]
        :param state: The state to compute stuff
        :type state: GameState
        :param shared_info: The shared info of the environment
        :type shared_info: dict[str, Any]
        """
