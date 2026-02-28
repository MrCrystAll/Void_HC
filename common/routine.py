from abc import abstractmethod
from collections.abc import Hashable
from typing import Any, Generic, TypeVar

from rlgym.api import ActionType, EngineActionType
from rlgym.rocket_league.api import GameState

from common.state_machine import StateMachine

StateMachineType = TypeVar("StateMachineType", bound=StateMachine)


class Routine(Generic[ActionType, EngineActionType, StateMachineType]):
    """A routine is an object holding a state machine and applying modifications to an action
    """
    def apply_outputs(
        self,
        actions: dict[Hashable, ActionType],
        current_output: dict[Hashable, EngineActionType],
        state: GameState,
        shared_info: dict[str, Any],
    ) -> dict[Hashable, EngineActionType]:
        """Applies modification to a dictionnary of agents/actions

        :param actions: The actions performed by the agents
        :type actions: dict[Hashable, ActionType]
        :param current_output: The output to modify
        :type current_output: dict[Hashable, EngineActionType]
        :param state: The state to use to calculate changes
        :type state: GameState
        :param shared_info: The shared info of the environment
        :type shared_info: dict[str, Any]
        :return: Modified actions
        :rtype: dict[Hashable, EngineActionType]
        """
        return current_output

    def reset(
        self,
        agents: list[Hashable],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ):
        """Resets the state machine of the object

        :param agents: Agents to reset
        :type agents: list[Hashable]
        :param initial_state: The state to reset on
        :type initial_state: GameState
        :param shared_info: The shared info upon reset
        :type shared_info: dict[str, Any]
        """
        self.state_machine.reset(agents, initial_state, shared_info)

    @property
    @abstractmethod
    def state_machine(self) -> StateMachineType:
        """The state machine of the routine is what drives the output of the apply_outputs method
        This is a getter to that machine

        :return: The state machine of the object
        :rtype: StateMachineType
        """
        pass
