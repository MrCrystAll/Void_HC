"""This module contains the base routine class"""

from abc import abstractmethod
from typing import Any, Generic, TypeVar

from rlgym.api import ActionType, EngineActionType, StateType, AgentID

from void_hc.api.hc_typing import HCMachineAction
from void_hc.api.state_machine import StateMachine

StateMachineType = TypeVar("StateMachineType", bound=StateMachine)


class Routine(
    Generic[AgentID, ActionType, EngineActionType, StateMachineType, StateType]
):
    """A routine is an object holding a state machine and applying modifications to an action"""

    @abstractmethod
    def apply_outputs(
        self,
        actions: dict[AgentID, ActionType],
        current_output: dict[AgentID, EngineActionType],
        state: StateType,
        shared_info: dict[str, Any],
    ) -> dict[AgentID, EngineActionType]:
        """Applies modification to a dictionnary of agents/actions

        :param actions: The actions performed by the agents
        :type actions: dict[AgentID, ActionType]
        :param current_output: The output to modify
        :type current_output: dict[AgentID, EngineActionType]
        :param state: The state to use to calculate changes
        :type state: StateType
        :param shared_info: The shared info of the environment
        :type shared_info: dict[str, Any]
        :return: Modified actions
        :rtype: dict[AgentID, EngineActionType]
        """

    def reset(
        self,
        agents: list[AgentID],
        initial_state: StateType,
        shared_info: dict[str, Any],
    ):
        """Resets the state machine of the object

        :param agents: Agents to reset
        :type agents: list[AgentID]
        :param initial_state: The state to reset on
        :type initial_state: StateType
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

    @abstractmethod
    def get_sub_action_from_top_action(self, top_action: HCMachineAction) -> ActionType:
        """Fetches the action from the action returned by the action parser

        :param top_action: The action returned by the action parser
        :type top_action: HCMachineAction
        :return: The action made for the routine and state machine
        :rtype: ActionType
        """
