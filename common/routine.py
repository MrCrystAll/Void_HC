from abc import abstractmethod
from collections.abc import Hashable
from typing import Any, Generic, TypeVar

from rlgym.api import ActionType, EngineActionType
from rlgym.rocket_league.api import GameState

from common.state_machine import StateMachine

StateMachineType = TypeVar("StateMachineType", bound=StateMachine)

class Routine(Generic[ActionType, EngineActionType, StateMachineType]):
    def apply_outputs(self, actions: dict[Hashable, ActionType], current_output: dict[Hashable, EngineActionType], state: GameState, shared_info: dict[str, Any]) -> dict[Hashable, EngineActionType]:
        return current_output
    
    def reset(self, agents: list[Hashable], initial_state: GameState, shared_info: dict[str, Any]):
        self.state_machine.reset(agents, initial_state, shared_info)
    
    @property
    @abstractmethod
    def state_machine(self) -> StateMachineType:
        pass