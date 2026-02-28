from abc import abstractmethod
from collections.abc import Hashable
from typing import Any, Generic

from rlgym.rocket_league.api import GameState

from common.hc_typing import MachineActionType, MachineStateType

class StateMachine(Generic[MachineStateType, MachineActionType]):
    def __init__(self) -> None:
        self.states: dict[Hashable, MachineStateType] = {}
        
    @abstractmethod
    def reset(self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]):
        pass
    
    @abstractmethod
    def step(self, actions: dict[Hashable, MachineActionType], state: GameState, shared_info: dict[str, Any]):
        pass