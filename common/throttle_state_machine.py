from collections.abc import Hashable
from enum import IntEnum, auto
from typing import Any

from rlgym.rocket_league.api import GameState

from common.state_machine import StateMachine

class HCThrottleAction(IntEnum):
    LOCK_THROTTLE = 0
    ACCELERATE = auto()
    SLOW_DOWN = auto()
    
class HCThrottleState(IntEnum):
    THROTTLE_NEUTRAL = auto()
    THROTTLE_UP = auto()
    THROTTLE_DOWN = auto()

class ThrottleStateMachine(StateMachine[HCThrottleState, HCThrottleAction]):
    def reset(self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]):
        for agent in agents:
            self.states[agent] = HCThrottleState.THROTTLE_NEUTRAL
            
    def _step_neutral(self, agent: Hashable, action: HCThrottleAction):
        match action:
            case HCThrottleAction.ACCELERATE:
                self.states[agent] = HCThrottleState.THROTTLE_UP
            case HCThrottleAction.SLOW_DOWN:
                self.states[agent] = HCThrottleState.THROTTLE_DOWN
                
    def _step_throttle_down(self, agent: Hashable, action: HCThrottleAction):
        match action:
            case HCThrottleAction.ACCELERATE:
                self.states[agent] = HCThrottleState.THROTTLE_UP
            case HCThrottleAction.LOCK_THROTTLE:
                self.states[agent] = HCThrottleState.THROTTLE_NEUTRAL
                
                
    def _step_throttle_up(self, agent: Hashable, action: HCThrottleAction):
        match action:
            case HCThrottleAction.SLOW_DOWN:
                self.states[agent] = HCThrottleState.THROTTLE_DOWN
            case HCThrottleAction.LOCK_THROTTLE:
                self.states[agent] = HCThrottleState.THROTTLE_NEUTRAL
    
    def step(self, actions: dict[Hashable, HCThrottleAction]):
        for agent, action in actions.items():
            match self.states[agent]:
                case HCThrottleState.THROTTLE_NEUTRAL:
                    self._step_neutral(agent, action)
                case HCThrottleState.THROTTLE_DOWN:
                    self._step_throttle_down(agent, action)
                case HCThrottleState.THROTTLE_UP:
                    self._step_throttle_up(agent, action)