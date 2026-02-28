from collections.abc import Hashable
from typing import Any

import numpy as np
from rlgym.rocket_league.common_values import CAR_MAX_SPEED
from rlgym.rocket_league.api import GameState

from common.pid import PID
from common.throttle_state_machine import HCThrottleState


class SpeedRegulator(PID):
    def __init__(self, p: float = 0, i: float = 0, d: float = 0, speed_increment: float = 10) -> None:
        super().__init__(p, i, d)
        self.speed_targets: dict[Hashable, float] = {}
        self.speed_increment = speed_increment
        self._last_tick_count = 0
        
    def update_target(self, states: dict[Hashable, HCThrottleState]):
        for agent, state in states.items():
            if state == HCThrottleState.THROTTLE_DOWN:
                self.speed_targets[agent] = max(self.speed_targets[agent] - self.speed_increment, -CAR_MAX_SPEED)
            elif state == HCThrottleState.THROTTLE_UP:
                self.speed_targets[agent] = min(self.speed_targets[agent] + self.speed_increment, CAR_MAX_SPEED)
                
    def update_error(self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]):
        _targets = self.get_targets(agents, state, shared_info)
        ticks_passed = max(state.tick_count - self._last_tick_count, 1)
        
        for agent in agents:
            _norm_current_speed = np.linalg.norm(state.cars[agent].physics.linear_velocity) / CAR_MAX_SPEED
            _norm_target_speed = _targets[agent] / CAR_MAX_SPEED
            
            _error = _norm_target_speed - _norm_current_speed
            
            self.apply_error(agent, ticks_passed, _error)
        self._last_tick_count = state.tick_count

    def reset(self, agents: list[Hashable], initial_state: GameState, shared_info: dict[str, Any]):
        for agent in agents:
            self.speed_targets[agent] = float(np.linalg.norm(initial_state.cars[agent].physics.linear_velocity))
        self._last_tick_count = initial_state.tick_count
    
    
    def get_targets(self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]) -> dict[Hashable, Any]:
        return self.speed_targets
    
    def get_output(self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]) -> dict[Hashable, Any]:
        self.update_error(agents, state, shared_info)
        _controls = {}
        
        for agent in agents:
            _controls[agent] = self._computed_error[agent]
            
        return _controls