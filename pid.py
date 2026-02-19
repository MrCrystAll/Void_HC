from abc import ABC, abstractmethod
import math
from typing import Any, Callable, Hashable

import numpy as np
from rlgym.rocket_league.api import GameState

class PID(ABC):
    def __init__(self, p: float = 0, i: float = 0, d: float = 0) -> None:
        self.p, self.i, self.d = p, i, d
        self.p_error = {}
        self.i_error = {}
        self.d_error = {}
        
    @abstractmethod
    def get_targets(self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]) -> dict[Hashable, Any]:
        pass
        
    @abstractmethod
    def reset(self, agents: list[Hashable], initial_state: GameState, shared_info: dict[str, Any]):
        pass
        
    @abstractmethod
    def get_output(self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]) -> dict[Hashable, Any]:
        pass
    
    def apply_error(self, agent: Hashable, ticks_passed: int, error: Any):
        self.p_error[agent] = error * self.p
        
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
        
        return self.p_error[agent] + self.i_error[agent] + self.d_error[agent]
    
        
class SteerPID(PID):
    def __init__(self, p: float = 0, i: float = 0, d: float = 0, boost_threshold: float = 0, handbrake_threshold: float = math.pi / 2) -> None:
        super().__init__(p, i, d)
        self._last_tick_count = 0
        self.boost_threshold = boost_threshold
        self.handbrake_threshold = handbrake_threshold
    
    def reset(self, agents: list[Hashable], initial_state: GameState, shared_info: dict[str, Any]):
        self._last_tick_count = initial_state.tick_count
        
    def get_targets(self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]) -> dict[Hashable, Any]:
        return {agent: state.ball.position for agent in agents}
    
    def get_output(self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]) -> dict[Hashable, Any]:
        _expected_yaws = {}
        
        _targets = self.get_targets(agents, state, shared_info)
        
        for agent in agents:
            _car = state.cars[agent]
            _agent_position = _car.physics.position
            
            dx = _targets[agent][0] - _agent_position[0]
            dy = _targets[agent][1] - _agent_position[1]
            
            _angle = math.atan2(dy, dx)
            
            _error = _angle - _car.physics.yaw
            _error = (_error + math.pi) % (2*math.pi) - math.pi
            
            ticks_passed = max(state.tick_count - self._last_tick_count, 1)
            
            _expected_yaws[agent] = (self.apply_error(agent, ticks_passed, _error), abs(_error) < self.boost_threshold, abs(_error) > self.handbrake_threshold) 
            
        self._last_tick_count = state.tick_count
            
        return _expected_yaws

