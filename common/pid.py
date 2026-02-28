from abc import ABC, abstractmethod
from collections.abc import Hashable
from typing import Any

import numpy as np
from rlgym.rocket_league.api import GameState

from common.target_shared_info_provider import TARGET_HEADER


class PID(ABC):
    def __init__(self, p: float = 0, i: float = 0, d: float = 0) -> None:
        self.p, self.i, self.d = p, i, d
        self.p_error = {}
        self.i_error = {}
        self.d_error = {}
        self._raw_error = {}
        self._computed_error = {}

    @abstractmethod
    def get_targets(
        self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]
    ) -> dict[Hashable, Any]:
        pass

    @abstractmethod
    def reset(
        self,
        agents: list[Hashable],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ):
        pass
    
    @abstractmethod
    def update_error(self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]):
        pass

    @abstractmethod
    def get_output(
        self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]
    ) -> dict[Hashable, Any]:
        pass

    def apply_error(self, agent: Hashable, ticks_passed: int, error: Any) -> None:
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

        self._computed_error[agent] = self.p_error[agent] + self.i_error[agent] + self.d_error[agent]


class SteerPID(PID):
    def __init__(
        self,
        p: float = 0,
        i: float = 0,
        d: float = 0
    ) -> None:
        super().__init__(p, i, d)
        self._last_tick_count = 0

    def reset(
        self,
        agents: list[Hashable],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ):
        self._last_tick_count = initial_state.tick_count

    def get_targets(
        self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]
    ) -> dict[Hashable, Any]:
        return {agent: state.ball.position for agent in agents}
    
    def update_error(self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]):
        _targets = self.get_targets(agents, state, shared_info)
        
        for agent, target in _targets.items():
            shared_info[TARGET_HEADER][agent]["steer"] = target
        
        ticks_passed = max(state.tick_count - self._last_tick_count, 1)
        
        for agent in agents:
            _car = state.cars[agent]
            _agent_position = _car.physics.position

            _to_target = _targets[agent] - _agent_position
            _to_target /= np.linalg.norm(_to_target)

            _error = np.cross(_car.physics.forward, _to_target)
            _error = np.dot(_error, _car.physics.up)
            
            self.apply_error(agent, ticks_passed, _error)
        self._last_tick_count = state.tick_count

    def get_output(
        self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]
    ) -> dict[Hashable, Any]:
        _expected_yaws = {}

        for agent in agents:

            _expected_yaws[agent] = self._computed_error[agent]

        return _expected_yaws


class PitchPID(PID):
    def __init__(self, p: float = 0, i: float = 0, d: float = 0) -> None:
        super().__init__(p, i, d)
        self._last_tick_count = 0

    def reset(
        self,
        agents: list[Hashable],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ):
        self._last_tick_count = initial_state.tick_count

    def get_targets(
        self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]
    ) -> dict[Hashable, Any]:
        return {agent: state.ball.position for agent in agents}
    
    def update_error(self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]):
        _targets = self.get_targets(agents, state, shared_info)
        ticks_passed = max(state.tick_count - self._last_tick_count, 1)
        
        for agent in agents:
            _car = state.cars[agent]
            _agent_position = _car.physics.position

            _to_target = _targets[agent] - _agent_position
            _to_target /= np.linalg.norm(_to_target)

            _error = np.cross(_car.physics.forward, _to_target)
            _error = np.dot(_error, _car.physics.left)

            self.apply_error(agent, ticks_passed, _error)
            
        self._last_tick_count = state.tick_count

    def get_output(
        self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]
    ) -> dict[Hashable, Any]:
        _expected_pitches = {}

        for agent in agents:
            _expected_pitches[agent] = self._computed_error[agent]

        return _expected_pitches


class RollPID(PID):
    def __init__(self, p: float = 0, i: float = 0, d: float = 0) -> None:
        super().__init__(p, i, d)
        self._last_tick_count = 0

    def reset(
        self,
        agents: list[Hashable],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ):
        self._last_tick_count = initial_state.tick_count

    def get_targets(
        self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]
    ) -> dict[Hashable, Any]:
        return {agent: np.asarray([0, 0, 1]) for agent in agents}
    
    def update_error(self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]):
        _targets = self.get_targets(agents, state, shared_info)

        for agent in agents:
            _car = state.cars[agent]

            _error = np.cross(_targets[agent], _car.physics.up)
            _error = np.dot(_error, _car.physics.forward)

            ticks_passed = max(state.tick_count - self._last_tick_count, 1)

            self.apply_error(agent, ticks_passed, _error)
            
        self._last_tick_count = state.tick_count

    def get_output(
        self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]
    ) -> dict[Hashable, Any]:
        _expected_pitches = {}

        for agent in agents:
            _expected_pitches[agent] = self._computed_error[agent]

        return _expected_pitches
