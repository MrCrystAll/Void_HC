from collections.abc import Hashable
from typing import Any, Iterator

import numpy as np
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import THROTTLE

from common.pid import PitchPID, RollPID, SteerPID


def cap(val, min_val, max_val):
    return max(min_val, min(max_val, val))


class GymActionReturn:
    def __init__(
        self,
        throttle: float = 0,
        steer: float = 0,
        pitch: float = 0,
        yaw: float = 0,
        roll: float = 0,
        jump: bool = False,
        boost: bool = False,
        handbrake: bool = False,
    ) -> None:
        self.throttle = throttle
        self.steer = steer
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        self.jump = jump
        self.boost = boost
        self.handbrake = handbrake

    def to_list(self) -> list[float]:
        return [
            cap(self.throttle, -1, 1),
            cap(self.steer, -1, 1),
            cap(self.pitch, -1, 1),
            cap(self.yaw, -1, 1),
            cap(self.roll, -1, 1),
            float(self.jump),
            float(self.boost),
            float(self.handbrake),
        ]
        
    def to_ndarray(self) -> np.ndarray:
        return np.asarray(self.to_list())


class HCGymBot:
    def __init__(self) -> None:
        self.steer_towards_ball_pid: SteerPID = SteerPID(
            3, 0.1, 0.2, boost_threshold=0, handbrake_threshold=0.6
        )
        self.in_air_steer_towards_ball_pid: SteerPID = SteerPID(
            0.4, 0.01, 0.1, boost_threshold=0, handbrake_threshold=2
        )
        self.pitch_towards_ball_pid: PitchPID = PitchPID(1, 0, 0.3)
        self.roll_stabilization_pid: RollPID = RollPID(0.5, 0.1, 0.5)
        self._flip_to_ball_routines: dict[Hashable, Iterator] = {}

    def reset(
        self,
        agents: list[Hashable],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ):
        self.steer_towards_ball_pid.reset(agents, initial_state, shared_info)
        self.in_air_steer_towards_ball_pid.reset(agents, initial_state, shared_info)
        self.pitch_towards_ball_pid.reset(agents, initial_state, shared_info)
        self.roll_stabilization_pid.reset(agents, initial_state, shared_info)
        
    def update_errors(self, agents: list[Hashable], game_state: GameState, shared_info: dict[str, Any]):
        self.steer_towards_ball_pid.update_error(agents, game_state, shared_info)
        self.in_air_steer_towards_ball_pid.update_error(agents, game_state, shared_info)
        self.pitch_towards_ball_pid.update_error(agents, game_state, shared_info)
        self.roll_stabilization_pid.update_error(agents, game_state, shared_info)
        
    def go_to_ball_output(self, agents: list[Hashable], game_state: GameState, shared_info: dict[str, Any], last_actions: dict[Hashable, np.ndarray]) -> dict[Hashable, GymActionReturn]:
        actions = {}

        on_ground_yaws = self.steer_towards_ball_pid.get_output(
            agents, game_state, shared_info
        )
        in_air_yaws = self.in_air_steer_towards_ball_pid.get_output(
            agents, game_state, shared_info
        )
        pitches = self.pitch_towards_ball_pid.get_output(
            agents, game_state, shared_info
        )
        rolls = self.roll_stabilization_pid.get_output(agents, game_state, shared_info)

        for agent in agents:
            if game_state.cars[agent].on_ground:
                yaws = on_ground_yaws
            else:
                yaws = in_air_yaws

            action = GymActionReturn(
                throttle=last_actions[agent][THROTTLE],
                roll=rolls[agent],
                pitch=pitches[agent],
                yaw=yaws[agent][0],
                steer=yaws[agent][0],
                boost=yaws[agent][1],
                handbrake=yaws[agent][2],
            )

            actions[agent] = action

        return actions
    
    def go_away_from_ball_output(self, agents: list[Hashable], game_state: GameState, shared_info: dict[str, Any], last_actions: dict[Hashable, np.ndarray]) -> dict[Hashable, GymActionReturn]:
        _base_actions = self.go_to_ball_output(agents, game_state, shared_info, last_actions).copy()
        
        for agent in agents:
            _base_actions[agent].steer *= -1
            _base_actions[agent].yaw *= -1
            
        return _base_actions
    
    def go_to_nearest_boost(self, agents: list[Hashable], game_state: GameState, shared_info: dict[str, Any], last_actions: dict[Hashable, np.ndarray]) -> dict[Hashable, GymActionReturn]:
        raise NotImplementedError("TODO: Implement nearest boost")
    
    def kickoff_routine(self, agents: list[Hashable], game_state: GameState, shared_info: dict[str, Any], last_actions: dict[Hashable, np.ndarray]) -> dict[Hashable, GymActionReturn]:
        raise NotImplementedError("TODO: Implement kickoff")
    
    def go_to_ball_speed_scaled(self, agents: list[Hashable], game_state: GameState, shared_info: dict[str, Any], last_actions: dict[Hashable, np.ndarray]) -> dict[Hashable, GymActionReturn]:
        _base_actions = self.go_to_ball_output(agents, game_state, shared_info, last_actions).copy()
        
        MAX_DIST_TO_BALL = 500
        
        for agent in agents:
            _car = game_state.cars[agent]
            
            _agent_to_ball = game_state.ball.position - game_state.cars[agent].physics.position
            _agent_dist_to_ball = np.linalg.norm(_agent_to_ball)
            _agent_to_ball /= _agent_dist_to_ball
            _agent_forward = game_state.cars[agent].physics.forward
            _agent_vel = _car.physics.linear_velocity
            _agent_speed = np.linalg.norm(_agent_vel)
            _agent_vel /= (_agent_speed + 1e-8)
            
            
            
            _base_actions[agent].throttle = float(min(_agent_dist_to_ball / MAX_DIST_TO_BALL, 1))
            
            if bool(_agent_dist_to_ball > MAX_DIST_TO_BALL) and bool(_agent_forward.dot(_agent_to_ball) > 0.9) and _agent_vel[:2].dot(_agent_forward[:2]) > 0.95 and _agent_speed > 600:
                _base_actions[agent].boost = True
                
                _flip_control = self.flip_to_ball([agent], game_state, shared_info, last_actions)
                _base_actions[agent].pitch = _flip_control[agent].pitch
                _base_actions[agent].yaw = _flip_control[agent].yaw
                _base_actions[agent].jump = _flip_control[agent].jump
             
        return _base_actions
    
    def flip_to_ball(self, agents: list[Hashable], game_state: GameState, shared_info: dict[str, Any], last_actions: dict[Hashable, np.ndarray]) -> dict[Hashable, GymActionReturn]:
        _actions = {}
        
        _agents_to_routine = []
        
        for agent in agents:
            if agent not in self._flip_to_ball_routines:
                _agents_to_routine.append(agent)
            else:
                try:
                    _actions[agent] = next(self._flip_to_ball_routines[agent])
                except StopIteration:
                    _agents_to_routine.append(agent)
                
        self.flip_towards_ball_routine(_agents_to_routine, game_state, shared_info, last_actions)
        
        for agent in _agents_to_routine:
            _actions[agent] = next(self._flip_to_ball_routines[agent])
            
        return _actions
    
    def flip_towards_ball_routine(self, agents: list[Hashable], game_state: GameState, shared_info: dict[str, Any], last_actions: dict[Hashable, np.ndarray]):
        _go_to_ball_steer = self.steer_towards_ball_pid.get_output(agents, game_state, shared_info)

        for agent in agents:
            _car = game_state.cars[agent]
            
            _agent_to_ball = _car.physics.position - game_state.ball.position
            _agent_to_ball /= np.linalg.norm(_agent_to_ball)
            
            _agent_forward = _car.physics.forward
            
            _agent_sequence = [
                GymActionReturn(throttle=last_actions[agent][THROTTLE], jump=True),
                GymActionReturn(throttle=last_actions[agent][THROTTLE], yaw=_go_to_ball_steer[agent][0], jump=False),
                GymActionReturn(throttle=last_actions[agent][THROTTLE], yaw=_go_to_ball_steer[agent][0], jump=False),
                GymActionReturn(throttle=last_actions[agent][THROTTLE], yaw=_go_to_ball_steer[agent][0], jump=False),
                GymActionReturn(throttle=last_actions[agent][THROTTLE], yaw=_go_to_ball_steer[agent][0], jump=False),
                GymActionReturn(throttle=last_actions[agent][THROTTLE], jump=True, yaw=_go_to_ball_steer[agent][0], pitch=np.dot(_agent_forward, _agent_to_ball)),
                GymActionReturn(throttle=last_actions[agent][THROTTLE], jump=True, yaw=_go_to_ball_steer[agent][0], pitch=np.dot(_agent_forward, _agent_to_ball)),
                GymActionReturn(throttle=last_actions[agent][THROTTLE], jump=True, yaw=_go_to_ball_steer[agent][0], pitch=np.dot(_agent_forward, _agent_to_ball)),
            ]
            
            self._flip_to_ball_routines[agent] = iter(_agent_sequence)