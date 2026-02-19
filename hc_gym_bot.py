from typing import Any, Hashable

import numpy as np
from rlgym.rocket_league.api import GameState

from pid import SteerPID

class GymActionReturn:
    def __init__(self, throttle: float = 0, steer: float = 0, pitch: float = 0, yaw: float = 0, roll: float = 0, jump: bool = False, boost: bool = False, handbrake: bool = False) -> None:
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
            self.throttle,
            self.steer,
            self.pitch,
            self.yaw,
            self.roll,
            float(self.jump),
            float(self.boost),
            float(self.handbrake)
        ]

class HCGymBot:
    def __init__(self) -> None:
        self.steer_towards_ball_pid: SteerPID = SteerPID(1, 0.01, 1, boost_threshold=0.3)
    
    def reset(self, agents: list[Hashable], initial_state: GameState, shared_info: dict[str, Any]):
        self.steer_towards_ball_pid.reset(agents, initial_state, shared_info)
    
    def get_output(self, agents: list[Hashable], game_state: GameState, shared_info: dict[str, Any]) -> dict[Hashable, GymActionReturn]:
        assert self.steer_towards_ball_pid is not None, "Steer PID has not been initialized"
        
        actions = {}
        
        yaws = self.steer_towards_ball_pid.get_output(agents, game_state, shared_info)
        
        for agent in agents:
            action = GymActionReturn(throttle=1, yaw=yaws[agent][0], steer=yaws[agent][0], boost=yaws[agent][1], handbrake=yaws[agent][2])
            
            actions[agent] = action
            
        return actions