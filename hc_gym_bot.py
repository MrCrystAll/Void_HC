from collections.abc import Hashable
from typing import Any

from rlgym.rocket_league.api import GameState

from pid import PitchPID, RollPID, SteerPID


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


class HCGymBot:
    def __init__(self) -> None:
        self.steer_towards_ball_pid: SteerPID = SteerPID(
            3, 0.1, 0.2, boost_threshold=0.05, handbrake_threshold=0.6
        )
        self.in_air_steer_towards_ball_pid: SteerPID = SteerPID(
            0.4, 0.01, 0.1, boost_threshold=0.05, handbrake_threshold=2
        )
        self.pitch_towards_ball_pid: PitchPID = PitchPID(1, 0, 0.3)
        self.roll_stabilization_pid: RollPID = RollPID(0.5, 0.1, 0.5)

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

    def get_output(
        self, agents: list[Hashable], game_state: GameState, shared_info: dict[str, Any]
    ) -> dict[Hashable, GymActionReturn]:
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
                throttle=1,
                roll=rolls[agent],
                pitch=pitches[agent],
                yaw=yaws[agent][0],
                steer=yaws[agent][0],
                boost=yaws[agent][1],
                handbrake=yaws[agent][2],
            )

            actions[agent] = action

        return actions
