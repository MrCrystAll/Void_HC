"""The routine for ATBA"""

from collections.abc import Hashable
from typing import Any

import numpy as np
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import THROTTLE, YAW, PITCH, STEER, ROLL

from void_hc.api.hc_typing import HCAction
from void_hc.atba.atba_primitives import ATBAState, HCMachineATBAAction
from void_hc.atba.atba_state_machine import ATBAStateMachine
from void_hc.atba.pids import PitchPID, RollPID, SteerPID
from void_hc.api.routine import Routine


class ATBARoutine(
    Routine[Hashable, HCMachineATBAAction, np.ndarray, ATBAStateMachine, GameState]
):
    """The ATBA (At The Ball Always) routine,
    allows the ball to drive towards or away from the ball"""

    def __init__(self) -> None:
        self.atba_state_machine = ATBAStateMachine()

        self.steer_towards_ball_pid: SteerPID = SteerPID(3, 0.1, 0.2)
        self.in_air_steer_towards_ball_pid: SteerPID = SteerPID(0.4, 0.01, 0.1)
        self.pitch_towards_ball_pid: PitchPID = PitchPID(1, 0, 0.3)
        self.roll_stabilization_pid: RollPID = RollPID(0.5, 0.1, 0.5)

    @property
    def state_machine(self) -> ATBAStateMachine:
        return self.atba_state_machine

    def apply_outputs(
        self,
        actions: dict[Hashable, HCMachineATBAAction],
        current_output: dict[Hashable, np.ndarray],
        state: GameState,
        shared_info: dict[str, Any],
    ) -> dict[Hashable, np.ndarray]:
        agents = list(actions.keys())

        self.steer_towards_ball_pid.update_error(agents, state, shared_info)
        self.in_air_steer_towards_ball_pid.update_error(agents, state, shared_info)
        self.pitch_towards_ball_pid.update_error(agents, state, shared_info)
        self.roll_stabilization_pid.update_error(agents, state, shared_info)

        self.atba_state_machine.step(
            {k: v.action for k, v in actions.items()}, state, shared_info
        )

        on_ground_yaws = self.steer_towards_ball_pid.get_output(
            agents, state, shared_info
        )
        in_air_yaws = self.in_air_steer_towards_ball_pid.get_output(
            agents, state, shared_info
        )
        pitches = self.pitch_towards_ball_pid.get_output(agents, state, shared_info)
        rolls = self.roll_stabilization_pid.get_output(agents, state, shared_info)

        for agent in agents:
            if state.cars[agent].on_ground:
                yaws = on_ground_yaws
            else:
                yaws = in_air_yaws

            if self.state_machine.get_state(agent) == ATBAState.LOCK_OFF_BALL:
                yaws[agent] *= -1
                pitches[agent] *= -1

            current_output[agent][:, THROTTLE] = 1
            current_output[agent][:, YAW] = yaws[agent]
            current_output[agent][:, STEER] = yaws[agent]
            current_output[agent][:, PITCH] = pitches[agent]
            current_output[agent][:, ROLL] = rolls[agent]

        return current_output

    def reset(
        self,
        agents: list[Hashable],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ):
        super().reset(agents, initial_state, shared_info)

        self.steer_towards_ball_pid.reset(agents, initial_state, shared_info)
        self.in_air_steer_towards_ball_pid.reset(agents, initial_state, shared_info)
        self.pitch_towards_ball_pid.reset(agents, initial_state, shared_info)
        self.roll_stabilization_pid.reset(agents, initial_state, shared_info)

    def get_sub_action_from_top_action(
        self, top_action: dict[str, HCAction]
    ) -> HCMachineATBAAction:
        _action = top_action["atba"]
        if not isinstance(_action, HCMachineATBAAction):
            raise ValueError(
                f"Expected {HCMachineATBAAction.__name__} "
                + f'at the "atba" slot but got {type(_action).__name__}'
            )
        return _action
