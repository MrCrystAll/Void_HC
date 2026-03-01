"""Module for the Flip routine"""

from collections.abc import Hashable
from typing import Any

import numpy as np
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import JUMP, PITCH, YAW

from void_hc.api.hc_typing import HCMachineAction
from void_hc.flip.flip_primitives import FlipAction, FlipState, HCMachineFlipAction
from void_hc.flip.flip_state_machine import FlipStateMachine
from void_hc.api.routine import Routine
from void_hc.api.target_shared_info_provider import TARGET_HEADER


class FlipRoutine(
    Routine[Hashable, HCMachineFlipAction, np.ndarray, FlipStateMachine, GameState]
):
    """The routine allowing the bot to flip based on a target"""

    def __init__(self) -> None:
        self.flip_state_machine = FlipStateMachine()

    def apply_outputs(
        self,
        actions: dict[Hashable, HCMachineFlipAction],
        current_output: dict[Hashable, np.ndarray],
        state: GameState,
        shared_info: dict[str, Any],
    ) -> dict[Hashable, np.ndarray]:
        self.flip_state_machine.step(
            {k: v.action for k, v in actions.items()}, state, shared_info
        )

        for agent, action in actions.items():
            match action.action:
                case FlipAction.JUMP:
                    current_output[agent] = self._create_jump_action(
                        current_output[agent], agent
                    )
                case FlipAction.FLIP:
                    current_output[agent] = self._create_flip_action(
                        current_output[agent],
                        agent,
                        action,
                        state,
                        shared_info,
                    )

        return current_output

    def _create_jump_action(self, output: np.ndarray, agent: Hashable) -> np.ndarray:
        _state = self.flip_state_machine.states[agent]

        match _state:
            case FlipState.ON_GROUND | FlipState.IS_JUMPING:
                output[:, JUMP] = 1
                output[:, YAW] = 0
                output[:, PITCH] = 0

        return output

    def _create_flip_action(
        self,
        output: np.ndarray,
        agent: Hashable,
        action: HCMachineFlipAction,
        state: GameState,
        shared_info: dict[str, Any],
    ) -> np.ndarray:
        _state = self.flip_state_machine.states[agent]

        _car = state.cars[agent]

        _target: np.ndarray = shared_info[TARGET_HEADER][agent]["steer"]

        _direction = _target - _car.physics.position
        _direction /= np.linalg.norm(_direction)
        
        _direction = _car.physics.rotation_mtx.T.dot(_direction)
        _direction = _direction[:2]

        action.direction = _direction

        _pitch_input = abs(action.direction[0]) * -np.sign(action.direction[0])
        _yaw_input = abs(action.direction[1]) * np.sign(action.direction[1])

        match _state:
            case FlipState.IS_FLIPPING:
                output[-1, [JUMP, PITCH, YAW]] = [1, _pitch_input, _yaw_input]
                output[:-3, JUMP] = 1

            case FlipState.IS_JUMPING:
                _agent_to_ball = state.ball.position - _car.physics.position
                _agent_to_ball /= np.linalg.norm(_agent_to_ball)

                output[:, JUMP] = 1
                output[-1, PITCH] = _pitch_input
                output[-1, YAW] = _yaw_input

                output[0, JUMP] = 0

        return output

    @property
    def state_machine(self) -> FlipStateMachine:
        return self.flip_state_machine

    def get_sub_action_from_top_action(
        self, top_action: HCMachineAction
    ) -> HCMachineFlipAction:
        _action = top_action["flip"]
        if not isinstance(_action, HCMachineFlipAction):
            raise ValueError(
                f'Expected {HCMachineFlipAction.__name__} at the "flip" slot but got {type(_action).__name__}'
            )
        return _action
