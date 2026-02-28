from collections.abc import Hashable
from typing import Any

import numpy as np
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import JUMP, PITCH, YAW

from common.flip.flip_primitives import FlipAction, FlipState, HCMachineFlipAction
from common.flip.flip_state_machine import FlipStateMachine
from common.machine_action import HCMachineAction
from common.routine import Routine
from common.target_shared_info_provider import TARGET_HEADER


class FlipRoutine(Routine[HCMachineAction, np.ndarray, FlipStateMachine]):
    def __init__(self) -> None:
        self.flip_state_machine = FlipStateMachine()
        
    def apply_outputs(self, actions: dict[Hashable, HCMachineAction], current_output: dict[Hashable, np.ndarray], state: GameState, shared_info: dict[str, Any]) -> dict[Hashable, np.ndarray]:
        self.flip_state_machine.step({k: v.flip_action for k, v in actions.items()}, state, shared_info)
        
        print(self.state_machine.states)
        
        for agent, action in actions.items():
            match action.flip_action:
                case FlipAction.JUMP:
                    current_output[agent] = self._create_jump_action(current_output[agent], agent)
                case FlipAction.FLIP:
                    current_output[agent] = self._create_flip_action(current_output[agent], agent, action.flip_action, state, shared_info)
        
        return current_output
    
    def _create_jump_action(self, output: np.ndarray, agent: Hashable) -> np.ndarray:
        _state = self.flip_state_machine.states[agent]
        
        match _state:
            case FlipState.ON_GROUND | FlipState.IS_JUMPING:
                output[:, JUMP] = 1
                output[:, YAW] = 0
                output[:, PITCH] = 0
                
        return output
    
    def _create_flip_action(self, output: np.ndarray, agent: Hashable, action: FlipAction, state: GameState, shared_info: dict[str, Any]) -> np.ndarray:
        _state = self.flip_state_machine.states[agent]
        
        _car = state.cars[agent]
        
        _target = shared_info[TARGET_HEADER][agent]["steer"]
        
        _direction = _target - _car.physics.position
        _direction /= np.linalg.norm(_direction)
        _direction = _direction[:2]
        
        _action = HCMachineFlipAction(action, _direction)
        
        print(_action.direction)
        
        _pitch_input = abs(_action.direction[0]) * -np.sign(_action.direction[0])
        _yaw_input = abs(_action.direction[1]) * np.sign(_action.direction[1])
        
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