"""The module of the Flip state machine"""

from collections.abc import Hashable
from typing import Any

from rlgym.rocket_league.api import GameState

from void_hc.api.state_machine import StateMachine
from void_hc.flip.flip_primitives import FlipAction, FlipState


class FlipStateMachine(StateMachine[Hashable, FlipState, FlipAction, GameState]):
    """The state machine for the Flip routine, holds all the states and transitions"""

    def reset(
        self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]
    ):
        for agent in agents:
            self._states[agent] = (
                FlipState.ON_GROUND
                if state.cars[agent].on_ground
                else FlipState.HAS_FLIPPED
            )

    def _update_on_ground(self, agent: Hashable, action: FlipAction):
        match action:
            case FlipAction.JUMP:
                self.transition(agent, FlipState.IS_JUMPING)
            case FlipAction.FLIP:
                self.transition(agent, FlipState.IS_FLIPPING)

    def _update_is_jumping(self, agent: Hashable, action: FlipAction):
        match action:
            case FlipAction.JUMP:
                self.transition(agent, FlipState.IS_DOUBLE_JUMPING)
            case FlipAction.FLIP:
                self.transition(agent, FlipState.IS_FLIPPING)

    def _update_is_flipping(self, agent: Hashable, state: GameState):
        _car = state.cars[agent]

        if not _car.is_flipping:
            self.transition(agent, FlipState.HAS_FLIPPED)

    def _update_has_flipped(self, agent: Hashable, state: GameState):
        _car = state.cars[agent]

        if _car.on_ground:
            self.transition(agent, FlipState.ON_GROUND)

    def _update_is_double_jumping(self, agent: Hashable, state: GameState):
        _car = state.cars[agent]

        if _car.on_ground:
            self.transition(agent, FlipState.ON_GROUND)

    def step(
        self,
        actions: dict[Hashable, FlipAction],
        state: GameState,
        shared_info: dict[str, Any],
    ):
        for agent, action in actions.items():
            match self._states[agent]:
                case FlipState.ON_GROUND:
                    self._update_on_ground(agent, action)
                case FlipState.IS_JUMPING:
                    self._update_is_jumping(agent, action)
                case FlipState.IS_FLIPPING:
                    self._update_is_flipping(agent, state)
                case FlipState.HAS_FLIPPED:
                    self._update_has_flipped(agent, state)
                case FlipState.IS_DOUBLE_JUMPING:
                    self._update_is_double_jumping(agent, state)
