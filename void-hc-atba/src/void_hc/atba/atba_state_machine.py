"""Module of the ATBA state machine"""

from collections.abc import Hashable
from typing import Any

from rlgym.rocket_league.api import GameState

from void_hc.atba.atba_primitives import ATBAAction, ATBAState
from void_hc.api.state_machine import StateMachine


class ATBAStateMachine(StateMachine[Hashable, ATBAState, ATBAAction, GameState]):
    """The ATBA state machine holds all the states and transitions of the ATBA model"""

    def step(
        self,
        actions: dict[Hashable, ATBAAction],
        state: GameState,
        shared_info: dict[str, Any],
    ):
        for agent, action in actions.items():
            match self._states[agent]:
                case ATBAState.LOCK_ON_BALL:
                    self._update_lock_on_ball(agent, action)
                case ATBAState.LOCK_OFF_BALL:
                    self._update_lock_off_ball(agent, action)

    def reset(
        self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]
    ):
        for agent in agents:
            self.transition(agent, ATBAState.LOCK_ON_BALL)

    def _update_lock_on_ball(self, agent: Hashable, action: ATBAAction):
        match action:
            case ATBAAction.GO_AWAY_FROM_BALL:
                self.transition(agent, ATBAState.LOCK_OFF_BALL)

    def _update_lock_off_ball(self, agent: Hashable, action: ATBAAction):
        match action:
            case ATBAAction.GO_TO_BALL:
                self.transition(agent, ATBAState.LOCK_ON_BALL)
