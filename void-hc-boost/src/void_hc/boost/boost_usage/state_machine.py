"""This is the module of the boost usage state machine"""

from collections.abc import Hashable
from typing import Any

from rlgym.rocket_league.api import GameState

from void_hc.api.state_machine import StateMachine
from void_hc.boost.boost_usage.primitives import BoostUsageAction, BoostUsageState


class BoostUsageStateMachine(
    StateMachine[Hashable, BoostUsageState, BoostUsageAction, GameState]
):
    """This is the boost usage state machine, you can see the schema in the README.md"""

    def reset(
        self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]
    ):
        for agent in agents:
            if state.cars[agent].boost_amount == 0:
                self._states[agent] = BoostUsageState.EMPTY_BOOST
            else:
                self._states[agent] = (
                    BoostUsageState.BOOSTING
                    if state.cars[agent].is_boosting
                    else BoostUsageState.NOT_BOOSTING
                )

    def _update_empty_boost(
        self, agent: Hashable, action: BoostUsageAction, state: GameState
    ):
        if state.cars[agent].boost_amount == 0:
            return

        match action:
            case BoostUsageAction.BOOST:
                self.transition(agent, BoostUsageState.BOOSTING)
            case BoostUsageAction.NO_BOOST:
                self.transition(agent, BoostUsageState.NOT_BOOSTING)

    def _update_not_boosting(
        self, agent: Hashable, action: BoostUsageAction, state: GameState
    ):
        if state.cars[agent].boost_amount == 0:
            self.transition(agent, BoostUsageState.EMPTY_BOOST)

        elif action == BoostUsageAction.BOOST:
            self.transition(agent, BoostUsageState.BOOSTING)

    def _update_boosting(
        self, agent: Hashable, action: BoostUsageAction, state: GameState
    ):
        if state.cars[agent].boost_amount == 0:
            self.transition(agent, BoostUsageState.EMPTY_BOOST)

        elif action == BoostUsageAction.NO_BOOST:
            self.transition(agent, BoostUsageState.BOOSTING)

    def step(
        self,
        actions: dict[Hashable, BoostUsageAction],
        state: GameState,
        shared_info: dict[str, Any],
    ):
        for agent, action in actions.items():
            match self._states[agent]:
                case BoostUsageState.EMPTY_BOOST:
                    self._update_empty_boost(agent, action, state)
                case BoostUsageState.NOT_BOOSTING:
                    self._update_not_boosting(agent, action, state)
                case BoostUsageState.BOOSTING:
                    self._update_boosting(agent, action, state)
