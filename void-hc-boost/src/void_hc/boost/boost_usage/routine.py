"""This module contains the boost usage routine"""

from collections.abc import Hashable
from typing import Any

import numpy as np

from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BOOST

from void_hc.api.hc_typing import HCMachineAction
from void_hc.api.routine import Routine

from void_hc.boost.boost_usage.primitives import (
    BoostUsageAction,
    BoostUsageState,
    HCMachineBoostUsageAction,
)
from void_hc.boost.boost_usage.state_machine import BoostUsageStateMachine


class BoostUsageRoutine(
    Routine[
        Hashable,
        HCMachineBoostUsageAction,
        np.ndarray,
        BoostUsageStateMachine,
        GameState,
    ]
):
    """The boost usage routine allows the bot to boost to do stuff"""

    def __init__(self) -> None:
        self.boost_usage_state_machine = BoostUsageStateMachine()

    def apply_outputs(
        self,
        actions: dict[Hashable, HCMachineBoostUsageAction],
        current_output: dict[Hashable, np.ndarray],
        state: GameState,
        shared_info: dict[str, Any],
    ) -> dict[Hashable, np.ndarray]:
        self.state_machine.step(
            {k: v.action for k, v in actions.items()}, state, shared_info
        )

        for agent, action in actions.items():
            if self.state_machine.get_state(agent) == BoostUsageState.EMPTY_BOOST:
                continue

            match action.action:
                case BoostUsageAction.BOOST:
                    current_output[agent][:, BOOST] = 1
                case BoostUsageAction.NO_BOOST:
                    current_output[agent][:, BOOST] = 0

        return current_output

    @property
    def state_machine(self) -> BoostUsageStateMachine:
        return self.boost_usage_state_machine

    def get_sub_action_from_top_action(
        self, top_action: HCMachineAction
    ) -> HCMachineBoostUsageAction:
        _action = top_action["boost"]
        if not isinstance(_action, HCMachineBoostUsageAction):
            raise ValueError(
                f"Expected {HCMachineBoostUsageAction.__name__} "
                + f'at the "boost" slot but got {type(_action).__name__}'
            )
        return _action
