from collections.abc import Hashable
from typing import Any

import numpy as np

from common.machine_action import HCMachineAction
from common.routine import Routine

from rlgym.rocket_league.api import GameState


class RoutineSequencer:
    def __init__(
        self, *routines: Routine[HCMachineAction, np.ndarray, Any], n_actions: int = 8
    ) -> None:
        self.routines = routines
        self.n_actions = n_actions

    def get_outputs(
        self,
        actions: dict[Hashable, HCMachineAction],
        state: GameState,
        shared_info: dict[str, Any],
    ) -> dict[Hashable, np.ndarray]:
        _inputs = {k: np.zeros((self.n_actions, 8)) for k in actions.keys()}

        for routine in self.routines:
            _inputs = routine.apply_outputs(actions, _inputs, state, shared_info)

        return _inputs

    def reset(
        self,
        agents: list[Hashable],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ):
        for routine in self.routines:
            routine.reset(agents, initial_state, shared_info)
