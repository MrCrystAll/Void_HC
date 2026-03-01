from typing import Any, Generic

import numpy as np

from rlgym.api import AgentID, StateType

from void_hc.api.hc_typing import HCMachineAction
from void_hc.api.routine import Routine


class RoutineSequencer(Generic[AgentID, StateType]):
    def __init__(
        self,
        *routines: Routine[AgentID, Any, np.ndarray, Any, StateType],
        n_actions: int = 8,
    ) -> None:
        self.routines = routines
        self.n_actions = n_actions

    def get_outputs(
        self,
        actions: dict[AgentID, HCMachineAction],
        state: StateType,
        shared_info: dict[str, Any],
    ) -> dict[AgentID, np.ndarray]:
        _inputs = {k: np.zeros((self.n_actions, 8)) for k in actions.keys()}

        for routine in self.routines:
            _inputs = routine.apply_outputs(
                {
                    k: routine.get_sub_action_from_top_action(v)
                    for k, v in actions.items()
                },
                _inputs,
                state,
                shared_info,
            )

        return _inputs

    def reset(
        self,
        agents: list[AgentID],
        initial_state: StateType,
        shared_info: dict[str, Any],
    ):
        for routine in self.routines:
            routine.reset(agents, initial_state, shared_info)
