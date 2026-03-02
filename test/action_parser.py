from collections.abc import Hashable
import random
from typing import Any, Dict, List

import numpy as np
from rlgym.api import ActionParser

from rlgym.rocket_league.api import GameState

from void_hc.api.hc_typing import HCMachineAction
from void_hc.api.routine_sequencer import RoutineSequencer
from void_hc.atba.atba_primitives import ATBAAction, HCMachineATBAAction
from void_hc.atba.atba_routine import ATBARoutine

from void_hc.boost.boost_usage.primitives import BoostUsageAction
from void_hc.boost.boost_usage.primitives import HCMachineBoostUsageAction

from void_hc.boost.boost_usage.routine import BoostUsageRoutine
from void_hc.flip.flip_primitives import HCMachineFlipAction
from void_hc.flip.flip_routine import FlipRoutine
from void_hc.flip.flip_state_machine import FlipAction


class HCBotEnhancedActionParser(
    ActionParser[Hashable, np.ndarray, np.ndarray, GameState, tuple[str, int]]
):
    def __init__(self, tick_skip: int = 8) -> None:
        self.routine_sequencer = RoutineSequencer(
            ATBARoutine(), FlipRoutine(), BoostUsageRoutine(), n_actions=tick_skip
        )
        self._lookup_table = self._make_lookup_table()

    def _make_lookup_table(self) -> np.ndarray:
        actions = []

        for atba_action in range(ATBAAction.N_ACTIONS):
            for flip_action in range(FlipAction.N_ACTIONS):
                for boost_action in range(BoostUsageAction.N_ACTIONS):
                    actions.append(
                        {
                            "atba": HCMachineATBAAction(ATBAAction(atba_action)),
                            "flip": HCMachineFlipAction(FlipAction(flip_action)),
                            "boost": HCMachineBoostUsageAction(
                                BoostUsageAction(boost_action)
                            ),
                        }
                    )

        return np.asarray(actions, dtype=HCMachineAction)

    def reset(
        self,
        agents: List[Hashable],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        self.routine_sequencer.reset(agents, initial_state, shared_info)

    def get_action_space(self, agent: Hashable) -> tuple[str, int]:
        return "discrete", self._lookup_table.shape[0] - 1

    def parse_actions(
        self,
        actions: Dict[Hashable, np.ndarray],
        state: GameState,
        shared_info: Dict[str, Any],
    ) -> Dict[Hashable, np.ndarray]:
        parsed_actions: dict[Hashable, HCMachineAction] = {}

        for agent, action in actions.items():
            # Action can have shape (Ticks, 1) or (Ticks)
            assert len(action.shape) == 1 or (
                len(action.shape) == 2 and action.shape[1] == 1
            )

            if len(action.shape) == 2:
                action = action.squeeze(1)

            parsed_actions[agent] = self._lookup_table[action.squeeze()]

        return self.routine_sequencer.get_outputs(parsed_actions, state, shared_info)

    def sample(self, agents: List[Hashable]):
        results = {}
        for agent in agents:
            _action_space = self.get_action_space(agent)
            results[agent] = np.asarray([random.randint(0, _action_space[1])])
        return results

    def pick(self, agents: List[Hashable], action: int):
        return {agent: np.asarray([action]) for agent in agents}

    def get_actions_with(self, filters: dict[str, tuple[int]]) -> list[int]:
        idxes = []

        for idx, action in enumerate(self._lookup_table):
            _valid_action = True
            for filter_k, filter_vs in filters.items():
                if filter_k not in action.keys():
                    _valid_action = False
                    break

                if action[filter_k].action not in filter_vs:
                    _valid_action = False
                    break

            if _valid_action:
                idxes.append(idx)
        return idxes
