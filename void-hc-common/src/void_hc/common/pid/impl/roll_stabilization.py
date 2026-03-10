from collections.abc import Hashable
from typing import Any

import numpy as np
from rlgym.rocket_league.api.game_state import GameState

from void_hc.common.pid.base.roll_pid import RollPID


class RollStabilizationPID(RollPID):
    """A roll PID that tries to stabilize roll to face upwards"""

    def get_targets(
        self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]
    ) -> dict[Hashable, Any]:
        return {agent: np.asarray([0, 0, 1]) for agent in agents}
