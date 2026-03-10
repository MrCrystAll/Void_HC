from collections.abc import Hashable
from typing import Any

from rlgym.rocket_league.api.game_state import GameState

from void_hc.common.get_ball_pred import get_ball_prediction
from void_hc.common.pid.base.steer_pid import SteerPID


class SteerToBallPID(SteerPID):
    def __init__(
        self,
        p: float = 0,
        i: float = 0,
        d: float = 0,
        use_ball_pred: bool = False,
        ball_pred_step_seconds: float = 1,
    ) -> None:
        super().__init__(p, i, d)
        self._use_ball_pred = use_ball_pred
        self.ball_pred_step_seconds = ball_pred_step_seconds

    def get_targets(
        self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]
    ) -> dict[Hashable, Any]:
        if not self._use_ball_pred:
            return {agent: state.ball.position for agent in agents}

        assert shared_info["ball_prediction"] is not None, (
            "Ball prediction provider is not given to the environment while the PID requires it, please update your config"
        )
        _results = {}

        for agent in agents:
            _results[agent] = get_ball_prediction(
                agent, self.ball_pred_step_seconds, state, shared_info
            )

        return _results
