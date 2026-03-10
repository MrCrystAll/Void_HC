from collections.abc import Hashable
from typing import Any

import numpy as np
from rlgym.rocket_league.api import GameState


def get_ball_prediction(
    agent: Hashable, step_seconds: float, state: GameState, shared_info: dict[str, Any]
) -> np.ndarray:
    """Returns the ball prediction for an agent based on its velocity and distance to the ball

    :param agent: The agent to get the prediction of
    :type agent: Hashable
    :param step_seconds: The step seconds used by the ball prediction provider
    :type step_seconds: float
    :param state: The state the agent is in
    :type state: GameState
    :param shared_info: The shared info of the environment
    :type shared_info: dict[str, Any]
    :return: The ball prediction of the agent
    :rtype: np.ndarray
    """

    assert shared_info["ball_prediction"] is not None, (
        "Ball prediction provider is not given to the environment while the PID requires it, please update your config"
    )

    _ball_pred_slices = shared_info["ball_prediction"]
    _n_slices = len(_ball_pred_slices) / 2

    _car = state.cars[agent]
    _dist_to_ball = state.ball.position - _car.physics.position

    _car_vel = _car.physics.linear_velocity.copy() + state.ball.linear_velocity
    _car_speed = np.linalg.norm(_car_vel)

    if _car_speed != 0:
        _car_vel /= _car_speed

    _vel_to_ball = _car_vel * _dist_to_ball

    _speed_to_ball = np.linalg.norm(_vel_to_ball)
    _time_to_ball = np.linalg.norm(_dist_to_ball)

    if _speed_to_ball != 0:
        _time_to_ball /= _car_speed

    _range = np.arange(
        0,
        _n_slices,
        step_seconds,
        dtype=float,
    )
    _idx = np.nonzero(_range > _time_to_ball)
    _idx = _idx[0][0] - 1 if _idx[0].size > 0 else -1

    return _ball_pred_slices[_idx].position
