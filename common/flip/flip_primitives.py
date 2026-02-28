"""All the primitive types of the flip routine"""

from enum import IntEnum, auto

import numpy as np

from common.hc_typing import HCAction, HCActionEnum


class FlipState(IntEnum):
    """All the possible states of the Flip state machine"""

    # Self explanatory
    ON_GROUND = 0

    # Active as long as the player is in the air from a single jump
    IS_JUMPING = auto()

    # Active when the player flipped and is yaw/pitch locked
    IS_FLIPPING = auto()

    # Active when the player double jumped / is double jumping
    IS_DOUBLE_JUMPING = auto()

    # Active when the player is free from the yaw/pitch lock but still in air
    HAS_FLIPPED = auto()


class FlipAction(HCActionEnum):
    """All the actions of the Flip state machine"""

    JUMP = 0
    FLIP = auto()
    NEUTRAL = auto()

    N_ACTIONS = auto()


class HCMachineFlipAction(HCAction[FlipAction]):
    """A Flip action containing the direction of a flip
    """
    def __init__(self, flip_action: FlipAction, direction: np.ndarray) -> None:
        super().__init__(flip_action)

        assert direction.shape == (2,), (
            "The direction vector should be a valid 2D vector," + \
                f"received {direction} (Shape: {direction.shape})"
        )

        self.direction = direction
