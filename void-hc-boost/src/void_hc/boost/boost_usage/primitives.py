"""The module containing the primitives for the boost usage routine"""

from enum import IntEnum, auto
from void_hc.api.hc_typing import HCAction, HCActionEnum


class BoostUsageState(IntEnum):
    """All the states of the boost usage state tree"""

    # The bot is actively boost
    BOOSTING = 0

    # The bot has no boost
    EMPTY_BOOST = auto()

    # The bot is not boosting despite having boost
    NOT_BOOSTING = auto()


class BoostUsageAction(HCActionEnum):
    """All the boost usage actions"""

    # The player is boosting
    BOOST = 0

    # The player is not boosting
    NO_BOOST = auto()

    # A util variable that holds the number of actions
    N_ACTIONS = auto()


class HCMachineBoostUsageAction(HCAction[BoostUsageAction]):
    """A class that holds the boost action along with
    some potential data if necessary in the future"""
