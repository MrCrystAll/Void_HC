from enum import IntEnum, auto

from common.hc_typing import HCAction, HCActionEnum


class ATBAAction(HCActionEnum):
    GO_TO_BALL = 0
    GO_AWAY_FROM_BALL = auto()
    NEUTRAL = auto()
    
    N_ACTIONS = auto()
    
class ATBAState(IntEnum):
    LOCK_ON_BALL = 0
    LOCK_OFF_BALL = auto()
    
    
class HCMachineATBAAction(HCAction[ATBAAction]):
    pass