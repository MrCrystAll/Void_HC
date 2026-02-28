"""The primitives of the ATBA routine"""

from enum import IntEnum, auto

from common.hc_typing import HCAction, HCActionEnum


class ATBAAction(HCActionEnum):
    """The actions of the ATBA state machine
    """
    
    # Locks the agent onto the ball
    GO_TO_BALL = 0
    
    # Locks the agent away from the ball
    GO_AWAY_FROM_BALL = auto()
    
    # Does nothing, it's a way to keep the lock on whatever you want without spamming the corresponding action
    NEUTRAL = auto()

    # A util variable that holds the number of actions
    N_ACTIONS = auto()


class ATBAState(IntEnum):
    """All the states of the ATBA machine state"""
    
    # The agent is targetting the ball
    LOCK_ON_BALL = 0
    
    # The agant is fleeing the ball
    LOCK_OFF_BALL = auto()


class HCMachineATBAAction(HCAction[ATBAAction]):
    """Mostly a placeholder class to respect the primitives i have for every routine
    This class is useful if i need to give data to the routine
    """
