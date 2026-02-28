from enum import IntEnum
from typing import Generic, TypeVar


MachineStateType = TypeVar("MachineStateType")
MachineActionType = TypeVar("MachineActionType")


class HCActionEnum(IntEnum):
    pass


HCActionType = TypeVar("HCActionType", bound=HCActionEnum)


class HCAction(Generic[HCActionType]):
    def __init__(self, action: HCActionType) -> None:
        self.action = action
