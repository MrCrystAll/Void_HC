from common.atba.atba_primitives import ATBAAction
from common.flip.flip_primitives import FlipAction

class HCMachineAction:
    def __init__(self, atba_action: ATBAAction, flip_action: FlipAction) -> None:
        self.atba_action = atba_action
        self.flip_action = flip_action