import random

from libretro.api.input.joypad import JoypadState
from strategies.base import Strategy


class SimpleLearningStrategy(Strategy):
    """
    Naive but SMW-aware simple policy using common controls:
      - D-pad: left/right movement
      - B: jump
      - A: spin jump
      - Y/X: run/pick-up
    """

    def __init__(self):
        self.jump_hold = 0
        self.spin_hold = 0

    def choose_action(self, t: int, frame) -> JoypadState:
        right = True
        left = False
        up = False
        down = False
        b = False
        a = False
        y = False
        x = False

        return JoypadState(
            left=left,
            right=right,
            up=up,
            down=down,
            b=b,
            a=a,
            y=y,
            x=x,
        )

