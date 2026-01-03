import random

from libretro.api.input.joypad import JoypadState
from strategies.base import Strategy


class RandomStrategy(Strategy):
    def choose_action(self, t: int, frame) -> JoypadState:
        """
        Very naive random strategy:
        - Hold RIGHT 80% of the time to keep moving
        - Randomly press B with low probability
        """
        hold_right = random.random() < 0.8
        jump = random.random() < 0.05
        return JoypadState(right=hold_right, b=jump)

