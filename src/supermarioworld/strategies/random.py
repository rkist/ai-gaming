import random

from libretro.api.input.joypad import JoypadState
from strategies.base import Strategy


class RandomStrategy(Strategy):
    """
    Naive but SMW-aware random policy using common controls:
      - D-pad: left/right movement
      - B: jump (hold for a few frames once started)
      - A: spin jump (exclusive with B, also held briefly)
      - Y/X: run/pick-up (held often to build speed)
    """

    def __init__(self):
        self.jump_hold = 0
        self.spin_hold = 0

    def choose_action(self, t: int, frame) -> JoypadState:
        # Movement: bias to move right, occasionally left/neutral
        roll = random.random()
        right = roll < 0.8
        left = 0.1 <= roll < 0.2  # small chance to step left
        up = random.random() < 0.02
        down = random.random() < 0.02

        # Run/pick-up (Y/X): held most of the time to build speed
        run = random.random() < 0.7

        # Jump / spin jump: mutually exclusive, hold for a few frames once started
        if self.jump_hold > 0:
            self.jump_hold -= 1
            b = True
            a = False
        elif self.spin_hold > 0:
            self.spin_hold -= 1
            b = False
            a = True
        else:
            start_jump = random.random() < 0.05
            start_spin = (not start_jump) and (random.random() < 0.02)
            if start_jump:
                self.jump_hold = random.randint(3, 6)
                b = True
                a = False
            elif start_spin:
                self.spin_hold = random.randint(3, 6)
                b = False
                a = True
            else:
                b = False
                a = False

        return JoypadState(
            left=left,
            right=right,
            up=up,
            down=down,
            b=b,   # B = jump/confirm
            a=a,   # A = spin jump
            y=run, # Y = run/pick up / fire
            x=run, # X mirrors Y for run
        )

