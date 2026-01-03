import queue

from libretro.api.input.joypad import JoypadState
from strategies.base import Strategy
from strategies.random import RandomStrategy

class ActionManager:
    """
    Encapsulates action policy, queueing, and libretro input generator.
    """

    def __init__(self, maxsize: int, strategy: Strategy | None = None):
        self.queue = queue.Queue(maxsize=maxsize)
        self.strategy: Strategy = strategy or RandomStrategy()

    def choose_action(self, t: int, frame) -> JoypadState:
        return self.strategy.choose_action(t, frame)

    def enqueue_action(self, action: JoypadState) -> None:
        """
        Non-blocking enqueue: drop oldest if the queue is full.
        """
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        self.queue.put_nowait(action)

    def input_gen(self):
        """
        RetroArch-style controller stream: yield one JoypadState per frame.
        If the AI hasn't produced one yet, reuse the last action.
        """
        last = JoypadState()

        # Press START for a bit so the game begins
        for _ in range(30):
            yield JoypadState(start=True)

        while True:
            try:
                last = self.queue.get_nowait()
            except queue.Empty:
                pass
            yield last

