import queue

from libretro.api.input.joypad import JoypadState


def create_actions_queue(maxsize: int) -> queue.Queue:
    return queue.Queue(maxsize=maxsize)


def input_gen(q: queue.Queue):
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
            last = q.get_nowait()
        except queue.Empty:
            pass
        yield last


def choose_action(t: int, frame) -> JoypadState:
    """
    Simple placeholder policy:
    - hold RIGHT always
    - tap B (jump in SMW) every ~45 frames
    Replace with your own model/policy as needed.
    """
    jump = (t % 45) == 0
    return JoypadState(right=True, b=jump)


def enqueue_action(actions_queue: queue.Queue, action: JoypadState) -> None:
    """
    Non-blocking enqueue: drop oldest if the queue is full.
    """
    if actions_queue.full():
        try:
            actions_queue.get_nowait()
        except queue.Empty:
            pass
    actions_queue.put_nowait(action)

