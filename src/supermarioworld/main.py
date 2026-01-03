import time
import queue
from pathlib import Path

import numpy as np

# libretro.py imports (names can vary slightly by version)
from libretro import SessionBuilder
from libretro.api.input.joypad import JoypadState
from libretro.drivers.input.iterable import IterableInputDriver
from libretro.drivers.video.software.array import ArrayVideoDriver

CORE = Path("cores/snes9x_libretro.dylib")
ROM  = Path("roms/supermarioworld.smc")

actions = queue.Queue(maxsize=2)  # small so we don't build latency
video = ArrayVideoDriver()

def input_gen():
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
            last = actions.get_nowait()
        except queue.Empty:
            pass
        yield last

builder = (
    SessionBuilder.defaults(CORE)
    .with_content(ROM)
    .with_video(video)
    .with_input(IterableInputDriver(input_gen()))
)

def to_rgba_numpy(shot) -> np.ndarray:
    """
    Convert libretro screenshot to (H, W, 4) uint8 RGBA, respecting pitch.
    """
    buf = np.frombuffer(shot.data, dtype=np.uint8)
    # pitch is bytes per row, might be wider than width*4
    rows = buf.reshape(shot.height, shot.pitch)
    rgba = rows[:, : shot.width * 4].reshape(shot.height, shot.width, 4)
    return rgba

with builder.build() as sess:
    t = 0
    while True:
        sess.run()
        t += 1

        shot = video.screenshot()
        if shot is None:
            continue

        frame = to_rgba_numpy(shot)

        # ---- YOUR AI GOES HERE ----
        # Example policy:
        # - hold RIGHT always
        # - tap B (jump in SMW) every ~45 frames
        jump = (t % 45) == 0

        # SNES mapping in libretro terms:
        # JoypadState(b=...) is SNES B, (a=...) is SNES A, etc.
        action = JoypadState(right=True, b=jump)

        # Non-blocking: if queue is full, drop older action
        if actions.full():
            try:
                actions.get_nowait()
            except queue.Empty:
                pass
        actions.put_nowait(action)

        # throttle if you want (otherwise it runs as fast as possible)
        # time.sleep(1/60)