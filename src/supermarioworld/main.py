import time
import queue
from pathlib import Path

import numpy as np

# libretro.py imports (names can vary slightly by version)
from libretro import SessionBuilder
from libretro.api.input.joypad import JoypadState
from libretro.drivers.input.iterable import IterableInputDriver
from libretro.drivers.video.software.array import ArrayVideoDriver

from config import load_config
from display import WindowDisplay

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
    # libretro.py 0.6 pattern-matching fails on PathLike in Python 3.12;
    # pass str paths to avoid the pattern TypeError.
    SessionBuilder.defaults(str(CORE))
    .with_content(str(ROM))
    .with_video(video)
    .with_input(IterableInputDriver(input_gen()))
)

def to_rgba_numpy(shot) -> np.ndarray:
    """
    Convert libretro screenshot to (H, W, 4) uint8 RGBA, respecting pitch.
    """
    buf = np.frombuffer(shot.data, dtype=np.uint8)
    # pitch/stride can vary by libretro driver version; default to width*4 if absent
    pitch = getattr(shot, "pitch", None) or getattr(shot, "stride", None) or (
        shot.width * 4
    )
    rows = buf.reshape(shot.height, pitch)
    rgba = rows[:, : shot.width * 4].reshape(shot.height, shot.width, 4)
    return rgba

cfg = load_config()

display = None
if cfg["watch"]:
    display = WindowDisplay(
        cfg["window_name"], cfg["window_width"], cfg["window_height"]
    )

with builder.build() as sess:
    t = 0
    running = True
    try:
        while running:
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

            # Display the frame in a window if enabled
            if cfg["watch"]:
                running = display.show(frame)
                if not running:
                    running = False

            # throttle if you want (otherwise it runs as fast as possible)
            if cfg["throttle_fps"]:
                time.sleep(1 / cfg["throttle_fps"])
    except KeyboardInterrupt:
        pass
    finally:
        if display:
            display.close()