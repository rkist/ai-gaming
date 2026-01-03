import time
from pathlib import Path

import numpy as np

# libretro.py imports (names can vary slightly by version)
from libretro import SessionBuilder
from libretro.api.input.joypad import JoypadState
from libretro.drivers.input.iterable import IterableInputDriver
from libretro.drivers.video.software.array import ArrayVideoDriver

from config import load_config
from display import WindowDisplay
from actions import (
    choose_action,
    enqueue_action,
    create_actions_queue,
    input_gen,
)

CORE = Path("cores/snes9x_libretro.dylib")
ROM  = Path("roms/supermarioworld.smc")

cfg = load_config()

video = ArrayVideoDriver()
actions_queue = create_actions_queue(cfg["action_queue_maxsize"])

builder = (
    SessionBuilder.defaults(str(CORE))
    .with_content(str(ROM))
    .with_video(video)
    .with_input(IterableInputDriver(input_gen(actions_queue)))
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

            action = choose_action(t, frame)
            enqueue_action(actions_queue, action)

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