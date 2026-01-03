import time
import logging
from pathlib import Path

import numpy as np

# libretro.py imports (names can vary slightly by version)
from libretro import SessionBuilder
from libretro.drivers.input.iterable import IterableInputDriver
from libretro.drivers.video.software.array import ArrayVideoDriver

from config import load_config
from display import WindowDisplay
from actions import ActionManager

CORE = Path("cores/snes9x_libretro.dylib")
ROM  = Path("roms/supermarioworld.smc")

cfg = load_config()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

video = ArrayVideoDriver()
actions = ActionManager(cfg["action_queue_maxsize"])

builder = (
    SessionBuilder.defaults(str(CORE))
    .with_content(str(ROM))
    .with_video(video)
    .with_input(IterableInputDriver(actions.input_gen()))
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

logger.info(
    "Starting SMW session strategy=%s queue_max=%s watch=%s throttle_fps=%s",
    type(actions.strategy).__name__,
    cfg["action_queue_maxsize"],
    cfg["watch"],
    cfg["throttle_fps"],
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

            action = actions.choose_action(t, frame)
            actions.enqueue_action(action)

            if t % 60 == 0:
                # logger.info("t=%d action=%s", t, action)
                hud = actions.extract_hud(frame)
                logger.info("hud=%s", hud)

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