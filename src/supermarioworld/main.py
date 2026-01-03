import time
import queue
from pathlib import Path

import numpy as np
import cv2
import tomllib

# libretro.py imports (names can vary slightly by version)
from libretro import SessionBuilder
from libretro.api.input.joypad import JoypadState
from libretro.drivers.input.iterable import IterableInputDriver
from libretro.drivers.video.software.array import ArrayVideoDriver

CORE = Path("cores/snes9x_libretro.dylib")
ROM  = Path("roms/supermarioworld.smc")

actions = queue.Queue(maxsize=2)  # small so we don't build latency
video = ArrayVideoDriver()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.toml"


def load_config():
    """
    Load optional config from config.toml.
    Supported keys under [supermarioworld]:
      watch (bool): show window with frames
      window_name (str): window title when watch is true
      throttle_fps (int|float|null): cap loop rate if set
      window_width (int|null): resize window width; preserves aspect if height missing
      window_height (int|null): resize window height; preserves aspect if width missing
    """
    cfg = {
        "watch": True,
        "window_name": "Super Mario World (libretro)",
        "throttle_fps": None,
        "window_width": None,
        "window_height": None,
    }
    try:
        with CONFIG_PATH.open("rb") as f:
            data = tomllib.load(f)
        section = data.get("supermarioworld", data)
        for key in cfg:
            if key in section:
                cfg[key] = section[key]
    except FileNotFoundError:
        pass
    except Exception:
        # ignore malformed config; keep defaults
        pass
    return cfg


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
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                # Resize if configured
                target_w = cfg["window_width"]
                target_h = cfg["window_height"]
                if target_w or target_h:
                    h, w, _ = frame_bgr.shape
                    if target_w and target_h:
                        new_w, new_h = target_w, target_h
                    elif target_w:
                        new_w = target_w
                        new_h = round(h * (target_w / w))
                    else:
                        new_h = target_h
                        new_w = round(w * (target_h / h))
                    frame_bgr = cv2.resize(frame_bgr, (int(new_w), int(new_h)), interpolation=cv2.INTER_AREA)
                cv2.imshow(cfg["window_name"], frame_bgr)

                # Exit if window closed or 'q' pressed
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(
                    cfg["window_name"], cv2.WND_PROP_VISIBLE
                ) < 1:
                    running = False

            # throttle if you want (otherwise it runs as fast as possible)
            if cfg["throttle_fps"]:
                time.sleep(1 / cfg["throttle_fps"])
    except KeyboardInterrupt:
        pass
    finally:
        if cfg["watch"]:
            cv2.destroyAllWindows()