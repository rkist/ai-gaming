from pathlib import Path

import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.toml"

DEFAULT_CONFIG = {
    "watch": True,
    "window_name": "Super Mario World (libretro)",
    "throttle_fps": None,
    "window_width": None,
    "window_height": None,
    "action_queue_maxsize": 2,
}


def load_config():
    """
    Load optional config from config.toml.
    Supported keys under [supermarioworld]:
      watch (bool): show window with frames
      window_name (str): window title when watch is true
      throttle_fps (int|float|null): cap loop rate if set
      window_width (int|null): resize window width; preserves aspect if height missing
      window_height (int|null): resize window height; preserves aspect if width missing
      action_queue_maxsize (int): max queued actions; lower keeps latency small
    """
    cfg = DEFAULT_CONFIG.copy()
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

