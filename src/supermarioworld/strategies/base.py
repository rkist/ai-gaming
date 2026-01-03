from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from libretro.api.input.joypad import JoypadState

from hud import extract_hud as extract_hud_fn


class Strategy(ABC):
    @abstractmethod
    def choose_action(self, t: int, frame) -> JoypadState:  # pragma: no cover - interface
        ...

    def extract_hud(self, frame: np.ndarray) -> dict:
        return extract_hud_fn(frame)

