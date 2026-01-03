from __future__ import annotations

from abc import ABC, abstractmethod
from libretro.api.input.joypad import JoypadState


class Strategy(ABC):
    @abstractmethod
    def choose_action(self, t: int, frame) -> JoypadState:  # pragma: no cover - interface
        ...

