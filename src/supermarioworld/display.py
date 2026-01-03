from __future__ import annotations

import cv2
import numpy as np


class WindowDisplay:
    """
    Simple OpenCV window wrapper that can resize frames and detect close/quit.
    """

    def __init__(self, window_name: str, width: int | None, height: int | None):
        self.window_name = window_name
        self.target_w = width
        self.target_h = height

    def _resize(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not (self.target_w or self.target_h):
            return frame_bgr
        h, w, _ = frame_bgr.shape
        if self.target_w and self.target_h:
            new_w, new_h = self.target_w, self.target_h
        elif self.target_w:
            new_w = self.target_w
            new_h = round(h * (self.target_w / w))
        else:
            new_h = self.target_h
            new_w = round(w * (self.target_h / h))
        return cv2.resize(frame_bgr, (int(new_w), int(new_h)), interpolation=cv2.INTER_AREA)

    def show(self, frame_rgba: np.ndarray) -> bool:
        """
        Display a frame. Returns False if the window was closed or 'q' pressed.
        """
        frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
        frame_bgr = self._resize(frame_bgr)
        cv2.imshow(self.window_name, frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            return False
        return True

    def close(self):
        cv2.destroyWindow(self.window_name)
        cv2.destroyAllWindows()

