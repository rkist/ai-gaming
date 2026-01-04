import cv2
import numpy as np
from libretro.api.input.joypad import JoypadState
from strategies.base import Strategy


class SimpleLearningStrategy(Strategy):
    """
    Frame-driven policy with a tiny online-learned neural net.

    - Uses the entire RGBA frame (downsampled grayscale) as input.
    - Two hidden layers, then independent button probabilities (Bernoulli)
      for each control.
    - Learns continuously with a lightweight policy-gradient update that
      rewards estimated motion to the right, derived directly from the
      frame-to-frame translation (no HUD parsing).
    """

    def __init__(
        self,
        downsample_size: tuple[int, int] = (28, 32),
        hidden_dim: int = 64,
        hidden_dim2: int = 64,
        learning_rate: float = 1e-4,
        reward_scale: float = 1.0,
    ):
        self.h, self.w = downsample_size
        self.input_dim = self.h * self.w
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.lr = learning_rate
        self.reward_scale = reward_scale

        rng = np.random.default_rng(42)
        self.W1 = rng.normal(0, 0.05, size=(self.input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0, 0.05, size=(hidden_dim, hidden_dim2))
        self.b2 = np.zeros(hidden_dim2, dtype=np.float32)
        self.W3 = rng.normal(0, 0.05, size=(hidden_dim2, 8))
        self.b3 = np.zeros(8, dtype=np.float32)

        self.prev_frame_small: np.ndarray | None = None
        self.rng = rng

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Convert RGBA frame to downsampled grayscale in [0,1]."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
        small = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return small.astype(np.float32) / 255.0

    def _forward(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Two hidden-layer MLP; returns both hidden activations and button probs."""
        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(z2, 0.0)
        logits = h2 @ self.W3 + self.b3
        probs = 1.0 / (1.0 + np.exp(-logits))
        return h1, h2, logits, probs

    def _estimate_reward(self, prev: np.ndarray, curr: np.ndarray) -> float:
        """
        Estimate horizontal progress using dense optical flow, robust to scene cuts.
        Positive dx => rightward motion. Scene-change spikes are nulled.
        """
        prev_f = prev.astype(np.float32)
        curr_f = curr.astype(np.float32)

        # Detect large scene changes (e.g., room transition); neutral reward
        frame_delta = np.mean(np.abs(prev_f - curr_f))
        if frame_delta > 0.2:  # heuristic on normalized grayscale
            return 0.0

        flow = cv2.calcOpticalFlowFarneback(
            prev_f,
            curr_f,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        )

        # Focus on the central region to reduce HUD/border influence
        h, w = flow.shape[:2]
        y0, y1 = int(h * 0.15), int(h * 0.85)
        x0, x1 = int(w * 0.15), int(w * 0.85)
        central_flow = flow[y0:y1, x0:x1]

        dx = float(np.median(central_flow[..., 0]))
        dy = float(np.median(central_flow[..., 1]))

        reward = dx
        
        if abs(reward) > 1.0:
            print(f"large reward detected: reward={reward}")
            print(f"dx={dx}, dy={dy}")

        reward = float(np.clip(reward, -5.0, 5.0))
        return reward * self.reward_scale

    def _policy_update(
        self,
        x: np.ndarray,
        h1: np.ndarray,
        h2: np.ndarray,
        probs: np.ndarray,
        actions: np.ndarray,
        reward: float,
    ) -> None:
        """Simple REINFORCE-style update with independent Bernoulli heads."""
        if reward == 0.0:
            return
        # Clip reward to avoid explosion on unstable estimates
        reward = float(np.clip(reward, -5.0, 5.0))
        advantage = actions - probs  # shape (8,)

        delta3 = advantage * reward  # output layer gradient
        grad_W3 = np.outer(h2, delta3)
        grad_b3 = delta3

        dh2 = self.W3 @ delta3
        dz2 = dh2 * (h2 > 0)
        grad_W2 = np.outer(h1, dz2)
        grad_b2 = dz2

        dh1 = self.W2 @ dz2
        dz1 = dh1 * (h1 > 0)
        grad_W1 = np.outer(x, dz1)
        grad_b1 = dz1

        self.W1 += self.lr * grad_W1
        self.b1 += self.lr * grad_b1
        self.W2 += self.lr * grad_W2
        self.b2 += self.lr * grad_b2
        self.W3 += self.lr * grad_W3
        self.b3 += self.lr * grad_b3

    def choose_action(self, t: int, frame) -> JoypadState:
        """
        Decide controls from the full frame and update the policy online.

        The returned buttons correspond to:
          left, right, up, down, B (jump), A (spin), Y, X.
        """
        small = self._preprocess(frame)
        x = small.flatten()

        h1, h2, _, probs = self._forward(x)

        actions = (self.rng.random(probs.shape) < probs).astype(np.float32)
        reward = 0.0
        if self.prev_frame_small is not None:
            reward = self._estimate_reward(self.prev_frame_small, small)
            self._policy_update(x, h1, h2, probs, actions, reward)
            if t % 100 == 0:
                # Print reward so we can monitor online learning signal
                print(f"t={t} reward={reward:.4f}")

        self.prev_frame_small = small

        left, right, up, down, b_btn, a_btn, y_btn, x_btn = actions.astype(bool)
        return JoypadState(
            left=left,
            right=right,
            up=up,
            down=down,
            b=b_btn,
            a=a_btn,
            y=y_btn,
            x=x_btn,
        )

