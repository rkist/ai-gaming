import cv2
import numpy as np
from libretro.api.input.joypad import JoypadState
from strategies.base import Strategy


class SimpleLearningStrategy(Strategy):
    """
    Frame-driven policy with a tiny online-learned neural net.

    - Uses the entire RGBA frame (downsampled grayscale) as input.
    - Outputs independent button probabilities (Bernoulli) for each control.
    - Learns continuously with a lightweight policy-gradient update that
      rewards estimated motion to the right, derived directly from the
      frame-to-frame translation (no HUD parsing).
    """

    def __init__(
        self,
        downsample_size: tuple[int, int] = (28, 32),
        hidden_dim: int = 64,
        learning_rate: float = 1e-4,
        reward_scale: float = 1.0,
    ):
        self.h, self.w = downsample_size
        self.input_dim = self.h * self.w
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.reward_scale = reward_scale

        rng = np.random.default_rng(42)
        self.W1 = rng.normal(0, 0.05, size=(self.input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0, 0.05, size=(hidden_dim, 8))
        self.b2 = np.zeros(8, dtype=np.float32)

        self.prev_frame_small: np.ndarray | None = None
        self.rng = rng

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Convert RGBA frame to downsampled grayscale in [0,1]."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
        small = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return small.astype(np.float32) / 255.0

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One hidden-layer MLP; returns hidden activations and button probs."""
        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(z1, 0.0)
        logits = h1 @ self.W2 + self.b2
        probs = 1.0 / (1.0 + np.exp(-logits))
        return h1, logits, probs

    def _estimate_reward(self, prev: np.ndarray, curr: np.ndarray) -> float:
        """
        Estimate progress using phase correlation shift.
        Positive dx = motion to the right.
        """
        shift, _ = cv2.phaseCorrelate(prev.astype(np.float32), curr.astype(np.float32))
        dx, dy = shift
        # Encourage rightward movement; penalize large vertical jitter slightly
        reward = float(dx - 0.1 * abs(dy))
        return reward * self.reward_scale

    def _policy_update(
        self,
        x: np.ndarray,
        h1: np.ndarray,
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

        delta2 = advantage * reward
        grad_W2 = np.outer(h1, delta2)
        grad_b2 = delta2

        dh1 = self.W2 @ delta2
        dz1 = dh1 * (h1 > 0)
        grad_W1 = np.outer(x, dz1)
        grad_b1 = dz1

        self.W1 += self.lr * grad_W1
        self.b1 += self.lr * grad_b1
        self.W2 += self.lr * grad_W2
        self.b2 += self.lr * grad_b2

    def choose_action(self, t: int, frame) -> JoypadState:
        """
        Decide controls from the full frame and update the policy online.

        The returned buttons correspond to:
          left, right, up, down, B (jump), A (spin), Y, X.
        """
        small = self._preprocess(frame)
        x = small.flatten()

        h1, _, probs = self._forward(x)

        actions = (self.rng.random(probs.shape) < probs).astype(np.float32)
        reward = 0.0
        if self.prev_frame_small is not None:
            reward = self._estimate_reward(self.prev_frame_small, small)
            self._policy_update(x, h1, probs, actions, reward)

        self.prev_frame_small = small

        left, right, up, down, b, a, y, x_btn = actions.astype(bool)
        return JoypadState(
            left=left,
            right=right,
            up=up,
            down=down,
            b=b,
            a=a,
            y=y,
            x=x_btn,
        )

