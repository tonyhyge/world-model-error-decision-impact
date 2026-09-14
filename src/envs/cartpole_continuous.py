"""
Môi trường động học liên tục CartPole và tính toán Action Margin dựa trên hàm giá trị tham chiếu thủ công.
"""
from typing import Optional, Tuple
import numpy as np


class CartPoleDynamics:
    """
    Tích phân động học phi tuyến CartPole 4 chiều (x, x_dot, theta, theta_dot).
    Không gian hành động rời rạc: a in {0, 1} (Đẩy sang trái / Đẩy sang phải).
    """

    def __init__(
        self,
        gravity: float = 9.8,
        masscart: float = 1.0,
        masspole: float = 0.1,
        length: float = 0.5,
        force_mag: float = 10.0,
        dt: float = 0.02,
    ):
        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.total_mass = masscart + masspole
        self.length = length
        self.polemass_length = masspole * length
        self.force_mag = force_mag
        self.dt = dt

    def step(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, float, bool]:
        """Tích phân trạng thái tiếp theo từ trạng thái hiện tại s và hành động a."""
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * xacc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * thetaacc

        next_state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
        done = bool(
            x < -2.4 or x > 2.4 or theta < -12 * 2 * np.pi / 360 or theta > 12 * 2 * np.pi / 360
        )
        reward = 1.0  # Gymnasium default: includes the terminal transition.
        return next_state, reward, done


class CartPoleCompetentAgent:
    """
    Hàm chấm điểm tham chiếu thủ công (không phải nghiệm LQR hoặc V*) cho CartPole:
      V(s) = 50.0 / (1.0 + s^T P s)
      Q(s, a) = r(s, a) + gamma * V(step(s, a))
      margin m(s) = |Q(s, 0) - Q(s, 1)|
    """

    def __init__(self, dynamics: Optional[CartPoleDynamics] = None, gamma: float = 0.98):
        self.dynamics = dynamics or CartPoleDynamics()
        self.gamma = gamma
        self.P = np.diag([1.0, 0.5, 10.0, 1.0])

    def evaluate_v(self, state: np.ndarray) -> float:
        cost = float(state.T @ self.P @ state)
        return float(50.0 / (1.0 + cost))

    def evaluate_q(self, state: np.ndarray, action: int) -> float:
        next_s, r, done = self.dynamics.step(state, action)
        if done:
            return float(r)
        return float(r + self.gamma * self.evaluate_v(next_s))

    def get_action_values(self, state: np.ndarray) -> np.ndarray:
        q0 = self.evaluate_q(state, 0)
        q1 = self.evaluate_q(state, 1)
        return np.array([q0, q1], dtype=np.float64)

    def select_action(self, state: np.ndarray) -> int:
        q0 = self.evaluate_q(state, 0)
        q1 = self.evaluate_q(state, 1)
        return 1 if q1 >= q0 else 0

    def compute_margin(self, state: np.ndarray) -> float:
        q0 = self.evaluate_q(state, 0)
        q1 = self.evaluate_q(state, 1)
        return float(abs(q0 - q1))
