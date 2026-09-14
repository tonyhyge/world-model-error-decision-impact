"""
Định nghĩa cấu trúc Markov Decision Process (MDP) dạng bảng rời rạc.
"""
from typing import List, Optional
import numpy as np


class TabularMDP:
    """
    Biểu diễn Markov Decision Process hữu hạn (S, A, P, R, gamma, mu_0).
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        transitions: Optional[np.ndarray] = None,
        rewards: Optional[np.ndarray] = None,
        gamma: float = 0.95,
        initial_dist: Optional[np.ndarray] = None,
        state_names: Optional[List[str]] = None,
        action_names: Optional[List[str]] = None,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma

        if transitions is not None:
            assert transitions.shape == (num_states, num_actions, num_states), (
                f"Kích thước ma trận chuyển đổi không khớp: {transitions.shape} vs ({num_states}, {num_actions}, {num_states})"
            )
            self.transitions = transitions.astype(np.float64)
        else:
            self.transitions = np.zeros((num_states, num_actions, num_states), dtype=np.float64)

        if rewards is not None:
            assert rewards.shape == (num_states, num_actions), (
                f"Kích thước hàm phần thưởng không khớp: {rewards.shape} vs ({num_states}, {num_actions})"
            )
            self.rewards = rewards.astype(np.float64)
        else:
            self.rewards = np.zeros((num_states, num_actions), dtype=np.float64)

        if initial_dist is not None:
            assert len(initial_dist) == num_states, "Kích thước phân phối khởi tạo không khớp"
            self.initial_dist = initial_dist.astype(np.float64) / np.sum(initial_dist)
        else:
            self.initial_dist = np.ones(num_states, dtype=np.float64) / num_states

        self.state_names = state_names or [f"s_{i}" for i in range(num_states)]
        self.action_names = action_names or [f"a_{j}" for j in range(num_actions)]

    def validate(self, tol: float = 1e-6) -> bool:
        """Kiểm tra tính hợp lệ của ma trận xác suất chuyển trạng thái."""
        for s in range(self.num_states):
            for a in range(self.num_actions):
                row_sum = np.sum(self.transitions[s, a, :])
                if not np.isclose(row_sum, 1.0, atol=tol):
                    raise ValueError(f"Tổng xác suất tại (s={s}, a={a}) là {row_sum} != 1.0")
                if np.any(self.transitions[s, a, :] < -tol):
                    raise ValueError(f"Tồn tại xác suất âm tại (s={s}, a={a})")
        return True

    def copy(self) -> "TabularMDP":
        """Tạo bản sao độc lập của MDP."""
        return TabularMDP(
            num_states=self.num_states,
            num_actions=self.num_actions,
            transitions=self.transitions.copy(),
            rewards=self.rewards.copy(),
            gamma=self.gamma,
            initial_dist=self.initial_dist.copy(),
            state_names=list(self.state_names),
            action_names=list(self.action_names),
        )

    def with_transitions(self, new_transitions: np.ndarray) -> "TabularMDP":
        """Tạo MDP mới với ma trận chuyển trạng thái thay thế."""
        clone = self.copy()
        assert new_transitions.shape == self.transitions.shape
        clone.transitions = new_transitions.copy()
        return clone
