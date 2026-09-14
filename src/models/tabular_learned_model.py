"""
World Model dạng bảng ước lượng từ dữ liệu tương tác hữu hạn (Empirical Transition Counting).
"""
from typing import List, Tuple
import numpy as np
from src.envs.tabular_mdp import TabularMDP


class TabularLearnedModel:
    """
    Ước lượng P_hat(s' | s, a) và R_hat(s, a) bằng tần suất đếm kèm Laplace smoothing.
    """

    def __init__(self, num_states: int, num_actions: int, gamma: float = 0.95, alpha: float = 0.01):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha = alpha  # Hệ số làm mượt Laplace (smoothing parameter)

        self.transition_counts = np.zeros((num_states, num_actions, num_states), dtype=np.float64)
        self.reward_sums = np.zeros((num_states, num_actions), dtype=np.float64)
        self.state_action_counts = np.zeros((num_states, num_actions), dtype=np.float64)

    def update(self, s: int, a: int, r: float, next_s: int):
        """Cập nhật dữ liệu từ một bước chuyển trạng thái quan sát được."""
        self.transition_counts[s, a, next_s] += 1.0
        self.reward_sums[s, a] += r
        self.state_action_counts[s, a] += 1.0

    def fit_from_trajectories(self, trajectories: List[List[Tuple[int, int, float, int]]]):
        """Cập nhật từ tập các quỹ đạo kinh nghiệm."""
        for traj in trajectories:
            for s, a, r, next_s in traj:
                self.update(s, a, r, next_s)

    def to_mdp(self) -> TabularMDP:
        """Chuyển đổi các thông số ước lượng thành đối tượng TabularMDP."""
        P_hat = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=np.float64)
        R_hat = np.zeros((self.num_states, self.num_actions), dtype=np.float64)

        for s in range(self.num_states):
            for a in range(self.num_actions):
                n_sa = self.state_action_counts[s, a]
                if n_sa > 0:
                    # Ước lượng xác suất chuẩn tắc kèm Laplace smoothing nhẹ
                    counts = self.transition_counts[s, a, :] + self.alpha
                    P_hat[s, a, :] = counts / np.sum(counts)
                    R_hat[s, a] = self.reward_sums[s, a] / n_sa
                else:
                    # Nếu chưa từng quan sát cặp (s, a), khởi tạo phân phối đều
                    P_hat[s, a, :] = 1.0 / self.num_states
                    R_hat[s, a] = 0.0

        return TabularMDP(
            num_states=self.num_states,
            num_actions=self.num_actions,
            transitions=P_hat,
            rewards=R_hat,
            gamma=self.gamma,
        )
