"""
Cài đặt các thuật toán học tăng cường đối chuẩn:
- Model-Free: Q-Learning
- Model-Based: Dyna-Q với số bước lập kế hoạch (planning steps) có thể tùy chỉnh
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from src.envs.tabular_mdp import TabularMDP


class QLearningAgent:
    """Tác tử Model-Free Q-Learning."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((num_states, num_actions), dtype=np.float64)

    def select_action(self, s: int, rng: np.random.Generator) -> int:
        """Lựa chọn hành động theo chính sách epsilon-greedy."""
        if rng.random() < self.epsilon:
            return int(rng.integers(self.num_actions))
        return int(np.argmax(self.Q[s]))

    def update(self, s: int, a: int, r: float, next_s: int, done: bool):
        """Cập nhật giá trị Q theo phương trình Bellman một bước."""
        target = r if done else r + self.gamma * np.max(self.Q[next_s])
        self.Q[s, a] += self.lr * (target - self.Q[s, a])


class DynaQAgent(QLearningAgent):
    """
    Tác tử Model-Based Dyna-Q:
    Kết hợp cập nhật trực tiếp từ môi trường với N bước lập kế hoạch (planning)
    từ mô hình World Model được học hoặc bị tiêm nhiễu.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        planning_steps: int = 5,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        noise_level: float = 0.0,
    ):
        super().__init__(num_states, num_actions, lr, gamma, epsilon)
        self.planning_steps = planning_steps
        self.noise_level = noise_level
        self.model: Dict[Tuple[int, int], Tuple[float, int, bool]] = {}
        self.observed_states: List[int] = []
        self.observed_actions: Dict[int, List[int]] = {}

    def update(self, s: int, a: int, r: float, next_s: int, done: bool, rng: Optional[np.random.Generator] = None):
        """Cập nhật trực tiếp và thực hiện vòng lập kế hoạch trong mô hình."""
        if rng is None:
            rng = np.random.default_rng()

        # 1. Cập nhật trực tiếp từ tương tác thực tế
        super().update(s, a, r, next_s, done)

        # 2. Cập nhật bộ nhớ mô hình World Model
        self.model[(s, a)] = (r, next_s, done)
        if s not in self.observed_actions:
            self.observed_states.append(s)
            self.observed_actions[s] = []
        if a not in self.observed_actions[s]:
            self.observed_actions[s].append(a)

        # 3. Lập kế hoạch giả lập trong World Model
        for _ in range(self.planning_steps):
            if not self.observed_states:
                break
            rand_s = int(rng.choice(self.observed_states))
            rand_a = int(rng.choice(self.observed_actions[rand_s]))
            sim_r, sim_next_s, sim_done = self.model[(rand_s, rand_a)]

            # Tiêm nhiễu vào mô hình nếu có cấu hình noise_level > 0
            if self.noise_level > 0 and rng.random() < self.noise_level:
                sim_next_s = int(rng.integers(self.num_states))

            sim_target = sim_r if sim_done else sim_r + self.gamma * np.max(self.Q[sim_next_s])
            self.Q[rand_s, rand_a] += self.lr * (sim_target - self.Q[rand_s, rand_a])


def train_agent(
    env: TabularMDP,
    agent: QLearningAgent,
    episodes: int = 200,
    max_steps_per_episode: int = 100,
    seed: int = 42,
) -> List[float]:
    """Hàm huấn luyện tác tử trên Tabular MDP và ghi nhận tổng phần thưởng mỗi tập."""
    rng = np.random.default_rng(seed)
    planning_rng = np.random.default_rng(seed + 100000)
    returns = []

    for _ in range(episodes):
        # Chọn trạng thái khởi đầu theo phân phối mu_0
        s = int(rng.choice(env.num_states, p=env.initial_dist))
        ep_return = 0.0

        for _ in range(max_steps_per_episode):
            a = agent.select_action(s, rng)

            # Lấy mẫu trạng thái kế tiếp từ ma trận P
            p_trans = env.transitions[s, a, :]
            next_s = int(rng.choice(env.num_states, p=p_trans))
            r = float(env.rewards[s, a])

            # Kiểm tra absorbing goal
            done = bool(all(np.isclose(env.transitions[next_s, aa, next_s], 1.0)
                            and np.isclose(env.rewards[next_s, aa], 0.0)
                            for aa in range(env.num_actions)))

            if isinstance(agent, DynaQAgent):
                agent.update(s, a, r, next_s, done, planning_rng)
            else:
                agent.update(s, a, r, next_s, done)

            ep_return += r
            s = next_s
            if done:
                break

        returns.append(ep_return)

    return returns
