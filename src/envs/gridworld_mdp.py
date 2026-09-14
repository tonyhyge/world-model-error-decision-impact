"""
Môi trường Choice GridWorld 2D với rào cản và các tuyến cạnh tranh.
"""
from typing import List, Optional, Tuple
import numpy as np
from src.envs.tabular_mdp import TabularMDP


class ChoiceGridWorldMDP(TabularMDP):
    """
    GridWorld 2D ngẫu nhiên với 4 hành động rời rạc: UP(0), DOWN(1), LEFT(2), RIGHT(3).
    Có trạng thái đích (Goal), các ô nguy hiểm (Hazards) gây phạt nặng,
    và cơ chế trượt ngẫu nhiên (slippage probability).
    """

    def __init__(
        self,
        height: int = 5,
        width: int = 5,
        transitions: Optional[np.ndarray] = None,
        rewards: Optional[np.ndarray] = None,
        initial_dist: Optional[np.ndarray] = None,
        gamma: float = 0.95,
        goal_pos: Tuple[int, int] = (4, 4),
        hazards: Optional[List[Tuple[int, int]]] = None,
    ):
        self.height = height
        self.width = width
        self.goal_pos = goal_pos
        self.hazards = hazards or []
        self.goal_state = self.pos_to_state(goal_pos[0], goal_pos[1])

        super().__init__(
            num_states=height * width,
            num_actions=4,
            transitions=transitions,
            rewards=rewards,
            gamma=gamma,
            initial_dist=initial_dist,
            state_names=[f"({r},{c})" for r in range(height) for c in range(width)],
            action_names=["UP", "DOWN", "LEFT", "RIGHT"],
        )

    def pos_to_state(self, r: int, c: int) -> int:
        return r * self.width + c

    def state_to_pos(self, s: int) -> Tuple[int, int]:
        return (s // self.width, s % self.width)


def make_choice_gridworld(
    height: int = 5,
    width: int = 5,
    p_succ: float = 0.85,
    goal_pos: Optional[Tuple[int, int]] = None,
    hazards: Optional[List[Tuple[int, int]]] = None,
    hazard_penalty: float = -5.0,
    goal_reward: float = 10.0,
    step_cost: float = -0.1,
    gamma: float = 0.95,
) -> ChoiceGridWorldMDP:
    """
    Khởi tạo Choice GridWorld với xác suất chuyển trạng thái và ma trận phần thưởng xác định.
    """
    if goal_pos is None:
        goal_pos = (height - 1, width - 1)

    if hazards is None:
        # Đặt các hazard ở cột giữa nếu đủ diện tích
        mid_c = width // 2
        hazards = [(r, mid_c) for r in range(1, height - 1) if (r, mid_c) != goal_pos and (r, mid_c) != (0, 0)]
    else:
        hazards = [(r, c) for r, c in hazards if r < height and c < width and (r, c) != goal_pos and (r, c) != (0, 0)]

    num_states = height * width
    num_actions = 4  # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
    P = np.zeros((num_states, num_actions, num_states), dtype=np.float64)
    R = np.full((num_states, num_actions), step_cost, dtype=np.float64)

    # Định nghĩa hướng di chuyển
    moves = {
        0: (-1, 0),  # UP
        1: (1, 0),   # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1),   # RIGHT
    }
    perp_moves = {
        0: [2, 3],  # UP trượt sang LEFT hoặc RIGHT
        1: [2, 3],  # DOWN trượt sang LEFT hoặc RIGHT
        2: [0, 1],  # LEFT trượt sang UP hoặc DOWN
        3: [0, 1],  # RIGHT trượt sang UP hoặc DOWN
    }

    goal_s = goal_pos[0] * width + goal_pos[1]
    hazard_states = {r * width + c for r, c in hazards}

    for r in range(height):
        for c in range(width):
            s = r * width + c

            # Trạng thái đích là absorbing state
            if s == goal_s:
                for a in range(num_actions):
                    P[s, a, s] = 1.0
                    R[s, a] = 0.0
                continue

            for a in range(num_actions):
                # Xác suất đi thẳng
                dr, dc = moves[a]
                nr = max(0, min(height - 1, r + dr))
                nc = max(0, min(width - 1, c + dc))
                s_next = nr * width + nc
                P[s, a, s_next] += p_succ

                # Xác suất trượt trực giao
                slip_prob = (1.0 - p_succ) / 2.0
                for slip_a in perp_moves[a]:
                    sdr, sdc = moves[slip_a]
                    snr = max(0, min(height - 1, r + sdr))
                    snc = max(0, min(width - 1, c + sdc))
                    s_slip = snr * width + snc
                    P[s, a, s_slip] += slip_prob

                # Tính phần thưởng kỳ vọng một bước
                exp_reward = step_cost
                for dest in range(num_states):
                    if P[s, a, dest] > 0:
                        if dest == goal_s:
                            exp_reward += P[s, a, dest] * goal_reward
                        elif dest in hazard_states:
                            exp_reward += P[s, a, dest] * hazard_penalty
                R[s, a] = exp_reward

    # Phân phối khởi đầu: bắt đầu tại góc trên bên trái (0, 0)
    mu_0 = np.zeros(num_states, dtype=np.float64)
    mu_0[0] = 1.0

    env = ChoiceGridWorldMDP(
        height=height,
        width=width,
        transitions=P,
        rewards=R,
        initial_dist=mu_0,
        gamma=gamma,
        goal_pos=goal_pos,
        hazards=hazards,
    )
    env.validate()
    return env
