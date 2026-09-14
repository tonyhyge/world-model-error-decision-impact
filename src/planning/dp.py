"""
Thuật toán quy hoạch động chính xác: Value Iteration, Policy Evaluation,
tính toán độ chiếm dụng trạng thái (Occupancy) và kỳ vọng lợi nhuận (Return).
"""
from typing import Tuple, Union
import numpy as np
from src.envs.tabular_mdp import TabularMDP


def to_policy_matrix(policy: Union[np.ndarray, list], num_states: int, num_actions: int) -> np.ndarray:
    """Chuyển đổi mảng hành động 1D hoặc ma trận 2D thành ma trận chính sách chuẩn hóa (S, A)."""
    policy_arr = np.array(policy)
    if policy_arr.ndim == 1:
        assert len(policy_arr) == num_states, f"Độ dài policy {len(policy_arr)} != {num_states}"
        mat = np.zeros((num_states, num_actions), dtype=np.float64)
        for s, a in enumerate(policy_arr):
            mat[s, int(a)] = 1.0
        return mat
    elif policy_arr.ndim == 2:
        assert policy_arr.shape == (num_states, num_actions), f"Hình dạng policy không đúng {policy_arr.shape}"
        row_sums = policy_arr.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return policy_arr / row_sums
    else:
        raise ValueError(f"Số chiều không hỗ trợ: {policy_arr.ndim}")


def value_iteration(
    mdp: TabularMDP, tol: float = 1e-12, max_iter: int = 10000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quy hoạch động Value Iteration tìm giá trị tối ưu V*, Q* và chính sách tối ưu pi*.
    
    Returns:
        V_star (np.ndarray): Shape (num_states,), giá trị trạng thái tối ưu V*(s).
        Q_star (np.ndarray): Shape (num_states, num_actions), giá trị hành động tối ưu Q*(s, a).
        pi_star (np.ndarray): Shape (num_states,), hành động tối ưu a*(s).
    """
    V = np.zeros(mdp.num_states, dtype=np.float64)
    gamma = mdp.gamma
    P = mdp.transitions
    R = mdp.rewards

    for _ in range(max_iter):
        Q = R + gamma * np.einsum("ijk,k->ij", P, V)
        V_new = np.max(Q, axis=1)
        diff = np.max(np.abs(V_new - V))
        V = V_new
        if diff < tol:
            break

    Q_star = R + gamma * np.einsum("ijk,k->ij", P, V)
    pi_star = np.argmax(Q_star, axis=1)
    return V, Q_star, pi_star


def compute_q_from_v(mdp: TabularMDP, V: np.ndarray) -> np.ndarray:
    """Tính Q(s, a) = R(s, a) + gamma * sum_{s'} P(s' | s, a) * V(s')."""
    return mdp.rewards + mdp.gamma * np.einsum("ijk,k->ij", mdp.transitions, V)


def policy_evaluation(mdp: TabularMDP, policy: Union[np.ndarray, list]) -> np.ndarray:
    """
    Đánh giá chính sách bằng cách giải hệ phương trình tuyến tính Bellman:
        V^pi = (I - gamma * P^pi)^{-1} R^pi
    """
    pi_mat = to_policy_matrix(policy, mdp.num_states, mdp.num_actions)
    P_pi = np.einsum("ia,iaj->ij", pi_mat, mdp.transitions)
    R_pi = np.einsum("ia,ia->i", pi_mat, mdp.rewards)

    I = np.eye(mdp.num_states, dtype=np.float64)
    A = I - mdp.gamma * P_pi
    V_pi = np.linalg.solve(A, R_pi)
    return V_pi


def compute_occupancy(
    mdp: TabularMDP, policy: Union[np.ndarray, list]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính độ chiếm dụng trạng thái d^pi(s) và trạng thái-hành động d^pi(s, a) có chiết khấu:
        d^pi(s) = (1 - gamma) * mu_0^T (I - gamma * P^pi)^{-1}
    """
    pi_mat = to_policy_matrix(policy, mdp.num_states, mdp.num_actions)
    P_pi = np.einsum("ia,iaj->ij", pi_mat, mdp.transitions)

    I = np.eye(mdp.num_states, dtype=np.float64)
    A = I - mdp.gamma * P_pi
    d_s = (1.0 - mdp.gamma) * np.linalg.solve(A.T, mdp.initial_dist)
    d_sa = d_s[:, np.newaxis] * pi_mat
    return d_s, d_sa


def expected_discounted_return(mdp: TabularMDP, policy: Union[np.ndarray, list]) -> float:
    """Tính kỳ vọng lợi nhuận có chiết khấu J(pi) = E_{s_0 ~ mu_0} [V^pi(s_0)]."""
    V_pi = policy_evaluation(mdp, policy)
    return float(np.dot(mdp.initial_dist, V_pi))
