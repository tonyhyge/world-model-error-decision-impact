"""
Cài đặt các chỉ số đo lường hình học quyết định và phân tích thống kê.
"""
from typing import Optional, Tuple
import numpy as np


def compute_action_margin(Q: np.ndarray, pi_star: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Tính Action Margin thực tế của từng trạng thái:
        m(s) = Q(s, a*(s)) - max_{a != a*(s)} Q(s, a)
    """
    num_states, num_actions = Q.shape
    if pi_star is None:
        pi_star = np.argmax(Q, axis=1)

    margins = np.zeros(num_states, dtype=np.float64)
    for s in range(num_states):
        best_a = pi_star[s]
        other_actions = [a for a in range(num_actions) if a != best_a]
        if len(other_actions) > 0:
            runner_up_val = np.max(Q[s, other_actions])
            margins[s] = Q[s, best_a] - runner_up_val
        else:
            margins[s] = 0.0
    return margins


def compute_runner_up_action(Q: np.ndarray, pi_star: Optional[np.ndarray] = None) -> np.ndarray:
    """Xác định hành động về nhì (runner-up action) a^(2)(s)."""
    num_states, num_actions = Q.shape
    if pi_star is None:
        pi_star = np.argmax(Q, axis=1)

    runner_ups = np.zeros(num_states, dtype=np.int64)
    for s in range(num_states):
        best_a = pi_star[s]
        other_actions = [a for a in range(num_actions) if a != best_a]
        if len(other_actions) > 0:
            best_other_idx = np.argmax(Q[s, other_actions])
            runner_ups[s] = other_actions[best_other_idx]
        else:
            runner_ups[s] = best_a
    return runner_ups


def compute_model_margin(
    Q_model: np.ndarray, pi_true: np.ndarray, runner_up_true: np.ndarray
) -> np.ndarray:
    """
    Tính margin trong World Model đối với cặp (hành động tối ưu thực, hành động đối thủ thực):
        m_hat(s) = Q_hat(s, a*_true(s)) - Q_hat(s, a^(2)_true(s))
    """
    num_states = Q_model.shape[0]
    m_hat = np.zeros(num_states, dtype=np.float64)
    for s in range(num_states):
        m_hat[s] = Q_model[s, pi_true[s]] - Q_model[s, runner_up_true[s]]
    return m_hat


def compute_margin_deformation(m_model: np.ndarray, m_true: np.ndarray) -> np.ndarray:
    """Biến dạng margin Delta m(s) = m_hat(s) - m_true(s)."""
    return m_model - m_true


def compute_boundary_pressure(
    delta_m: np.ndarray, m_true: np.ndarray, eps: float = 1e-4
) -> np.ndarray:
    """
    Tính áp lực ranh giới quyết định (Normalized Boundary Pressure):
        B(s) = - Delta m(s) / (m_true(s) + eps)
    
    Chỉ số mô tả hậu nghiệm cho một đối thủ cố định. Với eps > 0,
    điều kiện đổi dấu chính xác là B > m / (m + eps), không phải B > 1.
    Trong không gian nhiều hành động, đối thủ khác có thể vượt trước.
    """
    return -delta_m / (m_true + eps)


def compute_decision_flip(pi_true: np.ndarray, pi_model: np.ndarray) -> np.ndarray:
    """Chỉ báo lật quyết định Z_flip(s) = 1 nếu pi_hat*(s) != pi*(s), ngược lại 0."""
    return (pi_true != pi_model).astype(np.int64)
