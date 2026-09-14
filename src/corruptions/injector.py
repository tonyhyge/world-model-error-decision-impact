"""
Cơ chế tạo sai số chuyển trạng thái cục bộ có kiểm soát lên World Model.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from src.envs.tabular_mdp import TabularMDP
from src.planning.dp import value_iteration, expected_discounted_return


@dataclass
class LocalizedError:
    """Đặc tả sai số cục bộ tại một cặp trạng thái - hành động (s, a)."""
    state: int
    action: int
    delta_p: np.ndarray  # Shape: (num_states,), sum(delta_p) = 0
    error_l1: float      # ||delta_p||_1 = 2 * TV(P, P_hat)


class CorruptedMDP:
    """
    Quản lý mô hình MDP bị sai số (P_hat) so với môi trường thực (P).
    """

    def __init__(self, true_mdp: TabularMDP, corrupted_mdp: TabularMDP, error: LocalizedError):
        self.true_mdp = true_mdp
        self.corrupted_mdp = corrupted_mdp
        self.error = error

        # Tính toán chính sách tối ưu trên môi trường thật và mô hình sai số
        self.v_true, self.q_true, self.pi_true = value_iteration(true_mdp)
        self.v_model, self.q_model, self.pi_model = value_iteration(corrupted_mdp)

    def compute_policy_loss(self) -> float:
        """
        Tính mức độ suy giảm hiệu năng trong môi trường thực:
            Loss = J(pi*) - J(pi_hat)
        """
        j_opt = expected_discounted_return(self.true_mdp, self.pi_true)
        j_model = expected_discounted_return(self.true_mdp, self.pi_model)
        return float(max(0.0, j_opt - j_model))


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Chiếu vector bất kỳ lên đơn diện xác suất (Probability Simplex)."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1.0))[0][-1]
    theta = float(cssv[rho] - 1.0) / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    return w / np.sum(w)


def inject_perturbation(
    mdp: TabularMDP,
    state: int,
    action: int,
    delta_p: np.ndarray,
) -> CorruptedMDP:
    """
    Tiêm vector sai số delta_p vào dòng xác suất chuyển trạng thái P(· | s, a).
    Đảm bảo dòng xác suất mới P_hat vẫn nằm trên đơn diện xác suất chuẩn.
    """
    p_orig = mdp.transitions[state, action, :].copy()
    p_raw = p_orig + delta_p
    p_proj = project_to_simplex(p_raw)

    actual_delta = p_proj - p_orig
    error_l1 = float(np.sum(np.abs(actual_delta)))

    new_transitions = mdp.transitions.copy()
    new_transitions[state, action, :] = p_proj

    corrupted_mdp = mdp.with_transitions(new_transitions)
    corrupted_mdp.validate()

    error = LocalizedError(
        state=state,
        action=action,
        delta_p=actual_delta,
        error_l1=error_l1,
    )
    return CorruptedMDP(true_mdp=mdp, corrupted_mdp=corrupted_mdp, error=error)
