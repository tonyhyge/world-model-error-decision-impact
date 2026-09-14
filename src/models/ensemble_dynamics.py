"""
Mô hình World Model mạng nơ-ron Ensemble (MLP Ensemble Dynamics) cho CartPole.
"""
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SingleDynamicsMLP(nn.Module):
    """Mạng MLP 2 lớp ẩn dự báo độ thay đổi trạng thái delta_s = s' - s."""

    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state: torch.Tensor, action_one_hot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action_one_hot], dim=-1)
        return self.net(x)


class MLPEnsembleDynamics:
    """
    Tập hợp (Ensemble) gồm K mô hình động học độc lập dự báo delta_s = s' - s.
    Cho phép ước lượng cả giá trị kỳ vọng dự báo và độ không chắc chắn (Uncertainty).
    """

    def __init__(
        self,
        num_models: int = 5,
        state_dim: int = 4,
        num_actions: int = 2,
        hidden_dim: int = 64,
        lr: float = 1e-3,
    ):
        self.num_models = num_models
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.models = [
            SingleDynamicsMLP(state_dim, num_actions, hidden_dim) for _ in range(num_models)
        ]
        self.optimizers = [optim.Adam(m.parameters(), lr=lr) for m in self.models]
        self.criterion = nn.MSELoss()

    def train_step(
        self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray
    ) -> float:
        """Thực hiện một bước tối ưu hóa loss MSE cho toàn bộ ensemble."""
        s_t = torch.tensor(states, dtype=torch.float32)
        ns_t = torch.tensor(next_states, dtype=torch.float32)
        delta_target = ns_t - s_t

        # One-hot encoding cho hành động rời rạc
        a_onehot = torch.zeros((len(actions), self.num_actions), dtype=torch.float32)
        for i, a in enumerate(actions):
            a_onehot[i, int(a)] = 1.0

        total_loss = 0.0
        for model, opt in zip(self.models, self.optimizers):
            opt.zero_grad()
            pred_delta = model(s_t, a_onehot)
            loss = self.criterion(pred_delta, delta_target)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        return total_loss / self.num_models

    def predict(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dự báo trạng thái tiếp theo và độ bất định của ensemble:
            Returns:
                pred_next_state (np.ndarray): Shape (state_dim,), trung bình dự báo.
                uncertainty (np.ndarray): Shape (state_dim,), phương sai giữa các mô hình.
        """
        s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        a_onehot = torch.zeros((1, self.num_actions), dtype=torch.float32)
        a_onehot[0, int(action)] = 1.0

        preds = []
        with torch.no_grad():
            for model in self.models:
                delta = model(s_t, a_onehot).squeeze(0).numpy()
                preds.append(state + delta)

        preds = np.array(preds)  # Shape (K, state_dim)
        mean_pred = np.mean(preds, axis=0)
        var_pred = np.var(preds, axis=0)
        return mean_pred, var_pred
