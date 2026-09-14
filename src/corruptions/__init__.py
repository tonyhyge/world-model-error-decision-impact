"""
Mô-đun tiêm sai số có kiểm soát vào World Model.
"""
from src.corruptions.injector import LocalizedError, CorruptedMDP, inject_perturbation
from src.corruptions.matched_pairs import generate_matched_error_pair

__all__ = [
    "LocalizedError",
    "CorruptedMDP",
    "inject_perturbation",
    "generate_matched_error_pair",
]
