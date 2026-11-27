"""On-policy first-visit MC Control - tabular, khong phu thuoc env cu the.

Thuat toan (Sutton & Barto, chuong 5, Box 5.3 - khop ma gia muc 5.1.3.1
trong bao cao cua nhom):
1. Sinh tron 1 episode theo chinh sach epsilon-greedy (epsilon-soft).
2. Duyet nguoc episode, tinh loi nhuan tich luy G = r + gamma*G.
3. Voi moi cap (s, a) chi cap nhat o lan XUAT HIEN DAU TIEN
    (first-visit) bang trung binh tang dan: Q <- Q + (G - Q)/N(s,a).
4. Chinh sach epsilon-greedy suy tu Q moi -> tu dong tot dan (GPI).

Agent chi yeu cau: state la tuple hashable, action la so nguyen,
valid_actions duoc truyen vao moi buoc. Moi dac thu cua env nam o
file group_adapter.py.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


class MonteCarloAgent:
    """Tabular on-policy first-visit MC Control, cai dat tu dau."""

    def __init__(
        self,
        number_of_actions: int,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        minimum_epsilon: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ) -> None:
        # So hanh dong (do adapter quyet dinh, o day = 13: 12 cap xe-ca + cho).
        self.number_of_actions = number_of_actions
        # He so chiet khau gamma (bao cao cua nhom: 0.95).
        self.discount_factor = discount_factor
        # Epsilon hien tai: xac suat tham do ngau nhien.
        self.epsilon = epsilon
        # Epsilon khong giam duoi nguong nay.
        self.minimum_epsilon = minimum_epsilon
        # He so giam epsilon sau moi episode.
        self.epsilon_decay = epsilon_decay
        # Bo sinh so ngau nhien rieng (tai lap duoc ket qua).
        self.random_generator = np.random.default_rng(seed)
        # Q(s, a): tu dong tao vector 0 khi gap state moi.
        self.q_table = defaultdict(
            lambda: np.zeros(self.number_of_actions, dtype=np.float64)
        )
        # N(s, a): so lan cap (s, a) da duoc cap nhat (mau so trung binh).
        self.returns_count = defaultdict(
            lambda: np.zeros(self.number_of_actions, dtype=np.float64)
        )

    def select_action(
        self,
        state: tuple,
        valid_actions: list[int],
        training: bool = True,
    ) -> int:
        """Chon hanh dong epsilon-greedy, CHI trong valid_actions.

        Day chinh la "chinh sach epsilon-mem" trong ma gia cua nhom:
        khong can bang pi rieng, vi pi luon duoc suy tu Q hien tai.
        """
        if not valid_actions:
            raise ValueError("valid_actions cannot be empty.")
        # Khi training, voi xac suat epsilon thi tham do ngau nhien.
        should_explore = (
            training and self.random_generator.random() < self.epsilon
        )
        if should_explore:
            return int(self.random_generator.choice(valid_actions))
        # Khai thac: chon Q lon nhat trong so hanh dong hop le.
        # Dung .get() thay vi [] de KHONG tao muc moi trong q_table
        # khi gap state chua tung thay - nho do Evaluation khong lam
        # thay doi Q-table du chi la them vector 0 (quy chuan Muc 5.2:
        # evaluation khong duoc cap nhat kien thuc duoi bat ky dang nao).
        q_values = self.q_table.get(state)
        if q_values is None:
            q_values = np.zeros(self.number_of_actions, dtype=np.float64)
        best_q_value = max(q_values[action] for action in valid_actions)
        # Pha hoa ngau nhien neu nhieu action cung Q toi da.
        best_actions = [
            action
            for action in valid_actions
            if np.isclose(q_values[action], best_q_value)
        ]
        return int(self.random_generator.choice(best_actions))

    def update_episode(self, episode: list[tuple[tuple, int, float]]) -> None:
        """Cap nhat Q sau khi episode KET THUC (diem khac biet cot loi
        cua MC so voi Q-learning/SARSA: hoc sau ca episode, khong hoc
        ngay tung buoc)."""
        if not episode:
            return
        # Buoc 1: tim buoc xuat hien DAU TIEN cua moi cap (s, a).
        # (Duyet nguoc + tap "da thay" se vo tinh lay lan CUOI CUNG,
        #  nen phai tim first-occurrence rieng - dung dinh nghia chuan.)
        first_occurrence: dict[tuple[tuple, int], int] = {}
        for step_index, (state, action, _reward) in enumerate(episode):
            if (state, action) not in first_occurrence:
                first_occurrence[(state, action)] = step_index
        # Buoc 2: duyet nguoc tinh G = r + gamma*G; chi cap nhat tai
        # buoc first-visit bang trung binh tang dan Q <- Q + (G-Q)/N.
        total_return = 0.0
        for step_index in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[step_index]
            total_return = reward + self.discount_factor * total_return
            if first_occurrence[(state, action)] != step_index:
                continue
            self.returns_count[state][action] += 1
            count = self.returns_count[state][action]
            self.q_table[state][action] += (
                total_return - self.q_table[state][action]
            ) / count

    def decay_exploration(self) -> None:
        """Giam epsilon sau moi episode."""
        self.epsilon = max(
            self.minimum_epsilon,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, file_path: str) -> None:
        """Luu Q-table + bang dem N(s, a) ra file pickle."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model_data = {
            "number_of_actions": self.number_of_actions,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "minimum_epsilon": self.minimum_epsilon,
            "epsilon_decay": self.epsilon_decay,
            "q_table": dict(self.q_table),
            "returns_count": dict(self.returns_count),
        }
        with path.open(mode="wb") as file:
            pickle.dump(model_data, file)

    @classmethod
    def load(cls, file_path: str, seed: int = 42) -> "MonteCarloAgent":
        """Nap lai agent da luu (kem bang dem de hoc tiep)."""
        with Path(file_path).open(mode="rb") as file:
            model_data = pickle.load(file)
        agent = cls(
            number_of_actions=model_data["number_of_actions"],
            discount_factor=model_data["discount_factor"],
            epsilon=model_data["epsilon"],
            minimum_epsilon=model_data["minimum_epsilon"],
            epsilon_decay=model_data["epsilon_decay"],
            seed=seed,
        )
        agent.q_table.update(model_data["q_table"])
        agent.returns_count.update(model_data["returns_count"])
        return agent
