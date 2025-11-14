from typing import Tuple

import numpy as np

import torch
import torch.nn as nn


"""
TODO: This skeleton code serves as a guideline;
feel free to modify or replace any part of it.
"""


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        """
        TODO: Freely define model layers
        [구현 완료]
        지침서(pdf)의 힌트에 따라 Actor와 Critic을 위한 별도의 MLP 네트워크를 정의합니다.
        
        """
        
        # Actor (Policy) 네트워크: obs -> action의 평균(mu)
        # 256개의 유닛을 가진 2개의 은닉층을 사용합니다.
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim)  # 출력: action의 평균 (mu)
        )
        
        # Critic (Value) 네트워크: obs -> state-value
        # 256개의 유닛을 가진 2개의 은닉층을 사용합니다.
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # 출력: 상태 가치 (value)
        )

        """
        [구현 완료]
        지침서(pdf)의 요구사항에 따라, 행동 분포의 표준 편차(std)를 위한 파라미터를 선언합니다.
        [cite: 148, 149]
        로그(log) 스케일로 파라미터를 생성하면 최적화 과정에서 값이 음수가 되는 것을 방지할 수 있습니다.
        (std = log_std.exp() 이므로 std는 항상 양수)
        """
        # act_dim (2) 크기의 학습 가능한 파라미터를 생성하고 0으로 초기화합니다.
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))

        # --- [추가 코드 시작] ---
        # Actor의 마지막 레이어(self.actor[4])의 편향(bias)을 조작합니다.
        
        # steer(행동 인덱스 0)의 bias는 0.0으로 초기화 (직진 기본)
        nn.init.constant_(self.actor[4].bias[0], 0.0)
        
        # speed(행동 인덱스 1)의 bias는 높은 값(예: 3.0 또는 4.0)으로 초기화
        # max_speed가 얼마인지 모르지만, 3.0 정도로 시작해봅니다.
        nn.init.constant_(self.actor[4].bias[1], 3.0) 
        # --- [추가 코드 끝] ---


    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        TODO: Freely define model outputs
        [구현 완료]
        지침서(pdf)의 요구사항에 따라 
        `algo.py`에서 사용할 `mu`, `value`, `logstd`를 반환합니다.
        """
        
        # Actor 네트워크를 통과시켜 action의 평균(mu)을 계산합니다.
        mu = self.actor(obs)
        
        # Critic 네트워크를 통과시켜 상태 가치(value)를 계산합니다.
        value = self.critic(obs)
        
        # log_std 파라미터를 반환합니다.
        # (algo.py에서 Normal 분포를 만들 때 브로드캐스팅되어 배치 내 모든 샘플에 동일한 std가 적용됩니다.)
        logstd = self.log_std

        return mu, value, logstd