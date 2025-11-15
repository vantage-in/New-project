import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box


"""
TODO: This skeleton code serves as a guideline;
feel free to modify or replace any part of it.
"""


class RCCarEnvTrainWrapper(gym.Wrapper):
    def __init__(self, base_env, max_steer, min_speed, max_speed, time_limit):
        super().__init__(base_env)
        self.base_env = base_env
        self.max_steer = max_steer
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.time_limit = time_limit

        # perform one reset to infer scan size and set spaces
        # (스켈레톤 코드 원본)
        _, _, scan_sample = self.base_env.reset()[0]
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=scan_sample.shape, dtype=np.float32)

        low = np.array([-self.max_steer, self.min_speed], dtype=np.float32)
        high = np.array([self.max_steer, self.max_speed], dtype=np.float32)
        self.action_space = Box(low=low, high=high, shape=(2,), dtype=np.float32)

        self.elapsed_steps = 0

        """
        TODO: Freely define attributes, methods, etc.
        [구현 완료] 
        - 진행 상황(웨이포인트)을 추적하기 위한 변수를 추가합니다.
        """
        self.current_waypoint = 0
        

    def reset(self, **kwargs):
        # init waypoint tracking
        self.elapsed_steps = 0

        obs, info = self.base_env.reset(**kwargs)
        _, _, scan = obs

        """
        TODO: Freely reset attributes, etc.
        [구현 완료]
        - 에피소드 시작 시 웨이포인트 카운터를 리셋합니다.
        """
        self.current_waypoint = 0

        return scan, info

    def step(self, action):
        # [구현 완료] 
        # wrapper.py는 벡터화된 환경(SyncVectorEnv)이 아닌 
        # PPO.learn() 내부의 make_env_fn()에 의해 개별적으로 래핑됩니다.
        # 따라서 action은 (2,) 형태의 1D 배열입니다. (steer, speed)
        # 이를 base_env.step()이 요구하는 (1, 2) 형태로 변환합니다.
        steer = np.clip(action[0], -self.max_steer, self.max_steer)
        speed = np.clip(action[1], self.min_speed, self.max_speed)
        wrapped_action = np.array([[steer, speed]], dtype=np.float32)

        obs, _, terminate, truncated, info = self.base_env.step(wrapped_action)
        _, _, scan = obs

        """
        TODO: Freely define reward
        [구현 완료] 
        프로젝트 목표(빠른 완주, 충돌 없음)에 맞춘 보상 함수 설계
        """
        
        # 1. 기본 보상: 시간 경과에 따른 페널티 (빨리 완주하도록 유도)
        reward = -0.1 # 'Living penalty'
        
        # 2. 속도 보상: '빠르게' 완주하도록 속도에 비례하는 보상
        # 속도(speed)가 높을수록 보상이 커집니다.
        reward += speed * 0.1

        # 3. 진행 보상: 웨이포인트를 통과할 때마다 큰 보상
        next_waypoint = int(info.get("waypoint", 0))
        if next_waypoint > self.current_waypoint:
            # 새로 통과한 웨이포인트 수만큼 큰 보상을 줍니다.
            reward += 10.0 * (next_waypoint - self.current_waypoint)
            self.current_waypoint = next_waypoint
            
        # 4. 터미널(종료) 보상/페널티
        if terminate:
            # 4a. 성공 (완주): 맵을 완주(웨이포인트 20개 통과)하면 매우 큰 보상
            if info.get("waypoint", 0) == 20:
                reward += 200.0
            # 4b. 실패 (충돌): 완주하지 못하고 종료되면(충돌) 매우 큰 페널티
            else:
                reward -= 50.0
        
        # (참고: truncated는 보통 시간 초과로 인한 종료를 의미하며, 이 경우엔 별도 페널티를 주지 않습니다.)

        self.elapsed_steps += 1

        return scan, reward, terminate, truncated, info