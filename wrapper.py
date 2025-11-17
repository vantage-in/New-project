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

        # Lidar 스캔 배열은 720개입니다 (0~719)
        # 따라서 정면 인덱스는 (720 / 2) - 1 = 359 입니다.
        self.center_lidar_idx = 359
        
        # --- [추가] 헤딩 리워드 계수 ---
        self.straight_reward_coeff = 0.5  # 직진 코스 보상 가중치
        self.straight_penalty_coeff = 1.0 # 직진 코스 페널티 가중치
        self.corner_reward_coeff = 0.5    # 코너 코스 보상 가중치
        self.corner_penalty_coeff = 1.0   # 코너 코스 페널티 가중치
        

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

    def _calculate_heading_reward(self, scan):
        """
        [수정된 함수] 디버깅을 위해 (heading_reward, debug_info)를 반환
        """
        
        max_dist = np.max(scan)
        
        # Lidar 최대값과 동일한 모든 인덱스를 찾습니다.
        max_indices = np.where(scan == max_dist)[0]
        
        if len(max_indices) == 0: 
            max_idx = self.center_lidar_idx
        else:
            # 최대값 인덱스들의 중앙값(median)을 계산합니다.
            max_idx = int(np.median(max_indices))
            
        heading_reward = 0.0
        course_type = "Unknown"
        cause_string = "None" # 디버깅을 위한 리워드 원인

        if max_dist >= 10.0:
            # === 1. 직진 코스 로직 ===
            course_type = "Straight"
            
            # 1a. 정면(359)이 5m 미만이면 페널티
            front_dist = scan[self.center_lidar_idx]
            if front_dist < 5.0:
                penalty = self.straight_penalty_coeff * (5.0 - front_dist)
                heading_reward -= penalty
                cause_string = f"Front Penalty (-{penalty:.2f})"
                
            # 1b. 최대 거리의 중앙(max_idx)이 정면(359)에 가까울수록 보상
            reward_range_half = 30 # (359 +/- 30) -> 329 ~ 389
            idx_diff = abs(max_idx - self.center_lidar_idx)
            
            if idx_diff < reward_range_half:
                bonus = self.straight_reward_coeff * ((reward_range_half - idx_diff) / reward_range_half)
                heading_reward += bonus
                cause_string = f"Center Bonus (+{bonus:.2f})"
            else:
                cause_string = "Straight (Neutral)" # 보상/페널티 범위 밖
                
        else:
            # === 2. 코너 코스 로직 ===
            
            diffs = np.abs(np.diff(scan))
            frontier_indices = np.where(diffs >= 2.0)[0]

            if len(frontier_indices) > 0:
                frontiers = []
                for i in frontier_indices:
                    dist_val = max(scan[i], scan[i+1])
                    frontiers.append((i, dist_val))
                
                best_frontier_idx = max(frontiers, key=lambda f: f[1])[0]
                
                idx_a = best_frontier_idx
                idx_b = best_frontier_idx + 1
                
                if scan[idx_a] < scan[idx_b]:
                    idx_inner = idx_a # 더 가까운 포인트 (인코스)
                    idx_outer = idx_b
                else:
                    idx_inner = idx_b
                    idx_outer = idx_a
                    
                if idx_inner < self.center_lidar_idx:
                    # === 2A. 좌회전 코스 ===
                    course_type = "Left Corner"
                    if max_idx <= idx_inner:
                        heading_reward -= self.corner_penalty_coeff * 1.0
                        cause_string = f"Corner Inner Penalty (-{self.corner_penalty_coeff:.2f})"
                    else:
                        reward_range_corner = 40 
                        idx_diff_corner = max_idx - idx_outer
                        if 0 <= idx_diff_corner < reward_range_corner:
                            bonus = self.corner_reward_coeff * ((reward_range_corner - idx_diff_corner) / reward_range_corner)
                            heading_reward += bonus
                            cause_string = f"Corner Bonus (+{bonus:.2f})"
                        elif idx_diff_corner >= reward_range_corner:
                            heading_reward -= self.corner_penalty_coeff * 0.5 
                            cause_string = f"Corner Outer Penalty (-{self.corner_penalty_coeff * 0.5:.2f})"
                        else:
                            cause_string = "Corner (Neutral)"
                else:
                    # === 2B. 우회전 코스 ===
                    course_type = "Right Corner"
                    if max_idx >= idx_inner:
                        heading_reward -= self.corner_penalty_coeff * 1.0
                        cause_string = f"Corner Inner Penalty (-{self.corner_penalty_coeff:.2f})"
                    else:
                        reward_range_corner = 40 
                        idx_diff_corner = idx_outer - max_idx 
                        if 0 <= idx_diff_corner < reward_range_corner:
                            bonus = self.corner_reward_coeff * ((reward_range_corner - idx_diff_corner) / reward_range_corner)
                            heading_reward += bonus
                            cause_string = f"Corner Bonus (+{bonus:.2f})"
                        elif idx_diff_corner >= reward_range_corner:
                            heading_reward -= self.corner_penalty_coeff * 0.5
                            cause_string = f"Corner Outer Penalty (-{self.corner_penalty_coeff * 0.5:.2f})"
                        else:
                            cause_string = "Corner (Neutral)"
            else:
                course_type = "Corner (No Frontier)"
                cause_string = "No Frontier Found"

        # 디버깅 정보를 딕셔너리로 반환
        debug_info = {
            'course': course_type,
            'max_idx': max_idx,
            'max_dist': max_dist,
            'cause': cause_string
        }
        return heading_reward, debug_info

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

        # --- [디버깅] 리워드 원인을 저장할 리스트 ---
        reward_causes = []
        
        # 1. 기본 보상: 시간 경과에 따른 페널티 (빨리 완주하도록 유도)
        reward = -0.1 # 'Living penalty'
        reward_causes.append(f"Base: {reward:.2f}")
        
        # 2. 속도 보상: '빠르게' 완주하도록 속도에 비례하는 보상
        # 속도(speed)가 높을수록 보상이 커집니다.
        reward += speed * 0.1

        # 3. 진행 보상: 웨이포인트를 통과할 때마다 큰 보상
        next_waypoint = int(info.get("waypoint", 0))
        if next_waypoint > self.current_waypoint:
            # 새로 통과한 웨이포인트 수만큼 큰 보상을 줍니다.
            wp_reward = 10.0 * (next_waypoint - self.current_waypoint)
            reward += wp_reward
            reward_causes.append(f"Waypoint: +{wp_reward:.2f}")
            self.current_waypoint = next_waypoint

        # === 2. 헤딩 보상 (새 로직) ===
        heading_reward, debug_info = self._calculate_heading_reward(scan)
        reward += heading_reward
        # 헤딩 보상이 0이 아닐 때만 원인 추가 (로그 깔끔하게)
        if heading_reward != 0.0:
            reward_causes.append(f"Heading: {heading_reward:.2f} ({debug_info['cause']})")
            
        # 4. 터미널(종료) 보상/페널티
        if terminate:
            # 4a. 성공 (완주): 맵을 완주(웨이포인트 20개 통과)하면 매우 큰 보상
            if info.get("waypoint", 0) == 20:
                term_reward = 50.0
                reward += term_reward
                reward_causes.append(f"Terminal: +{term_reward:.2f} (Success)")
            # 4b. 실패 (충돌): 완주하지 못하고 종료되면(충돌) 매우 큰 페널티
            else:
                term_penalty = -50.0
                reward += term_penalty
                reward_causes.append(f"Terminal: {term_penalty:.2f} (Collision)")

        # --- [추가] 디버그 최종 출력 ---
        debug_str = (
            f"[Debug] Step: {self.elapsed_steps} | "
            f"Course: {debug_info['course']} (Idx: {debug_info['max_idx']}, Dist: {debug_info['max_dist']:.2f}) | "
            f"Final Reward: {reward:.2f} | "
            f"Causes: {', '.join(reward_causes)}"
        )
        print(debug_str)
        # --- [디버그 끝] ---
        
        # (참고: truncated는 보통 시간 초과로 인한 종료를 의미하며, 이 경우엔 별도 페널티를 주지 않습니다.)

        self.elapsed_steps += 1

        return scan, reward, terminate, truncated, info