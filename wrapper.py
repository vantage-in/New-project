import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box


"""
TODO: This skeleton code serves as a guideline;
feel free to modify or replace any part of it.
"""
from collections import deque

class RCCarEnvTrainWrapper(gym.Wrapper):
    def __init__(self, base_env, max_steer, min_speed, max_speed, time_limit, mode):
        super().__init__(base_env)
        self.base_env = base_env
        self.max_steer = max_steer
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.time_limit = time_limit
        self.mode = mode

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
        self.MAX_LIDAR_DIST = 30.0

        # [설정 2] Frame Stacking & Stride 설정
        self.stack_num = 4          # 신경망에 들어갈 프레임 수
        self.stack_interval = 6     # 프레임 간격 (Stride) -> 6이면 0.15초 간격
        self.downsample_interval = 4 # 720 -> 180 다운샘플링

        # 버퍼 크기 계산: (4개 * 3간격) - 2(인덱스 보정) = 10개 이상의 과거가 필요
        # 넉넉하게 maxlen 설정
        self.buffer_maxlen = (self.stack_num * self.stack_interval)
        self.frames = deque(maxlen=self.buffer_maxlen)

        # 관측 공간 정의 (180 * 4 = 720 dim)
        self.raw_scan_len = 720
        self.single_frame_dim = self.raw_scan_len  // self.downsample_interval
        self.obs_dim = self.single_frame_dim * self.stack_num
        
        self.observation_space = Box(
            low=0.0, high=1.0, # 정규화했으므로 0~1
            shape=(self.obs_dim,), 
            dtype=np.float32
        )


        self.current_waypoint = 0

        # Lidar 스캔 배열은 720개입니다 (0~719)
        # 따라서 정면 인덱스는 (720 / 2) - 1 = 359 입니다.
        self.center_lidar_idx = 359
        
        # --- [추가] 헤딩 리워드 계수 ---
        self.straight_reward_coeff = 1.0  # 직진 코스 보상 가중치
        self.straight_penalty_coeff = 1.0 # 직진 코스 페널티 가중치
        self.corner_reward_coeff = 1.0    # 코너 코스 보상 가중치
        self.corner_penalty_coeff = 1.0   # 코너 코스 페널티 가중치
        
    def _process_scan(self, scan):
        """Lidar 1개 프레임 처리: 다운샘플링 -> 정규화"""
        # 1. 다운샘플링 (720 -> 180)
        processed = scan[::self.downsample_interval]
        
        # 2. 정규화 (0.0 ~ 1.0) [Normalize FIRST]
        processed = processed / self.MAX_LIDAR_DIST
        return np.clip(processed, 0.0, 1.0)

    def _get_stacked_obs(self):
        """버퍼에서 Stride 간격으로 꺼내서 합치기"""
        # 현재 버퍼에 있는 프레임들을 리스트로 변환 (오래된 것 -> 최신 순)
        buffer_list = list(self.frames)
        
        # 뒤에서부터 interval 간격으로 4개 추출
        # 예: [-1(현재), -4, -7, -10]
        selected_frames = buffer_list[::-1][::self.stack_interval][:self.stack_num]
        
        # 시간 순서를 맞추기 위해 다시 뒤집음 (과거 -> 현재)
        # (MLP라 순서가 크게 상관없지만, 일관성을 위해)
        selected_frames = selected_frames[::-1]
        
        # 만약 버퍼가 아직 꽉 차지 않았다면(초기화 직후), 제일 첫 프레임으로 채움
        while len(selected_frames) < self.stack_num:
            selected_frames.insert(0, selected_frames[0])
            
        return np.concatenate(selected_frames, axis=0)

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

        # 첫 프레임 처리 및 버퍼 채우기
        processed = self._process_scan(scan)
        
        # 초기 상태에서는 같은 프레임으로 버퍼를 가득 채움 (Warm-up)
        for _ in range(self.buffer_maxlen):
            self.frames.append(processed)

        return self._get_stacked_obs(), info

    def _calculate_heading_reward(self, scan):
        """
        [추가] wrapper.py에서 가져온 헤딩 보상 계산 로직 (검증용)
         [수정] 2025-11-18 요청사항 반영
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

        if max_dist >= 9.0:
            # === 1. 직진 코스 로직 ===
            course_type = "Straight"
            
            # 1a. 정면(359)이 5m 미만이면 페널티
            front_dist = scan[self.center_lidar_idx]
            if front_dist < 5.0:
                penalty = self.straight_penalty_coeff * (5.0 - front_dist)
                heading_reward -= penalty
                cause_string = f"Front Penalty (-{penalty:.2f})"
            else:
                diffs_straight = np.abs(np.diff(scan))
                frontier_indices_straight = np.where(diffs_straight >= 2.5)[0]

                if max_dist < 20.0 and len(frontier_indices_straight) > 0:
                # --- [A] 기울어진 직진 (Leaning Straight) 로직 ---

                    frontiers = []
                    for i in frontier_indices_straight:
                        dist_val = max(scan[i], scan[i+1])
                        frontiers.append((i, dist_val))
                    best_frontier_idx = max(frontiers, key=lambda f: f[1])[0]
                
                    idx_a = best_frontier_idx
                    idx_b = best_frontier_idx + 1

                    if scan[idx_a] < scan[idx_b]:
                        # (A-1. 좌-기울)
                        course_type = "Straight-Left"
                        idx_inner = min(idx_a + 10, 718)
                        idx_outer = min(idx_b + 10, 719)
                        
                        if self.center_lidar_idx <= idx_inner:
                            # [수정] 약한 inner 페널티 (코너의 50%)
                            penalty = self.corner_penalty_coeff * 3.0
                            heading_reward -= penalty
                            cause_string = f"Straight-Inner Penalty (-{penalty:.2f})"
                        else:
                            # [새 로직] 40 Flat + 40 Linear
                            flat_range = 40       # (0-39)
                            linear_range_end = 120 # (40-79)
                            idx_diff_corner = self.center_lidar_idx - idx_outer

                            if 0 <= idx_diff_corner < flat_range:
                                bonus = self.corner_reward_coeff * 1.0
                                heading_reward += bonus
                                cause_string = f"Straight Bonus (Flat) (+{bonus:.2f})"
                            elif flat_range <= idx_diff_corner < linear_range_end:
                                linear_range_len = linear_range_end - flat_range # 40
                                bonus = self.corner_reward_coeff * ((linear_range_end - idx_diff_corner) / linear_range_len)
                                heading_reward += bonus
                                cause_string = f"Straight Bonus (Linear) (+{bonus:.2f})"
                            elif cause_string == "None": # 5m 페널티도 안 받은 경우
                                cause_string = "Straight-Left (Neutral)"
                    
                    else:
                        # (A-2. 우-기울)
                        course_type = "Straight-Right"
                        idx_inner = max(idx_b - 10, 1)
                        idx_outer = max(idx_a - 10, 0)

                        if self.center_lidar_idx >= idx_inner:
                            # [수정] 약한 inner 페널티 (코너의 50%)
                            penalty = self.corner_penalty_coeff * 3.0
                            heading_reward -= penalty
                            cause_string = f"Straight-Inner Penalty (-{penalty:.2f})"
                        else:
                            # [새 로직] 40 Flat + 40 Linear
                            flat_range = 40
                            linear_range_end = 120
                            idx_diff_corner = idx_outer - self.center_lidar_idx

                            if 0 <= idx_diff_corner < flat_range:
                                bonus = self.corner_reward_coeff * 1.0
                                heading_reward += bonus
                                cause_string = f"Straight Bonus (Flat) (+{bonus:.2f})"
                            elif flat_range <= idx_diff_corner < linear_range_end:
                                linear_range_len = linear_range_end - flat_range # 40
                                bonus = self.corner_reward_coeff * ((linear_range_end - idx_diff_corner) / linear_range_len)
                                heading_reward += bonus
                                cause_string = f"Straight Bonus (Linear) (+{bonus:.2f})"
                            elif cause_string == "None":
                                cause_string = "Straight-Right (Neutral)"
                else:
                    # 1b. 최대 거리의 중앙(max_idx)이 정면(359)에 가까울수록 보상
                    course_type = "Straight-Center"
                    reward_range_half = 40 # [수정] 10 -> 20 (359 +/- 20) -> 339 ~ 379
                    idx_diff = abs(max_idx - self.center_lidar_idx)
                    
                    if idx_diff < reward_range_half:
                        # [수정] 범위가 넓어진 만큼 외각에서는 약한 보상을 받게 됨
                        bonus = 3.0 * self.straight_reward_coeff * ((reward_range_half - idx_diff) / reward_range_half)
                        heading_reward += bonus
                        cause_string = f"Center Bonus (+{bonus:.2f})"
                    elif cause_string == "None":
                        cause_string = "Straight (Neutral)" # 보상/페널티 범위 밖
                
        else:
            # === 2. 코너 코스 로직 ===
            
            diffs = np.abs(np.diff(scan))
            frontier_indices = np.where(diffs >= 1.0)[0]

            if len(frontier_indices) > 0:
                frontiers = []
                for i in frontier_indices:
                    dist_val = max(scan[i], scan[i+1])
                    frontiers.append((i, dist_val))
                
                best_frontier_idx = max(frontiers, key=lambda f: f[1])[0]
                
                idx_a = best_frontier_idx
                idx_b = best_frontier_idx + 1
                
                if scan[idx_a] < scan[idx_b]:
                    idx_inner = min(idx_a + 10, 718)# 더 가까운 포인트 (인코스)
                    idx_outer = min(idx_b + 10, 719)

                    # === 2A. 좌회전 코스 ===
                    course_type = "Left Corner"
                    if self.center_lidar_idx <= idx_inner:
                        heading_reward -= self.corner_penalty_coeff * 3.0
                        cause_string = f"Corner Inner Penalty (-{self.corner_penalty_coeff:.2f})"
                    else:
                        reward_range_corner = 80 # [수정] 20 -> 40 (보상 범위 2배)
                        flat_reward_range = 40
                        linear_reward_end = reward_range_corner
                        neutral_range_corner = 80 # [추가] 페널티 전 중립 구간 (20)
                        
                        idx_diff_corner = self.center_lidar_idx - idx_outer
                        
                        if 0 <= idx_diff_corner < flat_reward_range:
                            # 1. Flat 구간 (0 ~ 19): 최대 보상
                            bonus = self.corner_reward_coeff * 1.0
                            heading_reward += bonus
                            cause_string = f"Corner Bonus (Flat) (+{bonus:.2f})"
                        elif flat_reward_range <= idx_diff_corner < linear_reward_end:
                            # 2. Linear 구간 (20 ~ 39): 1.0 -> 0.0 으로 선형 감소
                            linear_range_len = linear_reward_end - flat_reward_range # 20
                            bonus = self.corner_reward_coeff * ((linear_reward_end - idx_diff_corner) / linear_range_len)
                            heading_reward += bonus
                            cause_string = f"Corner Bonus (Linear) (+{bonus:.2f})"
                        elif linear_reward_end <= idx_diff_corner < (linear_reward_end + neutral_range_corner): # [추가] (40 ~ 59)
                            cause_string = "Corner (Neutral)" # 중립 구간
                        elif idx_diff_corner >= (linear_reward_end + neutral_range_corner): # [수정] (>= 60)
                            heading_reward -= self.corner_penalty_coeff * 0.5 
                            cause_string = f"Corner Outer Penalty (-{self.corner_penalty_coeff * 0.5:.2f})"
                        else:
                            cause_string = "Corner (Neutral)"
                else:
                    idx_inner = max(idx_b - 10, 1)
                    idx_outer = max(idx_a - 10, 0)

                    # === 2B. 우회전 코스 ===
                    course_type = "Right Corner"
                    if self.center_lidar_idx >= idx_inner:
                        heading_reward -= self.corner_penalty_coeff * 3.0
                        cause_string = f"Corner Inner Penalty (-{self.corner_penalty_coeff:.2f})"
                    else:
                        # [수정] 좌회전과 동일한 구조로 변경
                        reward_range_corner = 80 # [수정] 80 -> 100
                        flat_reward_range = 40    # [추가]
                        linear_reward_end = reward_range_corner
                        neutral_range_corner = 80 # [수정] 80 -> 60
                        
                        idx_diff_corner = idx_outer - self.center_lidar_idx
                        
                        if 0 <= idx_diff_corner < flat_reward_range:
                            # 1. Flat 구간 (0 ~ 39): 최대 보상
                            bonus = self.corner_reward_coeff * 1.0
                            heading_reward += bonus
                            cause_string = f"Corner Bonus (Flat) (+{bonus:.2f})"
                        elif flat_reward_range <= idx_diff_corner < linear_reward_end:
                            # 2. Linear 구간 (40 ~ 99): 1.0 -> 0.0 으로 선형 감소
                            linear_range_len = linear_reward_end - flat_reward_range # 60
                            bonus = self.corner_reward_coeff * ((linear_reward_end - idx_diff_corner) / linear_range_len)
                            heading_reward += bonus
                            cause_string = f"Corner Bonus (Linear) (+{bonus:.2f})"
                        elif linear_reward_end <= idx_diff_corner < (linear_reward_end + neutral_range_corner): # (100 ~ 159)
                            cause_string = "Corner (Neutral)" # 중립 구간
                        elif idx_diff_corner >= (linear_reward_end + neutral_range_corner): # (>= 160)
                            heading_reward -= self.corner_penalty_coeff * 0.5
                            cause_string = f"Corner Outer Penalty (-{self.corner_penalty_coeff * 0.5:.2f})"
                        else:
                            cause_string = "Corner (Neutral)"
                    
            else:
                course_type = "Corner (No Frontier)"
                cause_string = "No Frontier Found"
                heading_reward -= 5.0

        # 디버깅 정보를 딕셔너리로 반환
        debug_info = {
            'course': course_type,
            'max_idx': max_idx,
            'max_dist': max_dist,
            'cause': cause_string
        }
        return heading_reward, debug_info
    
    def _generate_debug_log(self, reward_components, debug_info):
        """
        [새 함수] 
        val 모드에서만 호출되는 디버그 로그 생성 및 출력 함수
        """
        
        # 보상 원인 문자열 리스트 생성
        reward_causes = []
        reward_causes.append(f"Base: {reward_components['base']:.2f}")
        
        if reward_components['wp'] != 0.0:
            reward_causes.append(f"Waypoint: +{reward_components['wp']:.2f}")
            
        if reward_components['heading'] != 0.0:
            reward_causes.append(f"Heading: {reward_components['heading']:.2f} ({debug_info['cause']})")
            
        if reward_components['terminal'] != 0.0:
            if reward_components['terminal'] > 0:
                reward_causes.append(f"Terminal: +{reward_components['terminal']:.2f} (Success)")
            else:
                reward_causes.append(f"Terminal: {reward_components['terminal']:.2f} (Collision)")

        # 최종 디버그 문자열 출력
        debug_str = (
            f"[Debug] Step: {self.elapsed_steps} | "
            f"Course: {debug_info['course']} (Classify Idx: {debug_info['max_idx_FOR_CLASSIFICATION']}, Dist: {debug_info['max_dist']:.2f}) | "
            f"Final Reward: {reward_components['total']:.2f} | "
            f"Causes: {', '.join(reward_causes)}"
        )
        print(debug_str)

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
        reward -= abs(steer) * 1.0
        base_reward = reward

        # 3. 진행 보상: 웨이포인트를 통과할 때마다 큰 보상
        next_waypoint = int(info.get("waypoint", 0))
        if next_waypoint > self.current_waypoint:
            # 새로 통과한 웨이포인트 수만큼 큰 보상을 줍니다.
            wp_reward = 30.0 * (next_waypoint - self.current_waypoint)
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
                term_reward = 100.0
                reward += term_reward
                reward_causes.append(f"Terminal: +{term_reward:.2f} (Success)")
            # 4b. 실패 (충돌): 완주하지 못하고 종료되면(충돌) 매우 큰 페널티
            else:
                term_reward = -100.0
                reward += term_reward
                reward_causes.append(f"Terminal: {term_reward:.2f} (Collision)")

        if self.mode == "val":
            # train 모드에서는 이 함수가 호출되지 않아 오버헤드 없음
            reward_components = {
                'base': base_reward,
                'wp': wp_reward,
                'heading': heading_reward,
                'terminal': term_reward,
                'total': reward
            }
            self._generate_debug_log(reward_components, debug_info)
        
        # (참고: truncated는 보통 시간 초과로 인한 종료를 의미하며, 이 경우엔 별도 페널티를 주지 않습니다.)

        self.elapsed_steps += 1

        processed = self._process_scan(scan)
        self.frames.append(processed)

        return self._get_stacked_obs(), reward, terminate, truncated, info