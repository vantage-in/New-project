import os
import time
from typing import Any, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from gymnasium.vector import VectorEnv

from tqdm import tqdm

from .network import ActorCritic


"""
TODO: This skeleton code serves as a guideline;
feel free to modify or replace any part of it.
"""


class PPO:
    def __init__(
        self,
        vec_env: VectorEnv,
        ent_coef: float,
        learning_rate: float,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        n_steps: int = 2048,
        batch_size: int = 128,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        vf_coef: float = 1.0,
        max_grad_norm: float = 0.5,
        checkpoint_freq: int = int(1e6),
        checkpoint_prefix: Optional[str] = None,
        writer=None,
        use_wandb=False,
        # TODO: Freely change parameters
        obs_dim: int = 720,
        act_dim: int = 2,
        action_high: float = None,
        action_low: float = None,
    ):
        self.env = vec_env
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ent_coef = ent_coef
        self.lr = learning_rate
        self.seed = seed
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_prefix = checkpoint_prefix
        self.writer = writer
        self.use_wandb = use_wandb

        """
        TODO: Freely define attributes, methods, etc.
        [구현 완료] 롤아웃 버퍼 초기화
        """

        if self.use_wandb:
            import wandb

            self.wandb = wandb

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_envs = 1 if self.env is None else self.env.num_envs

        self.policy = ActorCritic(self.obs_dim, self.act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.total_steps = 0
        
        # [구현 완료] 롤아웃 데이터를 저장할 버퍼를 초기화합니다.
        # [cite: 103, 111]
        self.obs_buffer = torch.zeros((self.n_steps, self.num_envs, self.obs_dim)).to(self.device)
        self.actions_buffer = torch.zeros((self.n_steps, self.num_envs, self.act_dim)).to(self.device)
        self.logprobs_buffer = torch.zeros((self.n_steps, self.num_envs)).to(self.device)
        self.rewards_buffer = torch.zeros((self.n_steps, self.num_envs)).to(self.device)
        self.dones_buffer = torch.zeros((self.n_steps, self.num_envs)).to(self.device)
        self.values_buffer = torch.zeros((self.n_steps, self.num_envs)).to(self.device)
        
        # GAE 계산을 위한 버퍼
        self.advantages_buffer = torch.zeros((self.n_steps, self.num_envs)).to(self.device)
        self.returns_buffer = torch.zeros((self.n_steps, self.num_envs)).to(self.device)
        

        self.kwargs = {
            "ent_coef": ent_coef,
            "learning_rate": learning_rate,
            "seed": seed,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "checkpoint_freq": checkpoint_freq,
            "checkpoint_prefix": checkpoint_prefix,
            "writer": writer,
            "use_wandb": use_wandb,
            # TODO: Freely change parameters
            "action_high": action_high,
            "action_low": action_low,
        }

    def _select_action(self, obs_tensor: torch.Tensor):
        # [제공된 스켈레톤 코드 사용]
        # [cite: 112]
        with torch.no_grad():
            mu, value, logstd = self.policy(obs_tensor)

            std = logstd.exp()
            dist = Normal(mu, std)

            action = dist.sample()
            logprob = dist.log_prob(action)
            logprob = logprob.sum(-1) # 다변수 정규분포의 logprob 계산

        return action, logprob, value

    def _evaluate_actions(self, obs_tensor: torch.Tensor, actions: torch.Tensor):
        # [제공된 스켈레톤 코드 사용]
        # [cite: 128]
        mu, value, logstd = self.policy(obs_tensor)

        std = logstd.exp()
        dist = Normal(mu, std)
        logprob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)

        return logprob, entropy, value

    def learn(self, total_timesteps: int):
        obs, _ = self.env.reset()
        obs = np.array(obs)
        if obs.ndim == 1:
            obs = np.expand_dims(obs, 0)

        """
        TODO: train the agent using PPO algorithm
        (e.g.: collect rollouts into a buffer, compute GAE and losses, 
        perform minibatch updates, optimize the networks... etc.)
        [구현 완료]
        """

        start_time = time.time()
        with tqdm(total=total_timesteps, ncols=100) as pbar:
            while self.total_steps < total_timesteps:
                old_total_steps = self.total_steps

                num_steps = self.n_steps
                
                # for logging
                ep_ret_list, ep_len_list = [], []

                # =================================================================
                # Phase 1: 데이터 수집 (Rollout) [cite: 109-112]
                # =================================================================
                for step in range(num_steps):
                    obs_tensor = self._process_obs(obs)
                    action, logprob, value = self._select_action(obs_tensor)
                    
                    action_np = self._process_act(action)
                    next_obs, reward, done, truncated, infos = self.env.step(action_np)
                    
                    # 버퍼에 데이터 저장
                    self.obs_buffer[step] = obs_tensor
                    self.actions_buffer[step] = action
                    self.logprobs_buffer[step] = logprob
                    self.values_buffer[step] = value.squeeze()
                    self.rewards_buffer[step] = torch.tensor(reward).to(self.device)
                    self.dones_buffer[step] = torch.tensor(done, dtype=torch.float32).to(self.device)

                    # for logging
                    if "episode" in infos:
                        env_dones = infos.get("_episode", [])
                        for i, d in enumerate(env_dones):
                            if d:
                                ep_ret_list.append(infos["episode"]["r"][i])
                                ep_len_list.append(infos["episode"]["l"][i])

                    obs = np.array(next_obs)
                    self.total_steps += self.num_envs
                    if self.total_steps >= total_timesteps:
                        break
                
                # for logging
                if len(ep_ret_list) > 0:
                    self._log("charts/ep_ret_mean", np.mean(ep_ret_list), self.total_steps)
                    self._log("charts/ep_len_mean", np.mean(ep_len_list), self.total_steps)
                
                # =================================================================
                # Phase 2: Advantage 및 Returns 계산 (GAE) [cite: 113-117, 25-28]
                # =================================================================
                with torch.no_grad():
                    # 마지막 스텝의 value를 부트스트래핑을 위해 계산
                    next_obs_tensor = self._process_obs(obs)
                    _, next_value, _ = self.policy(next_obs_tensor)
                    next_value = next_value.squeeze()
                    next_done = torch.tensor(done, dtype=torch.float32).to(self.device)
                    
                    gae = 0
                    # 끝에서부터 역순으로 GAE 계산
                    for t in reversed(range(self.n_steps)):
                        # TD error (delta) [cite: 25]
                        delta = self.rewards_buffer[t] + self.gamma * next_value * (1.0 - next_done) - self.values_buffer[t]
                        # GAE [cite: 26]
                        gae = delta + self.gamma * self.gae_lambda * (1.0 - next_done) * gae
                        
                        self.advantages_buffer[t] = gae
                        
                        next_value = self.values_buffer[t]
                        next_done = self.dones_buffer[t]
                        
                    # Returns 계산 [cite: 28]
                    self.returns_buffer = self.advantages_buffer + self.values_buffer

                # =================================================================
                # Phase 3: 모델 업데이트 (Optimization) [cite: 118-133]
                # =================================================================
                
                # 롤아웃 데이터를 (n_steps * num_envs) 크기로 펼칩니다.
                b_obs = self.obs_buffer.reshape((-1, self.obs_dim))
                b_logprobs = self.logprobs_buffer.reshape(-1)
                b_actions = self.actions_buffer.reshape((-1, self.act_dim))
                b_advantages = self.advantages_buffer.reshape(-1)
                b_returns = self.returns_buffer.reshape(-1)

                # --- [추가 코드 시작] ---
                # Critic 학습 안정화를 위해 Returns(타겟)를 정규화합니다.
                b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-8)
                # --- [추가 코드 끝] ---
                
                total_batch_size = self.n_steps * self.num_envs
                
                b_v_loss, b_p_loss, b_entropy = [], [], []  # for logging

                for epoch in range(self.epochs):
                    # 매 에포크마다 데이터를 셔플합니다. [cite: 119]
                    indices = np.random.permutation(total_batch_size)
                    
                    # 미니배치 단위로 업데이트 [cite: 119]
                    for start in range(0, total_batch_size, self.batch_size):
                        end = start + self.batch_size
                        mb_indices = indices[start:end]
                        
                        # 미니배치 데이터
                        mb_obs = b_obs[mb_indices]
                        mb_logprobs = b_logprobs[mb_indices]
                        mb_actions = b_actions[mb_indices]
                        mb_advantages = b_advantages[mb_indices]
                        mb_returns = b_returns[mb_indices]
                        
                        # 현재 정책으로 미니배치 재평가 [cite: 120, 128]
                        new_logprob, entropy, new_value = self._evaluate_actions(mb_obs, mb_actions)
                        new_value = new_value.squeeze()

                        # 확률 비율 (Ratio) 계산 [cite: 31]
                        logratio = new_logprob - mb_logprobs
                        ratio = logratio.exp()
                        
                        # Advantage 정규화 (학습 안정성을 위해)
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        # Policy Loss (Clipped Surrogate Objective) [cite: 33, 129]
                        pg_loss1 = mb_advantages * ratio
                        pg_loss2 = mb_advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                        pg_loss = -torch.min(pg_loss1, pg_loss2).mean() # [cite: 39] (L_CLIP)

                        # Value Loss [cite: 38, 130]
                        v_loss = 0.5 * ((new_value - mb_returns) ** 2).mean() # (L_VF)

                        # Entropy Loss [cite: 38, 130]
                        entropy_loss = -entropy.mean() # (S)
                        
                        # 최종 손실 함수 [cite: 39]
                        # L = -L_CLIP + c1 * L_VF - c2 * S
                        loss = pg_loss + self.vf_coef * v_loss + self.ent_coef * entropy_loss
                        
                        # 옵티마이저 스텝 [cite: 131, 133]
                        self.optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        
                        # 로깅
                        b_v_loss.append(v_loss.item())
                        b_p_loss.append(pg_loss.item())
                        b_entropy.append(-entropy_loss.item()) # entropy는 양수로 로깅

                self._log("losses/value_loss", np.mean(b_v_loss), self.total_steps)
                self._log("losses/policy_loss", np.mean(b_p_loss), self.total_steps)
                self._log("losses/entropy", np.mean(b_entropy), self.total_steps)
                self._log("charts/SPS", int((self.total_steps - old_total_steps) / (time.time() - start_time)), self.total_steps)
                start_time = time.time() # SPS 계산을 위해 시간 초기화

                if (old_total_steps // self.checkpoint_freq) < (self.total_steps // self.checkpoint_freq):
                    save_step = (self.total_steps // self.checkpoint_freq) * self.checkpoint_freq
                    self.save(f"{self.checkpoint_prefix}_{save_step}.pt")

                pbar.update(self.total_steps - old_total_steps)

        print("Learning finished")

    def save(self, model_path: str):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "kwargs": self.kwargs,
            },
            model_path,
        )
        print(f"Saved model to {model_path}")

    @classmethod
    def load(cls, model_path: str, vec_env: VectorEnv = None, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(model_path, map_location=device)

        kwargs = checkpoint.get("kwargs", {})

        model = cls(vec_env=vec_env, device=device, **kwargs)

        model.policy.load_state_dict(checkpoint["policy_state_dict"])
        model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.total_steps = checkpoint.get("total_steps", 0)

        print(f"Loaded model from {model_path}")
        return model

    def predict(self, raw_obs: Any, deterministic: bool = False) -> np.ndarray:
        obs_arr = np.asarray(raw_obs)
        if obs_arr.ndim == 1:
            obs_arr = np.expand_dims(obs_arr, 0)

        obs_tensor = self._process_obs(obs_arr)
        with torch.no_grad():
            mu, _, logstd = self.policy(obs_tensor)

            std = logstd.exp()
            if deterministic:
                actions = mu
            else:
                dist = Normal(mu, std)
                actions = dist.sample()

            return self._process_act(actions)

    def _process_obs(self, scan_array: np.ndarray) -> torch.Tensor:
        """
        TODO: preprocess observations for model input
        [구현 완료]
        """
        # numpy 배열을 torch 텐서로 변환하고 디바이스로 보냅니다.
        obs_tensor = torch.tensor(scan_array, dtype=torch.float32).to(self.device)
        return obs_tensor

    def _process_act(self, action_tensor: torch.Tensor) -> np.ndarray:
        """
        TODO: postprocess actions for model output
        [구현 완료]
        """
        # torch 텐서를 numpy 배열로 변환합니다.
        action_array = action_tensor.detach().cpu().numpy()
        return action_array

    def _log(self, tag: str, value: float, step: int):
        if self.use_wandb:
            self.wandb.log({tag: value, "global_step": step})
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)