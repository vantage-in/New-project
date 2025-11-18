import os
import time
from datetime import datetime
import argparse
import traceback

import numpy as np
from ruamel.yaml import YAML
from easydict import EasyDict

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_prefix

from message.msg import Result, Query
from rccar_gym.env_wrapper import RCCarWrapper
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

###################################################
########## YOU CAN ONLY CHANGE THIS PART  #########

"""
TODO:
Freely import modules, define methods and classes, etc.

You may add extra Python scripts, but remember to push them to GitHub.
Notify TA if any additional modules need to be installed on the evaluation server.
(For deep learning, please use PyTorch.)

However, it is NOT ALLOWED to use pre-built PPO implementations (e.g.: Stable-Baselines3 or RLlib).
You must implement the training process on your own.

[구현 완료]
- PPO 알고리즘, Wrapper, Torch 등 필요한 모듈을 import 합니다.
"""

TEAM_NAME = "ROADRUNNER"  # [수정] 팀 이름 변경

import random

import torch

from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from .wrapper import RCCarEnvTrainWrapper  # [추가] 우리가 구현한 Wrapper
from .algo import PPO  # [추가] 우리가 구현한 PPO 알고리즘


def make_env_fn(index, args, maps, max_steer, min_speed, max_speed, time_limit):
    def _thunk():
        # [수정] RCCarWrapper 래핑을 RCCarEnvTrainWrapper로 변경
        base_env = RCCarWrapper(args=args, maps=maps, render_mode=None, seed=args.seed + index)
        # [수정] 우리가 정의한 보상함수를 사용하는 Wrapper로 감쌉니다.
        env = RCCarEnvTrainWrapper(base_env, max_steer=max_steer, min_speed=min_speed, max_speed=max_speed, time_limit=time_limit, mode=args.mode)
        return RecordEpisodeStatistics(env)

    return _thunk


################### CHANGE END  ###################
###################################################


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="Set seed number.")
    parser.add_argument("--env_config", default="configs/env.yaml", type=str, help="Path to environment config file (.yaml)")  # or ../..
    parser.add_argument("--dynamic_config", default="configs/dynamic.yaml", type=str, help="Path to dynamic config file (.yaml)")  # or ../..
    parser.add_argument("--render", default=True, action="store_true", help="Whether to render or not.")
    parser.add_argument("--no_render", default=False, action="store_true", help="No rendering.")
    parser.add_argument("--mode", default="val", type=str, help="Whether train new model or not")
    parser.add_argument("--traj_dir", default="trajectory", type=str, help="Saved trajectory path relative to 'IS_TEAMNAME/project/'")
    parser.add_argument("--model_dir", default="model", type=str, help="Model path relative to 'IS_TEAMNAME/project/'")

    ###################################################
    ########## YOU CAN ONLY CHANGE THIS PART ##########
    """
    TODO:
    Change the model name as you want.
    Note that this will used for evaluation by the server as well.
    You can add any arguments you want.
    """

    parser.add_argument("--model_name", default="roadrunner_ppo.pt", type=str, help="Model name to save and load") # [수정] 모델 이름 변경
    parser.add_argument("--checkpoint_freq", default=int(5e5), type=int, help="Save every N timesteps during training")
    parser.add_argument("--wandb", default=False, action="store_true", help="(Optional) Enable wandb logging")
    parser.add_argument("--tb", default=False, action="store_true", help="(Optional) Enable tensorboard logging")
    
    # [추가] 학습을 위한 하이퍼파라미터 (PPO.learn()의 기본값을 오버라이드)
    parser.add_argument("--total_timesteps", default=int(1e7), type=int, help="Total timesteps to train")
    parser.add_argument("--num_envs", default=16, type=int, help="Number of parallel environments")
    parser.add_argument("--n_steps", default=2048, type=int, help="Steps to run in each environment per policy update")
    parser.add_argument("--batch_size", default=64, type=int, help="Minibatch size for PPO update")
    parser.add_argument("--n_epochs", default=10, type=int, help="Number of epochs to update policy per rollout")
    parser.add_argument("--lr", default=3e-5, type=float, help="Learning rate")
    parser.add_argument("--ent_coef", default=1e-6, type=float, help="Entropy coefficient")


    ################### CHANGE END  ###################
    ###################################################

    args = parser.parse_args()
    args = EasyDict(vars(args))

    # render
    if args.no_render:
        args.render = False

    ws_path = os.path.join(get_package_prefix("rccar_bringup"), "../..")

    # map files
    args.maps = os.path.join(ws_path, "maps")
    args.maps = [map for map in os.listdir(args.maps) if os.path.isdir(os.path.join(args.maps, map))]

    # configuration files
    args.env_config = os.path.join(ws_path, args.env_config)
    args.dynamic_config = os.path.join(ws_path, args.dynamic_config)

    with open(args.env_config, "r") as f:
        task_args = EasyDict(YAML().load(f))
    with open(args.dynamic_config, "r") as f:
        dynamic_args = EasyDict(YAML().load(f))

    args.update(task_args)
    args.update(dynamic_args)

    # Trajectory & Model Path
    project_path = os.path.join(ws_path, f"src/rccar_bringup/rccar_bringup/project/IS_{TEAM_NAME}/project")
    args.project_path = project_path
    args.traj_dir = os.path.join(project_path, args.traj_dir)
    args.model_dir = os.path.join(project_path, args.model_dir)
    args.model_path = os.path.join(args.model_dir, args.model_name)

    return args


class RCCarPolicy(Node):
    def __init__(self, args):
        super().__init__(f"{TEAM_NAME}_project2")
        self.args = args
        self.mode = args.mode

        self.query_sub = self.create_subscription(Query, "/query", self.query_callback, 10)
        self.result_pub = self.create_publisher(Result, "/result", 10)

        self.dt = args.timestep
        self.max_speed = args.max_speed
        self.min_speed = args.min_speed
        self.max_steer = args.max_steer
        self.maps = args.maps
        self.render = args.render
        self.time_limit = 180.0

        self.project_path = args.project_path
        self.traj_dir = args.traj_dir
        self.model_dir = args.model_dir
        self.model_name = args.model_name
        self.model_path = args.model_path

        ###################################################
        ########## YOU CAN ONLY CHANGE THIS PART ##########
        """
        TODO:
        Freely change the codes (__init__, train, load, get_action) to increase the performance.
        You can also add attributes, methods, etc.
        [구현 완료]
        - 디바이스 설정을 'cuda' 우선으로 변경
        - 시드 설정
        """

        # [수정] GPU 사용이 가능하면 GPU를 사용하도록 변경
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"

        seed = int(self.args.seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)

        self.model = None

        self.load() # [수정 없음] 모드에 따라 train() 또는 load() 호출
        self.get_logger().info(f">>> Running Project 2 for TEAM {TEAM_NAME}")

    def train(self):
        self.get_logger().info(">>> Start model training")
        """
        Train and save your model.
        You can either use this part or explicitly train using other python codes.
        [구현 완료]
        - PPO 모델을 인스턴스화하고 학습을 진행합니다.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        writer = None
        if self.args.wandb:
            import wandb

            wandb.init(
                project=f"IS25_project2_{TEAM_NAME}",
                sync_tensorboard=self.args.tb,
                config=vars(self.args),
                name=timestamp,
                save_code=True,
            )
        elif self.args.tb:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = os.path.join(self.project_path, "runs", timestamp)
            writer = SummaryWriter(log_dir=log_dir)
            writer.add_text(
                "configs",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
            )
        
        # [수정] args에서 num_envs 가져오기
        num_envs = self.args.num_envs 
        env_fns = [make_env_fn(i, self.args, self.maps, self.max_steer, self.min_speed, self.max_speed, self.time_limit) for i in range(num_envs)]
        vec_env = SyncVectorEnv(env_fns)

        total_timesteps = int(self.args.total_timesteps)

        # [수정] PPO 모델 인스턴스화
        if not os.path.exists(self.model_path):
            model = PPO(
                vec_env=vec_env,
                ent_coef=self.args.ent_coef,
                learning_rate=self.args.lr,
                device=self.device,
                seed=int(self.args.seed),
                n_steps=self.args.n_steps,
                batch_size=self.args.batch_size,
                n_epochs=self.args.n_epochs,
                checkpoint_freq=self.args.checkpoint_freq,
                checkpoint_prefix=f"{self.model_dir}/{timestamp}/{os.path.splitext(self.model_name)[0]}",
                writer=writer,
                use_wandb=self.args.wandb,
                # algo.py의 기본값 (obs_dim=720, act_dim=2)을 사용합니다.
            )
        else:
            # [수정] PPO.load 사용하여 모델 로드
            model = PPO.load(self.model_path, vec_env, device=self.device)
            model.writer = writer
            model.use_wandb = self.args.wandb
            self.get_logger().info(f"Continue training for the existing model from {self.model_path}")
            
            # 2. 로드된 모델 객체에 'self.args'의 새 하이퍼파라미터를 강제로 덮어씌웁니다.
            self.get_logger().warn("[OVERWRITE] Forcing new hyperparameters on loaded model...")

            # 덮어쓸 파라미터 딕셔너리 생성
            # (PPO.__init__의 인자 이름과 self.args의 이름을 매핑)
            new_params = {
                "learning_rate": self.args.lr,
                "ent_coef": self.args.ent_coef,
                "n_steps": self.args.n_steps,
                "batch_size": self.args.batch_size,
                "n_epochs": self.args.n_epochs,
                "checkpoint_freq": self.args.checkpoint_freq,
                # --- 필요시 다른 하이퍼파라미터도 여기에 추가 ---
                # "gamma": self.args.gamma,
                # "gae_lambda": self.args.gae_lambda,
                # "clip_range": self.args.clip_range,
                # "vf_coef": self.args.vf_coef,
                # "max_grad_norm": self.args.max_grad_norm,
            }

            # 3. 모델 객체의 속성(model.batch_size)과
            #    다음 저장을 위한 kwargs(model.kwargs['batch_size'])를 모두 덮어씁니다.
            for key, value in new_params.items():
                if hasattr(model, key):
                    setattr(model, key, value) # model.batch_size = 64
                    model.kwargs[key] = value  # model.kwargs['batch_size'] = 64
                    self.get_logger().info(f"  > Overwrote '{key}' to {value}")
                else:
                    self.get_logger().warn(f"  > Cannot find attribute '{key}' to overwrite.")
            
            # 4. 체크포인트 저장 경로도 새 타임스탬프로 덮어씁니다. (이전 수정 사항)
            new_prefix = f"{self.model_dir}/{timestamp}/{os.path.splitext(self.model_name)[0]}"
            model.checkpoint_prefix = new_prefix
            model.kwargs["checkpoint_prefix"] = new_prefix
            self.get_logger().info(f"  > Overwrote 'checkpoint_prefix' to {new_prefix}")

        # [수정] 학습 시작
        model.learn(total_timesteps=total_timesteps)

        # [수정] 학습 완료 후 모델 저장 및 할당
        model.save(self.model_path)
        self.model = model

        self.get_logger().info(f">>> Trained model {self.model_name} is saved")

        if self.args.wandb:
            wandb.finish()
        if self.args.tb:
            writer.close()

    def load(self):
        """
        Load your trained model.
        Make sure not to train a new model when self.mode == 'val'.
        [구현 완료]
        - 스켈레톤 코드의 로직을 그대로 사용합니다.
        - 'val' 모드일 때 PPO.load를 호출합니다.
        - 'train' 모드일 때 self.train()을 호출합니다.
        """
        if self.mode == "val":
            if not os.path.exists(self.model_path):
                raise AssertionError(f"Model not found: {self.model_path}")
            # [수정] PPO.load 호출
            self.model = PPO.load(self.model_path, device=self.device)

        elif self.mode == "train":
            # [수정 없음] 학습 모드일 경우 train 함수 호출
            self.train()

        else:
            raise AssertionError("mode should be one of 'train' or 'val'.")

    def get_action(self, scan_obs):
        """
        Predict action using obs - 'scan' data.
        Be sure to satisfy the limitation of steer and speed values.
        [구현 완료]
        - PPO.predict 메소드를 사용하여 행동을 결정합니다.
        - 평가 시에는 deterministic=True로 설정합니다.
        - 행동 값을 clip하여 반환합니다.
        """
        # [수정] PPO 모델의 predict 메소드 호출
        # (1, 720) 형태로 만들기 위해 reshape
        action_arr = self.model.predict(scan_obs.reshape(1, -1), deterministic=True)
        action_arr = action_arr[0] # (1, 2) -> (2,)

        # [수정] action을 steer와 speed로 분리하고 clip 적용
        steer = float(np.clip(action_arr[0], -self.max_steer, self.max_steer))
        speed = float(np.clip(action_arr[1], self.min_speed, self.max_speed))

        # [수정] env.step()에 맞는 (1, 2) 형태로 반환
        action = np.array([[steer, speed]])

        return action

    ################### CHANGE END  ###################
    ###################################################

    def query_callback(self, query_msg):

        id = query_msg.id
        team = query_msg.team
        map = query_msg.map
        trial = query_msg.trial
        _exit = query_msg.exit

        result_msg = Result()

        START_TIME = time.time()

        try:
            if team != TEAM_NAME:
                return

            if map not in self.maps:
                END_TIME = time.time()
                result_msg.id = id
                result_msg.team = team
                result_msg.map = map
                result_msg.trial = trial
                result_msg.time = END_TIME - START_TIME
                result_msg.waypoint = 0
                result_msg.n_waypoints = 20
                result_msg.success = False
                result_msg.fail_type = "Invalid Track"
                self.get_logger().info(">>> Invalid Track")
                self.result_pub.publish(result_msg)
                return

            self.get_logger().info(f"[{team}] START TO EVALUATE! MAP NAME: {map}")

            ### New environment
            env = RCCarWrapper(args=self.args, maps=[map], render_mode="human_fast" if self.render else None)
            track = env._env.unwrapped.track
            if self.render:
                env.unwrapped.add_render_callback(track.centerline.render_waypoints)

            obs, _ = env.reset(seed=self.args.seed)
            _, _, scan = obs

            step = 0
            terminate = False

            while True:
                act = self.get_action(scan)
                steer = np.clip(act[:, 0], -self.max_steer, self.max_steer)
                speed = np.clip(act[:, 1], self.min_speed, self.max_speed)

                obs, _, terminate, _, info = env.step(np.stack([steer, speed], axis=1))
                _, _, scan = obs
                step += 1

                if self.render:
                    env.render()

                if time.time() - START_TIME > self.time_limit:
                    END_TIME = time.time()
                    result_msg.id = id
                    result_msg.team = team
                    result_msg.map = map
                    result_msg.trial = trial
                    result_msg.time = step * self.dt
                    result_msg.waypoint = info["waypoint"]
                    result_msg.n_waypoints = 20
                    result_msg.success = False
                    result_msg.fail_type = "Time Out"
                    self.get_logger().info(f">>> Time Out: {map}")
                    self.result_pub.publish(result_msg)
                    env.close()
                    break

                if terminate:
                    END_TIME = time.time()
                    result_msg.id = id
                    result_msg.team = team
                    result_msg.map = map
                    result_msg.trial = trial
                    result_msg.time = step * self.dt
                    result_msg.waypoint = info["waypoint"]
                    result_msg.n_waypoints = 20
                    if info["waypoint"] == 20:
                        result_msg.success = True
                        result_msg.fail_type = "-"
                        self.get_logger().info(f">>> Success: {map}")
                    else:
                        result_msg.success = False
                        result_msg.fail_type = "Collision"
                        self.get_logger().info(f">>> Collision: {map}")
                    self.result_pub.publish(result_msg)
                    env.close()
                    break

        except Exception as e:
            END_TIME = time.time()
            result_msg.id = id
            result_msg.team = team
            result_msg.map = map
            result_msg.trial = trial
            result_msg.time = END_TIME - START_TIME
            result_msg.waypoint = 0
            result_msg.n_waypoints = 20
            result_msg.success = False
            result_msg.fail_type = "Script Error"
            error_message = traceback.format_exc()
            self.get_logger().error(f">>> Script Error - {str(e)}\n{error_message}")
            self.result_pub.publish(result_msg)

        if _exit:
            exit(0)
        return


def main():
    args = get_args()
    rclpy.init()
    node = RCCarPolicy(args)
    rclpy.spin(node)


if __name__ == "__main__":
    main()