# -*- coding: utf-8 -*-
from collections import deque
import random
import cv2
import torch
import numpy as np
import time
import datetime
import pathlib
import gym
from .abstract_game import AbstractGame

class Env():
    def __init__(self, env):
        self.device = "cpu"
        actions = np.array([0, 1, 3, 4, 11, 12])
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0
        self.window = 4
        self.state_buffer = deque([], maxlen=4)
        self.training = False
        self.screen = None
        self.env = env
        self.done = None
        self.info = None
        # self.players = env.players
        self.players = 1
        
    def seed(self, seed):
        self.env.seed(seed)
        
    def interact(self, act0):
        act = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        if act0 == 2 or act0 == 4:
            act[4] = 1
        if act0 == 3 or act0 == 5:
            act[5] = 1
        act1 = act
        obs, rew, done, info = self.env.step(act1)
        self.done = done
        self.screen = obs
        self.info = info
        return np.array(rew)

    def interact_2P(self, act01, act02):
        act = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        if act01 == 2 or act01 == 4:
            act[4] = 1
        if act01 == 3 or act01 == 5:
            act[5] = 1
        if act02 == 2 or act02 == 4:
            act[6] = 1
        if act02 == 3 or act02 == 5:
            act[7] = 1
        act1 = act
        obs, rew, done, info = self.env.step(act1)
        self.done = done
        self.screen = obs
        self.info = info
        return np.array(rew)

    def _get_state(self):
        state = cv2.resize(cv2.cvtColor(self.screen, cv2.COLOR_BGR2GRAY), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        self._reset_buffer()
        self.env.reset()
        for _ in range(random.randrange(30)):
            self.interact(0)
            if self.done:
                self.env.reset()
        observation = self._get_state()
        self.state_buffer.append(observation)
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        if self.players == 1:
            reward, done = 0, False
        if self.players == 2:
            reward, done = np.array([0.0, 0.0]), False
        for t in range(4):
            reward += self.interact(action)
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.done
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        return torch.stack(list(self.state_buffer), 0), reward, done, self.info

    def step_2P(self, action1, action2):
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        if self.players == 1:
            reward, done = 0, False
        if self.players == 2:
            reward, done = np.array([0.0, 0.0]), False
        for t in range(4):
            reward += self.interact_2P(action1, action2)
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.done
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        return torch.stack(list(self.state_buffer), 0), reward, done, self.info

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        self.env.render()
        time.sleep(0.01)

    def close(self):
        cv2.destroyAllWindows()

class MuZeroConfig:
    def __init__(self):
        self.seed = 0
        self.max_num_gpus = None
        self.observation_shape = (4, 84, 84)
        self.action_space = list(range(6))
        self.players = list(range(1))
        self.stacked_observations = 0
        self.muzero_player = 0
        self.opponent = None
        self.num_workers = 1
        self.selfplay_on_gpu = False
        self.max_moves = 500
        self.num_simulations = 50
        self.discount = 0.997
        self.temperature_threshold = None
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.network = "resnet"
        self.support_size = 10
        self.downsample = False
        self.blocks = 1
        self.channels = 2
        self.reduced_channels_reward = 2
        self.reduced_channels_value = 2
        self.reduced_channels_policy = 2
        self.resnet_fc_reward_layers = []
        self.resnet_fc_value_layers = []
        self.resnet_fc_policy_layers = []
        self.encoding_size = 8
        self.fc_representation_layers = []
        self.fc_dynamics_layers = [16]
        self.fc_reward_layers = [16]
        self.fc_value_layers = [16]
        self.fc_policy_layers = [16]
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.save_model = True
        self.training_steps = 10000
        self.batch_size = 128
        self.checkpoint_interval = 10
        self.value_loss_weight = 1
        self.train_on_gpu = torch.cuda.is_available()
        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.lr_init = 0.02
        self.lr_decay_rate = 0.8
        self.lr_decay_steps = 1000
        self.replay_buffer_size = 500
        self.num_unroll_steps = 10
        self.td_steps = 50
        self.PER = True
        self.PER_alpha = 0.5
        self.use_last_model_value = True
        self.reanalyse_on_gpu = False
        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = 1.5

    def visit_softmax_temperature_fn(self, trained_steps):
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25

class Game(AbstractGame):
    def __init__(self, seed=None):
        self.env = Env(gym.make("Pong-v0"))
        if seed is not None:
            self.env.seed(seed)

    def step(self, action):
        observation, reward, done, _ = self.env.step(action)
        return observation, reward, done

    def legal_actions(self):
        return list(range(6))

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()
        input("Press enter to take a step ")

    def action_to_string(self, action_number):
        actions = {
            0: "No-op",
            1: "Fire",
            2: "Up",
            3: "Right",
            4: "Left",
            5: "Down",
        }
        return f"{action_number}. {actions[action_number]}"