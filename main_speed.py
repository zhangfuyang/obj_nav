import time
import torch
import numpy as np
from habitat.config.default import get_config as cfg_env
import random
import habitat

num_processes = 1


class VectorSingleEnv(habitat.VectorEnv):
    r"""VectorEnv with single Env on main Process, avoiding IPC overheads."""

    def __init__(
            self,
            make_env_fn,
            env_fn_args = None,
            auto_reset_done = True,
            multiprocessing_start_method = "forkserver",
            workers_ignore_signals = False,
    ):
        self._num_envs = len(env_fn_args)
        assert (self._num_envs == 1), "can only create 1 env"
        self._auto_reset_done = auto_reset_done
        self._is_closed = True
        self._env = make_env_fn(*env_fn_args[0])
        self._is_closed = False
        self._paused = []

        self.observation_spaces = [self._env.observation_space]
        self.action_spaces = [self._env.action_space]
        self.number_of_episodes = [self._env.number_of_episodes]

    def current_episodes(self):
        return [self._env.current_episode]

    def count_episodes(self):
        return [self._env.number_of_episodes]

    def episode_over(self):
        return [self._env.episode_over]

    def get_metrics(self):
        return [self._env.get_metrics()]

    def reset(self):
        return [self._env.reset()]

    def reset_at(self, index):
        assert (index == 0), "only valid for a single env"
        return [self._env.reset()]

    def step(self, data):
        action = data[0]
        if isinstance(action, (int, np.integer, str)):
            action = {"action": {"action": action}}
        observations, reward, done, info = self._env.step(**action)
        if self._auto_reset_done and done:
            observations = self._env.reset()
        return [(observations, reward, done, info)]

    def close(self):
        if self._is_closed:
            return
        self._env.close()
        self._is_closed = True

    def pause_at(self, index):
        self._paused = [index]  # TODO: hacky no-op for now

    def resume_all(self):
        self._paused = []  # TODO: hacky no-op for now

    def render(self, mode = "human", *args, **kwargs):
        images = self._env.render(args)  # TODO: actually test this code path
        tile = habitat.core.utils.tile_images(images)
        if mode == "human":
            cv2 = habitat.core.utils.try_cv2_import()
            cv2.imshow("vecenv", tile[:, :, ::-1])
            cv2.waitKey(1)
            return None
        elif mode == "rgb_array":
            return tile
        else:
            raise NotImplementedError


class Our_Agent_speed(habitat.RLEnv):
    def __init__(self, rank, config_env):
        self.rank = rank
        super().__init__(config_env)

        self.info = {}
        self.stopped = False
        # loading dataset info file
        self.split = config_env.DATASET.SPLIT
        self.episodes_dir = config_env.DATASET.EPISODES_DIR.format(
            split=self.split)

    @property
    def episode_over(self):
        return self.stopped

    def reset(self):
        self.stopped = False
        obs = super().reset()

        return obs, self.info

    def step(self, action):
        thread_step_time = time.time()
        assert self.stopped is False
        action = {'action': random.randint(0, 3)}
        if action['action'] == 0:
            self.stopped = True
        inner_step_time = time.time()
        obs, rew, done, _ = super().step(action)
        print('Thread {}, inner step fps:{:.2f}'.format(self.rank, 1/(time.time() - inner_step_time)))
        print('Thread {}, step fps:{:.2f}'.format(self.rank, 1/(time.time() - thread_step_time)))

        return obs, rew, done, self.info

    def get_info(self, observation):
        """This function is not used, Habitat-RLEnv requires this function"""
        info = {}
        return info

    def get_done(self, observation):
        if self.stopped:
            done = True
        else:
            done = False
        return done

    def get_reward_range(self):
        """This function is not used, Habitat-RLEnv requires this function"""
        return (0., 1.0)

    def get_reward(self, observations):
        return 0


def make_env_fn(config_env, rank):
    env = Our_Agent_speed(rank=rank, config_env=config_env)
    return env


def construct_envs():
    basic_config = cfg_env(
        config_paths=["envs/habitat/configs/tasks/objectnav_mp3d.yaml"])
    basic_config.defrost()
    basic_config.DATASET.SPLIT = 'val'
    basic_config.freeze()

    VectorEnvClass = habitat.VectorEnv if num_processes else VectorSingleEnv

    envs = VectorEnvClass(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip([basic_config for _ in range(num_processes)], range(num_processes))
            )
        ), auto_reset_done=False
    )

    return envs


def main():
    begin_time = time.time()

    envs = construct_envs()
    envs.reset()
    for step in range(10000):
        step_time = time.time()
        envs.step([{'action': 0} for _ in range(num_processes)])
        print('main step fps: {:.2f}'.format(1 / (time.time() - step_time)))
        for idx, done in enumerate(envs.episode_over()):
            if done:
                envs.reset_at(idx)

        # ------------------------------------------------------------------
    end_time = time.time()
    print('entire running spend: {} mins'.format((end_time-begin_time)/60))


if __name__ == "__main__":
    main()
