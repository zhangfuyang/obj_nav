import habitat
import numpy as np
import cv2
from arguments import get_args
import time
from envs.habitat import construct_envs
from habitat.config.default import get_config as cfg_env
import os
from habitat import make_dataset


class test_env(habitat.RLEnv):
    def __init__(self, config_env, dataset):
        super().__init__(config_env, dataset)

        # loading dataset info file
        self.split = config_env.DATASET.SPLIT

        self.info = {}

        # episode
        self.timestep = 0
        self.stopped = False
        self.path_length = 1e-5
        self.trajectory_states = []
        self.last_sim_location = ()

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


def _get_scenes_from_folder(content_dir):
    scene_dataset_ext = ".json.gz"
    scenes = []
    for filename in os.listdir(content_dir):
        if filename.endswith(scene_dataset_ext):
            scene = filename[: -len(scene_dataset_ext)]
            scenes.append(scene)
    scenes.sort()
    return scenes

def make_env_fn(config_env, rank):
    dataset = make_dataset(config_env.DATASET.TYPE, config=config_env.DATASET)

    env = test_env(config_env=config_env, dataset=dataset)

    return env


def main():
    args = get_args()
    basic_config = cfg_env(config_paths=["envs/habitat/configs/"
                                         + args.task_config])
    basic_config.defrost()
    basic_config.DATASET.SPLIT = args.split
    basic_config.DATASET.DATA_PATH = \
        basic_config.DATASET.DATA_PATH.replace("v1", args.version)
    basic_config.DATASET.EPISODES_DIR = \
        basic_config.DATASET.EPISODES_DIR.replace("v1", args.version)
    basic_config.freeze()

    scenes = basic_config.DATASET.CONTENT_SCENES
    if "*" in basic_config.DATASET.CONTENT_SCENES:
        content_dir = os.path.join(basic_config.DATASET.EPISODES_DIR.format(
            split=args.split), "content")
        scenes = _get_scenes_from_folder(content_dir)

    basic_config.defrost()
    basic_config.DATASET.CONTENT_SCENES = scenes
    basic_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = 0
    agent_sensors = []
    agent_sensors.append("RGB_SENSOR")
    agent_sensors.append("DEPTH_SENSOR")
    basic_config.SIMULATOR.AGENT_0.SENSORS = agent_sensors

    basic_config.SIMULATOR.RGB_SENSOR.WIDTH = args.frame_width
    basic_config.SIMULATOR.RGB_SENSOR.HEIGHT = args.frame_height
    basic_config.SIMULATOR.RGB_SENSOR.HFOV = args.hfov
    basic_config.SIMULATOR.RGB_SENSOR.POSITION = [0, args.camera_height, 0]

    basic_config.SIMULATOR.DEPTH_SENSOR.WIDTH = args.frame_width
    basic_config.SIMULATOR.DEPTH_SENSOR.HEIGHT = args.frame_height
    basic_config.SIMULATOR.DEPTH_SENSOR.HFOV = args.hfov
    basic_config.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = args.min_depth
    basic_config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = args.max_depth
    basic_config.SIMULATOR.DEPTH_SENSOR.POSITION = [0, args.camera_height, 0]
    basic_config.freeze()

    #env = test_env(basic_config, dataset)
    #env = habitat.Env(basic_config)
    vvv = VectorSingleEnv
    env = vvv(make_env_fn=make_env_fn, env_fn_args=tuple(tuple(zip([basic_config], range(1)))),
                            auto_reset_done=False)
    obs = env.reset()

    for step in range(1000000):
        step_time = time.time()
        obs = env.step([{'action':1}])
        print('main step fps: {:.2f}'.format(1 / (time.time() - step_time)))
        if step % 20 == 0:
            obs = env.reset()
        #print(done)

        # ------------------------------------------------------------------


if __name__ == "__main__":
    main()


