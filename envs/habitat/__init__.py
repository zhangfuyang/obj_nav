# Parts of the code in this file have been borrowed from:
#    https://github.com/facebookresearch/habitat-api
import os
import numpy as np
import torch
import habitat
from habitat.config.default import get_config as cfg_env
from habitat import make_dataset

from agents.our_agent import Our_Agent


def make_env_fn(args, config_env, rank):
    dataset = make_dataset(config_env.DATASET.TYPE, config=config_env.DATASET)
    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    config_env.freeze()

    env = Our_Agent(args=args, rank=rank, config_env=config_env, dataset=dataset)

    env.seed(rank)
    return env


def _get_scenes_from_folder(content_dir):
    scene_dataset_ext = ".json.gz"
    scenes = []
    for filename in os.listdir(content_dir):
        if filename.endswith(scene_dataset_ext):
            scene = filename[: -len(scene_dataset_ext)]
            scenes.append(scene)
    scenes.sort()
    return scenes


def construct_envs(args):
    env_configs = []
    args_list = []

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

    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )

        scene_split_sizes = [int(np.floor(len(scenes) / args.num_processes))
                             for _ in range(args.num_processes)]
        for i in range(len(scenes) % args.num_processes):
            scene_split_sizes[i] += 1

    print("Scenes per thread:")
    for i in range(args.num_processes):
        config_env = cfg_env(config_paths=["envs/habitat/configs/"
                                           + args.task_config])
        config_env.defrost()

        if len(scenes) > 0:
            config_env.DATASET.CONTENT_SCENES = scenes[
                sum(scene_split_sizes[:i]):
                sum(scene_split_sizes[:i + 1])
            ]
            print("Thread {}: {}".format(i, config_env.DATASET.CONTENT_SCENES))

        if i < args.num_processes_on_first_gpu:
            gpu_id = 0
        else:
            gpu_id = int((i - args.num_processes_on_first_gpu)
                         // args.num_processes_per_gpu) + args.sim_gpu_id
        gpu_id = min(torch.cuda.device_count() - 1, gpu_id)
        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id

        agent_sensors = []
        agent_sensors.append("RGB_SENSOR")
        agent_sensors.append("DEPTH_SENSOR")
        if args.use_gt_obj:
            agent_sensors.append("SEMANTIC_SENSOR")

        config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors

        # Reseting episodes manually, setting high max episode length in sim
        config_env.ENVIRONMENT.MAX_EPISODE_STEPS = 10000000
        config_env.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True
        config_env.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE = True

        config_env.SIMULATOR.RGB_SENSOR.WIDTH = args.frame_width
        config_env.SIMULATOR.RGB_SENSOR.HEIGHT = args.frame_height
        config_env.SIMULATOR.RGB_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.RGB_SENSOR.POSITION = [0, args.camera_height, 0]

        config_env.SIMULATOR.DEPTH_SENSOR.WIDTH = args.frame_width
        config_env.SIMULATOR.DEPTH_SENSOR.HEIGHT = args.frame_height
        config_env.SIMULATOR.DEPTH_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = args.min_depth
        config_env.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = args.max_depth
        config_env.SIMULATOR.DEPTH_SENSOR.POSITION = [0, args.camera_height, 0]
        config_env.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = True

        if args.use_gt_obj:
            config_env.SIMULATOR.SEMANTIC_SENSOR.WIDTH = args.frame_width
            config_env.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = args.frame_height
            config_env.SIMULATOR.SEMANTIC_SENSOR.HFOV = args.hfov
            config_env.SIMULATOR.SEMANTIC_SENSOR.POSITION = \
                [0, args.camera_height, 0]

        config_env.SIMULATOR.TURN_ANGLE = args.turn_angle
        config_env.DATASET.SPLIT = args.split
        config_env.DATASET.DATA_PATH = \
            config_env.DATASET.DATA_PATH.replace("v1", args.version)
        config_env.DATASET.EPISODES_DIR = \
            config_env.DATASET.EPISODES_DIR.replace("v1", args.version)

        config_env.freeze()
        env_configs.append(config_env)

        args_list.append(args)

    VectorEnvClass = habitat.VectorEnv if args.num_processes > 1 else VectorSingleEnv
    envs = VectorEnvClass(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(args_list, env_configs, range(args.num_processes))
            )
        ), auto_reset_done=False
    )

    return envs


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


class EnvWrap():
    def __init__(self, envs, device):
        self._envs = envs
        self._device = device

    def reset(self):
        x = self._envs.reset()
        obs, infos = zip(*x)
        return torch.from_numpy(np.stack(obs)).float().to(self._device), list(infos)

    def step(self, actions):
        x = self._envs.step(actions)
        obs, rews, dones, infos = zip(*x)
        return torch.from_numpy(np.stack(obs)).float().to(self._device), \
               torch.from_numpy(np.stack(rews)).float().to(self._device),\
               torch.from_numpy(np.stack(dones)).float().to(self._device),\
               list(infos)

    def reset_at(self, idx):
        x = self._envs.reset_at(idx)[0]
        obs, info = x
        return torch.from_numpy(obs).float().to(self._device), info
