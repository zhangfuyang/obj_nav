import os
import time
import json
import gym
import torch
import numpy as np
from envs.habitat import construct_envs, EnvWrap
from arguments import get_args
from constants import category2objectid, mp3d_region_id2name
import matplotlib.pyplot as plt
import cv2
from sem_map import Semantic_Mapping
from vis_utils import visualization

os.environ["OMP_NUM_THREADS"] = "1"


def get_new_pose_batch(pose, rel_pose_change):
    pose[:, 1] += rel_pose_change[:, 0] * \
                  torch.sin(pose[:, 2] / 57.29577951308232) \
                  + rel_pose_change[:, 1] * \
                  torch.cos(pose[:, 2] / 57.29577951308232)
    pose[:, 0] += rel_pose_change[:, 0] * \
                  torch.cos(pose[:, 2] / 57.29577951308232) \
                  - rel_pose_change[:, 1] * \
                  torch.sin(pose[:, 2] / 57.29577951308232)
    pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

    pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
    pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

    return pose


def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Logging and loss variables
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")
    begin_time = time.time()

    # initial modules
    semantic_map_module = Semantic_Mapping(args)
    semantic_map_module.eval()

    # Calculating semantic map sizes
    map_size = args.map_size_cm // args.map_resolution
    map_h, map_w = map_size, map_size

    # initial semantic_map
    # 0: obstacle map
    # 1: explored area
    # 2: current agent location
    # 3: past agent location
    # 4,5,6,.. : semantic
    num_scenes = args.num_processes
    semantic_map_dim = 4 + len(category2objectid.keys()) * args.use_obj + \
              len(mp3d_region_id2name.keys()) * args.use_region
    semantic_map = torch.zeros(num_scenes, semantic_map_dim, map_h, map_w).float().to(device)

    # set initial agent location in semantic_map
    initial_r, initial_l = semantic_map.shape[2] // 2, semantic_map.shape[3] // 2
    semantic_map[:, 2:4, initial_r-1:initial_r+2, initial_l-1:initial_l+2] = 1.0

    # Initial agent pose to the center of the semantic map
    agent_pose_m = torch.zeros(num_scenes, 3).float().to(device)
    agent_pose_m[:, 0] = args.map_size_cm / 100 / 2
    agent_pose_m[:, 1] = args.map_size_cm / 100 / 2

    # Starting environments
    envs = construct_envs(args)
    envs = EnvWrap(envs, device)
    obs, infos = envs.reset()
    semantic_map = semantic_map_module(obs, semantic_map, agent_pose_m)

    for e in range(num_scenes):
        img = obs[e,:3].cpu().numpy().transpose(1,2,0)[...,::-1].astype(np.uint8)
        visualization(title='Thread {}'.format(e), goal_name=infos[e]['goal_name'],
                      img=img, semantic_map=semantic_map[e],
                      agent_pose_m=agent_pose_m[e], arg=args)

    for step in range(args.num_training_frames // args.num_processes + 1):
        step_time = time.time()
        goal_map = []
        if args.agent_type != 'model':
            goal_map.append(None)
            obs, reward, done, infos = \
                envs.step([{'action': 0, } for _ in range(args.num_processes)])
        else:
            # use action to pass info
            real_target = []
            for i in range(args.num_processes):
                obj_map = semantic_map[i, infos[i]['goal_cat_id'] + 4].cpu().numpy()
                if obj_map.sum() <= 0:
                    real_target.append(False)
                    target_map = np.zeros_like(obj_map)
                    target_map[0:10,0:10] = 1
                    #target_map[np.random.randint(obj_map.shape[0]),
                    #           np.random.randint(obj_map.shape[1])] = 1
                    goal_map.append(target_map)
                else:
                    goal_map.append(obj_map)
                    real_target.append(True)
            obs, reward, done, infos = \
                envs.step([{'action': {
                    'obstacle_map': semantic_map[i,0].cpu().numpy(),
                    'goal_map': goal_map[i],
                    'pose_m': agent_pose_m[i].cpu().numpy(),
                    'real_target': real_target[i]}
                } for i in range(args.num_processes)])

        print('main step fps: {:.2f}'.format(1 / (time.time() - step_time)))

        # accumulate agent pose
        poses_change = torch.from_numpy(np.asarray(
            [infos[e]['sensor_pose'] for e in range(num_scenes)])
        ).float().to(device)

        # update pose
        agent_pose_m = get_new_pose_batch(agent_pose_m, poses_change)

        semantic_map = semantic_map_module(obs, semantic_map, agent_pose_m)

        for e, x in enumerate(done):
            if x:
                obs_r, info_r = envs.reset_at(e)
                obs[e] = obs_r
                infos[e] = info_r
                print('reset')

                # reset agent_pose_m
                agent_pose_m[e, 0] = args.map_size_cm / 100 / 2
                agent_pose_m[e, 1] = args.map_size_cm / 100 / 2

                # reset semantic map
                semantic_map[e] = 0
                # set initial agent location in semantic_map
                initial_r, initial_l = semantic_map.shape[2] // 2, semantic_map.shape[3] // 2
                semantic_map[e, 2:4, initial_r-1:initial_r+2, initial_l-1:initial_l+2] = 1.0
                semantic_map[e] = semantic_map_module(obs[e:e+1], semantic_map[e:e+1], agent_pose_m[e:e+1])[0]

        for e in range(num_scenes):
            img = obs[e,:3].cpu().numpy().transpose(1,2,0)[...,::-1].astype(np.uint8)
            visualization(title='Thread {}'.format(e), goal_name=infos[e]['goal_name'],
                          img=img, semantic_map=semantic_map[e], goal_map=goal_map[e],
                          agent_pose_m=agent_pose_m[e], arg=args)

        # ------------------------------------------------------------------


    end_time = time.time()
    print('entire running spend: {} mins'.format((end_time-begin_time)/60))


if __name__ == "__main__":
    main()
