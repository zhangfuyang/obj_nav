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
    for step in range(args.num_training_frames // args.num_processes + 1):
        step_time = time.time()
        obs, reward, done, infos = \
            envs.step([{'action': 0} for _ in range(args.num_processes)])
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

        vis = obs[0, :3].transpose(1,2,0).astype(np.uint8)
        vis = vis[..., ::-1]
        vis = np.concatenate((vis, np.zeros((vis.shape[0], 5, 3)).astype(np.uint8)), 1)
        obj_semantic_one_hot = obs[0, 4:4+len(category2objectid)]
        color_palette = np.array([
            (255, 179, 0), (128, 62, 117), (255, 104, 0),
            (166, 189, 215), (193, 0, 32), (206, 162, 98),
            (129, 112, 102), (0, 125, 52), (246, 118, 142),
            (0, 83, 138), (255, 122, 92), (83, 55, 122),
            (255, 142, 0), (179, 40, 81), (244, 200, 0),
            (127, 24, 13), (147, 170, 0), (89, 51, 21),
            (241, 58, 19), (35, 44, 22), (255, 255, 255)
        ])
        color_palette = color_palette[:, ::-1]
        object_mask = np.zeros((obj_semantic_one_hot.shape[1], obj_semantic_one_hot.shape[2], 3))
        for object_i in range(obj_semantic_one_hot.shape[0]):
            object_mask += color_palette[object_i][np.newaxis, np.newaxis, ...] * obj_semantic_one_hot[object_i][..., np.newaxis]
        vis = np.concatenate((vis, object_mask.astype(np.uint8)), 1)

        depth = obs[0, 3]

        vis = np.concatenate((vis, np.ones((60, vis.shape[1], 3), dtype=np.uint8)), 0)

        x_loc = 20
        for idx, region_type_name in enumerate(category2objectid.keys()):
            cv2.putText(vis, str(idx),
                        (x_loc, vis.shape[0]-10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255),
                        thickness=1)
            cv2.rectangle(vis, (x_loc, vis.shape[0]-40),
                          (x_loc+20, vis.shape[0]-30),
                          (color_palette[idx]).astype(np.uint).tolist(),
                          thickness=-1)
            x_loc += 60

        cv2.imshow("vis", vis)
        cv2.waitKey(1)
        plt.imshow(depth)
        plt.show()

        # ------------------------------------------------------------------


    end_time = time.time()
    print('entire running spend: {} mins'.format((end_time-begin_time)/60))


if __name__ == "__main__":
    main()
