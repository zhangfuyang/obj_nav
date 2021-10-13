import os
import time
import json
import gym
import torch
import numpy as np
from envs.habitat import construct_envs
from arguments import get_args
from constants import coco_categories
import matplotlib.pyplot as plt
import cv2

os.environ["OMP_NUM_THREADS"] = "1"


def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Logging and loss variables
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")
    begin_time = time.time()

    # Starting environments
    envs = construct_envs(args)
    obs, infos = envs.reset()

    finished = np.zeros((args.num_processes))
    for step in range(args.num_training_frames // args.num_processes + 1):
        if finished.sum() == args.num_processes:
            break

        step_time = time.time()
        obs, reward, done, infos = envs.step([{'action': 0} for _ in range(args.num_processes)])
        print('main step fps: {:.2f}'.format(1 / (time.time() - step_time)))

        for e, x in enumerate(done):
            if x:
                obs_r, info_r = envs.reset_at(e)[0]
                obs[e] = obs_r
                infos[e] = info_r
                print('reset')

        #vis = obs[0, :3].transpose(1,2,0).astype(np.uint8)
        #vis = vis[..., ::-1]
        #obj_semantic_one_hot = obs[0, 4:4+len(coco_categories)]
        #cv2.imshow(infos[0]['goal_name'], img[...,::-1])
        #cv2.waitKey(1)

        # ------------------------------------------------------------------


    end_time = time.time()
    print('entire running spend: {} mins'.format((end_time-begin_time)/60))


if __name__ == "__main__":
    main()
