import cv2
import numpy as np
import os
import glob

file_dir = 'tmp_final_v3/dump/exp1'
exp_episodes = file_dir + '/episodes'
output_dir = file_dir + '/video'
fps = 3
for thread in os.listdir(exp_episodes):
    exp_folder = os.path.join(exp_episodes, thread)
    for exp_name in os.listdir(exp_folder):
        images_path = os.path.join(exp_folder, exp_name)
        with open(os.path.join(images_path,'info.txt'), 'r') as f:
            a = f.readlines()
            s = a[0].strip().split(':')[1]
            if int(s) == 0:
                success = False
            else:
                success = True
            s = a[1].strip().split(':')[1].strip()
            target = s
            scene_name = a[2].strip().split(':')[1].strip().split('/')[-2].strip()
            eps_id = a[3].strip().split(':')[1].strip()


        filenames = sorted(glob.glob(os.path.join(images_path, '*.png')),
                           key=lambda name: int(name.split('-')[-1].split('.')[0]))
        img_array = []
        for filename in filenames:
            img = cv2.imread(filename)
            h,w,l = img.shape
            size = (w,h)
            img_array.append(img)

        os.makedirs(os.path.join(output_dir, target), exist_ok=True)
        os.makedirs(os.path.join(output_dir, target, 'success'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, target, 'failure'), exist_ok=True)
        video_name = thread+'_'+exp_name+'_'+scene_name+'_'+eps_id+'.avi'
        if success:
            out = cv2.VideoWriter(os.path.join(output_dir, target, 'success', video_name),cv2.VideoWriter_fourcc(*'DIVX'),  fps, size)
        else:
            out = cv2.VideoWriter(os.path.join(output_dir, target, 'failure', video_name),cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        for img in img_array:
            out.write(img)
        out.release()
        print(images_path)