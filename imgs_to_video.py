import cv2
import numpy as np
import os
import glob

file_dir = 'result'
exp_episodes = file_dir + '/episodes'
output_dir = 'video'
fps = 3
os.makedirs(output_dir, exist_ok=True)
for thread in os.listdir(exp_episodes):
    exp_folder = os.path.join(exp_episodes, thread)
    for exp_name in os.listdir(exp_folder):
        images_path = os.path.join(exp_folder, exp_name)
        filenames = sorted(glob.glob(os.path.join(images_path, '*.png')),
                           key=lambda name: int(name.split('/')[-1].split('.')[0]))
        img_array = []
        for filename in filenames:
            img = cv2.imread(filename)
            h,w,l = img.shape
            size = (w,h)
            img_array.append(img)

        video_name = thread+'_'+exp_name+'.avi'
        out = cv2.VideoWriter(os.path.join(output_dir, video_name),cv2.VideoWriter_fourcc(*'DIVX'),  fps, size)
        for img in img_array:
            out.write(img)
        out.release()
        print(images_path)