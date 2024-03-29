import habitat
import gym
from torchvision import transforms
from PIL import Image
from agents.utils.semantic_prediction import SemanticPredMaskRCNN
from file_utils import parse_house
from constants import category2objectid, objid_mp3did, mp3d_region_label2id, mp3d_region_id2name
import numpy as np
import quaternion
import cv2
import random
import matplotlib.path as mpltPath
import time
import envs.utils.pose as pu
import skimage.morphology
from fmm_planner import FMMPlanner
import math
import matplotlib.pyplot as plt


class Our_Agent(habitat.RLEnv):
    def __init__(self, args, rank, config_env, dataset):
        self.args = args
        self.rank = rank
        super().__init__(config_env, dataset)

        # loading dataset info file
        self.split = config_env.DATASET.SPLIT
        self.episodes_dir = config_env.DATASET.EPISODES_DIR.format(
            split=self.split)

        # initialize semantic segmentation prediction model
        if self.args.use_gt_obj == 0:
            #TODO add semantic model
            self.sem_pred = None
            #self.sem_pred = SemanticPredMaskRCNN(args)
        else:
            self.sem_pred = None

        # specifying action and observation space
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(0, 255,
                                                (3, args.frame_height,
                                                 args.frame_width),
                                                dtype='uint8')

        ### initialization
        self.episode_no = 0

        # scene info
        self.scene_name = ""
        self.info = {}

        # episode
        self.timestep = 0
        self.stopped = False
        self.path_length = 1e-5
        self.trajectory_states = []
        self.last_sim_location = ()

        # gt
        if self.args.use_gt_obj or self.args.use_gt_region:
            self.gt_annotation = {} # annotation
        else:
            self.gt_annotation = None
        if self.args.use_gt_obj and self.args.use_obj:
            H, W = args.frame_height, args.frame_width
            self.obj_semantic_extract_pattern = \
                np.array([t[1] for t in objid_mp3did])[np.newaxis, np.newaxis, ...] * np.ones((H,W,1))
        if self.args.use_gt_region:
            self.region_semantic_info = {} # use for get region semantic information
        else:
            self.region_semantic_info = None

        # used for choosing one shortest path when using gt agent
        self.gt_shortest_path = None

        # heuristic action
        self.selem = skimage.morphology.disk(3)
        self.last_action = -1
        self.col_width = 1 # used for labeling the collision place
        map_size = args.map_size_cm // args.map_resolution
        self.collision_map = np.zeros((map_size, map_size))
        self.visited = np.zeros((map_size, map_size))
        self.prev_pos_m = np.zeros(3)
        self.prev_pos_m[0] = args.map_size_cm / 100 / 2
        self.prev_pos_m[1] = args.map_size_cm / 100 / 2

    def reset(self):
        """
        Returns:
            obs (ndarray): observation (ndims x H x W): RGB+depth+(object prediction)+(room prediction)
            info (dict): contains timestep, pose, goal category, etc
        """
        self.episode_no += 1

        # initialization of episode
        if self.scene_name != self.habitat_env.sim.habitat_config.SCENE:
            new_scene = True
        else:
            new_scene = False
        self.scene_name = self.habitat_env.sim.habitat_config.SCENE
        self.timestep = 0
        self.stopped = False
        self.path_length = 1e-5
        self.trajectory_states = []

        # store gt if needed
        if self.gt_annotation is not None:
            if self.scene_name.split('/')[-2] not in self.gt_annotation.keys():
                file_name = self.scene_name.split('.')[0] + '.house'
                house = parse_house(file_name)
                self.gt_annotation[self.scene_name.split('/')[-2]] = house

        ### set obs
        obs = super().reset()
        self.last_sim_location = self.get_sim_location()

        # set shortest_path for gt agent
        if self.args.agent_type == 'gt':
            shortest_paths = self.current_episode.shortest_paths
            self.gt_shortest_path = random.choice(shortest_paths)

        # preprocessing depth
        #for i in range(obs['depth'].shape[1]):
        #    obs['depth'][:, i, 0][obs['depth'][:, i, 0] == 0.] = obs['depth'][:, i, 0].max()

        # process obj semantic
        if self.args.use_obj:
            if self.args.use_gt_obj:
                raw_semantic = obs['semantic']
                # TODO save into file, read from file
                if new_scene:
                    # update mapping if first time in this scene
                    scene = self.habitat_env.sim.semantic_annotations()
                    instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
                    self.mapping = np.array([ instance_id_to_label_id[i] for i in range(len(instance_id_to_label_id)) ])
                # remove noise obj
                remove_id = np.unique(raw_semantic[obs['depth'][...,0] < 0.3])
                mapping = self.mapping.copy()
                mapping[remove_id] = -1

                obj_semantic = np.take(mapping, raw_semantic)
                depth = obs['depth'][:,:,0]
                obj_semantic[depth > self.args.max_gt_obj] = -1 # too far, set to -1
                obs['obj_semantic'] = (
                        self.obj_semantic_extract_pattern == obj_semantic[..., np.newaxis]).astype(np.float)
            else:
                #TODO use maskrcnn
                H, W = obs['depth'].shape[0], obs['depth'].shape[1]
                obs['obj_semantic'] = np.zeros((H,W,len(category2objectid.keys())))
        # process region semantic
        if self.args.use_region:
            if self.args.use_gt_region:
                ### initialize this new episode
                agent_sim_loc = self.habitat_env.sim.get_agent_state(0).position
                agent_sim_rot = self.habitat_env.sim.get_agent_state(0).rotation
                initial_pos = np.array((-agent_sim_loc[2], -agent_sim_loc[0]))
                initial_axis = quaternion.as_euler_angles(agent_sim_rot)[0]
                agent_pose_m_np = np.zeros(3)
                agent_pose_m_np[0] = self.args.map_size_cm / 100. / 2
                agent_pose_m_np[1] = self.args.map_size_cm / 100. / 2
                initial_info = {'agent_sim_loc': agent_sim_loc, 'agent_sim_rot': agent_sim_rot,
                                'initial_pos': initial_pos, 'initial_axis': initial_axis,
                                'agent_pose_m_np': agent_pose_m_np}
                self.region_semantic_info['initial_info'] = initial_info
                #TODO cache
                gt_regions_coord = self.get_regions_in_one_level() # get all the region in the same level
                H = W = self.args.map_size_cm // self.args.map_resolution
                gt_region_mask = np.zeros((H, W)) ## gt_region_mask is the mask for calculate region segmantation
                for (region_cm, region_type_id) in gt_regions_coord:
                    if region_type_id == 0:
                        continue
                    polygon = (region_cm * 100. / self.args.map_resolution).astype(np.int)
                    gt_region_mask = cv2.drawContours(gt_region_mask, [polygon], 0,
                                                      region_type_id, -1)
                    # TODO fix small hole and undefined place
                self.region_semantic_info['gt_region_mask'] = gt_region_mask
                ### finish initialization

                xyz = self.depth2point(obs['depth'])
                obs['region_semantic'] = self.point2region_semantic(xyz)
            else:
                # TODO use prediction regions
                H, W = obs['depth'].shape[0], obs['depth'].shape[1]
                obs['region_semantic'] = np.zeros((H, W, len(mp3d_region_id2name.keys())))

        features_name = ['rgb', 'depth']
        if self.args.use_obj:
            features_name += ['obj_semantic']
        if self.args.use_region:
            features_name += ['region_semantic']
        obs_concat = np.concatenate([obs[name] for name in features_name], axis=2).transpose(2,0,1)

        ### set info
        self.info['time'] = self.timestep
        self.info['sensor_pose'] = [0., 0., 0.]
        self.info['goal_cat_id'] = category2objectid[self.current_episode.object_category]
        self.info['goal_name'] = self.current_episode.object_category
        self.info['scene_name'] = self.current_episode.scene_id
        self.info['episode_id'] = self.current_episode.episode_id

        ### reset heuristic action
        self.last_action = -1
        self.col_width = 1
        self.collision_map[:] = 0
        self.visited[:] = 0
        self.prev_pos_m[0] = self.args.map_size_cm / 100 / 2
        self.prev_pos_m[1] = self.args.map_size_cm / 100 / 2
        self.prev_pos_m[2] = 0

        return obs_concat, self.info

    def step(self, action):
        """
        Args:
            action: dict with following keys:
                    'action' (int): 0 stop, 1 forward, 2 left, 3 right
        Returns:
            obs (ndarray): observation (ndims x H x W): RGB+depth+(object prediction)+(room prediction)
            reward (float):
            done (bool):
            info (dict): contains timestep, pose, goal category, etc
        """
        thread_step_time = time.time()
        assert self.stopped is False
        if self.args.agent_type == 'random':
            action = {'action': random.randint(0, 3)}
        elif self.args.agent_type == 'gt':
            action = self.gt_shortest_path[self.timestep].action
            action = {'action': 0 if action is None else action}
        else:
            obstacle_map, goal_map, pose_m, real_target = \
                action['obstacle_map'], action['goal_map'], action['pose_m'], action['real_target']
            action = {'action': self._plan(obstacle_map, goal_map, pose_m, real_target)}
            # model
            pass
        if action['action'] == 0:
            self.stopped = True
        inner_step_time = time.time()
        self.last_sim_location = self.get_sim_location()
        obs, rew, done, super_info = super().step(action)
        for key in super_info.keys():
            self.info[key] = super_info[key]
        print('Thread {}, inner step fps:{:.2f}'.format(self.rank, 1/(time.time() - inner_step_time)))
        #preprocessing depth
        #for i in range(obs['depth'].shape[1]):
        #    obs['depth'][:, i][obs['depth'][:, i] == 0.] = obs['depth'][:, i].max()

        self.timestep += 1
        self.trajectory_states.append(action)
        dx, dy, do = self.get_pose_change()
        self.info['sensor_pose'] = [dx, dy, do]
        self.path_length += pu.get_l2_distance(0, dx, 0, dy)

        # process obj semantic
        if self.args.use_obj:
            if self.args.use_gt_obj:
                raw_semantic = obs['semantic']
                # remove noise obj
                remove_id = np.unique(raw_semantic[obs['depth'][...,0] < 0.3])
                mapping = self.mapping.copy()
                mapping[remove_id] = -1

                obj_semantic = np.take(mapping, raw_semantic)
                depth = obs['depth'][:,:,0]
                obj_semantic[depth > self.args.max_gt_obj] = -1 # too far

                obs['obj_semantic'] = (
                        self.obj_semantic_extract_pattern == obj_semantic[..., np.newaxis]).astype(np.float)
            else:
                #TODO use maskrcnn
                H, W = obs['depth'].shape[0], obs['depth'].shape[1]
                obs['obj_semantic'] = np.zeros((H,W,len(category2objectid.keys())))
        # process region semantic
        if self.args.use_region:
            if self.args.use_gt_region:
                xyz = self.depth2point(obs['depth'])
                obs['region_semantic'] = self.point2region_semantic(xyz)
            else:
                # TODO use prediction regions
                H, W = obs['depth'].shape[0], obs['depth'].shape[1]
                obs['region_semantic'] = np.zeros((H, W, len(mp3d_region_id2name.keys())))

        features_name = ['rgb', 'depth']
        if self.args.use_obj:
            features_name += ['obj_semantic']
        if self.args.use_region:
            features_name += ['region_semantic']
        obs_concat = np.concatenate([obs[name] for name in features_name], axis=2).transpose(2,0,1)
        print('Thread {}, step fps:{:.2f}'.format(self.rank, 1/(time.time() - thread_step_time)))

        return obs_concat, rew, done, self.info

    def get_info(self, observation):
        return self.habitat_env.get_metrics()

    def get_done(self, observation):
        if self.timestep >= self.args.max_episode_length - 1:
            done = True
        elif self.stopped:
            done = True
        else:
            done = False
        return done

    def get_reward_range(self):
        """This function is not used, Habitat-RLEnv requires this function"""
        return (0., 1.0)

    def get_reward(self, observations):
        return 0

    def get_pose_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(
            curr_sim_pose, self.last_sim_location)
        return dx, dy, do

    def get_sim_location(self):
        """Returns x, y, o pose of the agent in the Habitat simulator."""

        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis %
                                          (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_regions_in_one_level(self):
        ### get regions in the same level ###
        gt_annot = self.gt_annotation
        scene_name = self.scene_name.split('/')[-2]
        agent_sim_loc = self.habitat_env.sim.get_agent_state(0).position
        agent_sim_rot = self.habitat_env.sim.get_agent_state(0).rotation
        regions = gt_annot[scene_name]['regions']
        bbox_list = []
        polygon_list = []
        for region in regions:
            center = region['center']
            sizes = region['dim']
            x_min = center[0]-sizes[0]/2
            x_max = center[0]+sizes[0]/2
            y_min = center[1]-sizes[1]/2
            y_max = center[1]+sizes[1]/2
            z_min = center[2]-sizes[2]/2
            z_max = center[2]+sizes[2]/2
            if agent_sim_loc[0] > x_min and agent_sim_loc[0] < x_max and \
                    agent_sim_loc[1] > y_min and agent_sim_loc[1] < y_max and \
                    agent_sim_loc[2] > z_min and agent_sim_loc[2] < z_max:
                vertices = region['vertices']
                points = [(x,z) for (x,y,z) in vertices]
                path = mpltPath.Path(points)
                if path.contains_points(np.array((agent_sim_loc[0], agent_sim_loc[2]))[np.newaxis, ...]):
                    polygon_list.append(region['id'])
                else:
                    bbox_list.append(region['id'])
        if len(polygon_list) > 0:
            for region_id in polygon_list:
                cur_level = regions[region_id]['level_id']
                if regions[region_id]['label'] != 'h' and \
                        regions[region_id]['label'] != 's': # not hallway or stair
                    break
        elif len(bbox_list) > 0:
            for region_id in bbox_list:
                cur_level = regions[region_id]['level_id']
                if regions[region_id]['label'] != 'h' and \
                        regions[region_id]['label'] != 's': # not hallway or stair
                    break
        else:
            temp_min_d = 100
            best_region = None
            for region in regions:
                region_h = region['vertices'][0][1]
                temp_ = np.abs(agent_sim_loc[1] - region_h)
                if temp_ < temp_min_d:
                    temp_min_d = temp_
                    best_region = region
            cur_level = best_region['level_id']

        same_level_regions = gt_annot[scene_name]['levels'][cur_level]['regions']

        ### get region coordinate in semantic map ###
        initial_pos = np.array((-agent_sim_loc[2], -agent_sim_loc[0]))
        axis = quaternion.as_euler_angles(agent_sim_rot)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis %
                                          (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_sim_rot)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_sim_rot)[1]
        if o > np.pi:
            o -= 2 * np.pi

        all_regions = []
        rot_matrix = np.array([[np.cos(o), np.sin(o)],[-np.sin(o), np.cos(o)]])
        agent_pose_m_np = np.zeros(3)
        agent_pose_m_np[0] = self.args.map_size_cm / 100. / 2
        agent_pose_m_np[1] = self.args.map_size_cm / 100. / 2
        for region in same_level_regions:
            vertices = region['vertices']
            region_type_id = mp3d_region_label2id[region['label']]
            v_pos = -np.array(vertices)
            v_pos = v_pos[:,[2,0]]
            real_pos = agent_pose_m_np[:2][np.newaxis, ...] + \
                       np.dot(v_pos - initial_pos[np.newaxis, ...], rot_matrix.transpose())
            all_regions.append((real_pos, region_type_id))
        return all_regions

    def point2region_semantic(self, point):
        initial_info = self.region_semantic_info['initial_info']
        gt_region_mask = self.region_semantic_info['gt_region_mask']
        H,W = self.args.frame_height, self.args.frame_width

        # tranform by initial location
        agent_sim_loc = initial_info['agent_sim_loc']
        agent_sim_rot = initial_info['agent_sim_rot']
        initial_pos = initial_info['initial_pos']
        initial_axis = initial_info['initial_axis']
        initial_agent_pose_m_np = initial_info['agent_pose_m_np']
        if (initial_axis % (2 * np.pi)) < 0.1 or (initial_axis %
                                                  (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_sim_rot)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_sim_rot)[1]
        if o > np.pi:
            o -= 2 * np.pi

        xy = -point[[2,0],:]
        xy = xy.transpose()
        rot_matrix = np.array([[np.cos(o), np.sin(o)],[-np.sin(o), np.cos(o)]])
        depth_pos_m = initial_agent_pose_m_np[:2][np.newaxis, ...] + \
                      np.dot(xy - initial_pos[np.newaxis, ...], rot_matrix.transpose())
        depth_map_pos = depth_pos_m * 100. / self.args.map_resolution

        #TODO check if pos out of the range
        depth_region_id = gt_region_mask[depth_map_pos[:,1].astype(np.int), depth_map_pos[:,0].astype(np.int)]
        region_semantic = depth_region_id.astype(np.int).reshape(H,W)
        one_hot_region_semantic = np.zeros((H,W, len(mp3d_region_id2name.keys())))
        for region_id in range(len(mp3d_region_id2name.keys())):
            one_hot_region_semantic[..., region_id] = region_semantic == region_id
        return one_hot_region_semantic

    def depth2point(self, depth):
        hfov = self.args.hfov / 180. * np.pi
        H,W = self.args.frame_height, self.args.frame_width
        K = np.array(
            [[1 / np.tan(hfov / 2.), 0., 0., 0.],
             [0., W/H / np.tan(hfov / 2.), 0., 0.],
             [0., 0.,  1, 0],
             [0., 0., 0, 1]]
        )

        xs, ys = np.meshgrid(np.linspace(-1,1,W), np.linspace(1,-1,H))
        depth = depth[..., 0].reshape(1,H,W) * 0.9
        xs = xs.reshape(1,H,W)
        ys = ys.reshape(1,H,W)

        # Unproject
        # negate depth as the camera looks along -Z
        xys = np.vstack((xs * depth, ys * depth, -depth, np.ones(depth.shape)))
        xys = xys.reshape(4, -1)
        xy_c0 = np.matmul(np.linalg.inv(K), xys)

        quaternion_0 = self.habitat_env._sim.get_agent_state().sensor_states['depth'].rotation
        translation_0 = self.habitat_env._sim.get_agent_state().sensor_states['depth'].position
        rotation_0 = quaternion.as_rotation_matrix(quaternion_0)
        T_world_camera0 = np.eye(4)
        T_world_camera0[0:3,0:3] = rotation_0
        T_world_camera0[0:3,3] = translation_0
        xy_c1 = np.matmul(T_world_camera0, xy_c0)
        xyz = xy_c1[:3]
        xyz[1] = -xyz[1]

        return xyz

    def _plan(self, obstacle_map, goal_map, pose_m, real_target):
        obstacle = np.rint(obstacle_map)
        start_x, start_y, start_o = pose_m
        start_x = int(start_x * 100 / self.args.map_resolution)
        start_y = int(start_y * 100 / self.args.map_resolution)
        start_y, start_x = pu.threshold_poses([start_x, start_y], obstacle_map.shape) # switch x<->y

        self.visited[start_x, start_y] = 1

        # collision check
        if self.last_action == 1:
            last_sim_loc = self.prev_pos_m
            curr_sim_loc = pose_m

            x1, y1, t1 = last_sim_loc
            x2, y2, _ = curr_sim_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < self.args.collision_threshold:  # Collision
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                             ((i + buf) * np.cos(np.deg2rad(t1))
                              + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                             ((i + buf) * np.sin(np.deg2rad(t1))
                              - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / self.args.map_resolution), \
                               int(c * 100 / self.args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        #plt.imshow(obstacle)
        #plt.show()

        traversible = skimage.morphology.binary_dilation(obstacle, self.selem) != True
        traversible[self.collision_map==1] = 0
        traversible[self.visited == 1] = 1 # mark visited loc as traversible
        traversible[int(start_x)-1:int(start_x)+2, int(start_y)-1:int(start_y)+2] = 1

        traversible = add_boundary(traversible)
        goal_map = add_boundary(goal_map, value=0)

        planner = FMMPlanner(traversible)

        selem = skimage.morphology.disk(10)
        goal_map = skimage.morphology.binary_dilation(
            goal_map, selem) != True
        goal_map = 1 - goal_map * 1.
        planner.set_multi_goal(goal_map)

        stg_x, stg_y, _, stop = planner.get_short_term_goal([start_x+1, start_y+1])
        stg_x -= 1
        stg_y -= 1

        if stop and real_target:
            action = 0
        else:
            angle_st_goal = math.degrees(math.atan2(stg_x - start_x,
                                                    stg_y - start_y))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            if relative_angle > self.args.turn_angle / 2.:
                action = 3  # Right
            elif relative_angle < -self.args.turn_angle / 2.:
                action = 2  # Left
            else:
                action = 1  # Forward

        self.last_action = action
        self.prev_pos_m[0] = pose_m[0]
        self.prev_pos_m[1] = pose_m[1]
        self.prev_pos_m[2] = pose_m[2]
        return action
