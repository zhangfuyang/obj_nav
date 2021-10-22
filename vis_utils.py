import cv2
import torch
import numpy as np
from constants import obj_color_palette, region_color_palette, category2objectid, mp3d_region_id2name


def visualization(img, semantic_map=None, agent_pose_m=None, arg=None, title="vis", goal_name='', width=1500):
    '''
    Args:
        img: rgb
        semantic_map: [3, map_size, map_size].
                        dim 0: obstacle map
                        dim 1: visited map
                        ...
        agent_pose_m: [3]
                        dim 0-1: x, y loc in meter.
                                 agent_pose_m[:2] * 100 / arg.map_resolution is corresponding loc in semantic_map
                        dim   2: orientation in degree
        arg:
        h_size: vis height [default: 512]
    Returns:
    '''
    if type(img) == torch.Tensor:
        img = img.detach().cpu().numpy()
    if type(semantic_map) == torch.Tensor:
        semantic_map = semantic_map.detach().cpu().numpy()
    if type(agent_pose_m) == torch.Tensor:
        agent_pose_m = agent_pose_m.detach().cpu().numpy()

    # draw explored map
    vis_semantic_map = (semantic_map[1] >= 0.1)[..., np.newaxis] * \
                       np.array((242., 242., 242.))[np.newaxis, np.newaxis, ...] # render obstacle

    # draw obstacle map
    place = np.where(semantic_map[0] >= 0.1)
    vis_semantic_map[place[0], place[1]] = 152. # render obstacle


    ####  start: draw agent arrow ####
    agent_loc = agent_pose_m[:2] * 100. / arg.map_resolution
    def get_contour_arrow(loc, orentation, size=10):
        x, y = loc
        o = orentation / 180 * np.pi
        pt1 = (int(x), int(y))
        pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)),
               int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)))
        pt3 = (int(x + size * np.cos(o)), int(y + size * np.sin(o)))
        pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)),
               int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)))

        return np.array([pt1, pt2, pt3, pt4])

    _arrow = get_contour_arrow(agent_loc, agent_pose_m[2])
    vis_semantic_map = cv2.drawContours(vis_semantic_map, [_arrow], 0, (255, 0, 0), -1)
    ####   end:  draw agent arrow ####

    place = np.where(vis_semantic_map.max(2)==0)
    vis_semantic_map[place[0], place[1], :] = 255
    vis_semantic_map[:,:5] = 0
    vis_semantic_map[:,-5:] = 0
    vis_semantic_map[:5, :] = 0
    vis_semantic_map[-5:, :] = 0

    #### start: draw obj/region semantic map ####
    combine_feat = []
    if arg.use_obj:
        obj_semantic = vis_semantic_map.copy()
        sem_mask = semantic_map[4:4+len(category2objectid.keys())]
        color_palette = obj_color_palette
        for mask_i in range(sem_mask.shape[0]):
            color = color_palette[mask_i]
            mask = sem_mask[mask_i]
            place = np.where(mask != 0)
            obj_semantic[place[0], place[1], :] = color
        combine_feat.append(obj_semantic)

    if arg.use_region:
        region_semantic = vis_semantic_map.copy()
        sem_mask = semantic_map[-len(mp3d_region_id2name.keys()):]
        color_palette = region_color_palette
        for mask_i in range(sem_mask.shape[0]):
            color = color_palette[mask_i]
            mask = sem_mask[mask_i]
            place = np.where(mask != 0)
            region_semantic[place[0], place[1], :] = color
        combine_feat.append(region_semantic)

    if len(combine_feat) == 0:
        combine_feat.append(vis_semantic_map)

    vis_semantic_map = np.concatenate(combine_feat, 1)
    vis_semantic_map = np.flipud(vis_semantic_map)

    ####   end: draw semantic map ####


    #### start: combine with img ####
    img_h, img_w = img.shape[:2]
    semantic_map_h, semantic_map_w = vis_semantic_map.shape[:2]
    scale = semantic_map_h / img_h
    img_resized = cv2.resize(img, dsize=(round(img_w*scale), semantic_map_h))
    img_resized[:,:5] = 0
    img_resized[:,-5:] = 0
    img_resized[:5, :] = 0
    img_resized[-5:, :] = 0
    combined_vis = np.concatenate((img_resized, vis_semantic_map), 1)
    #### end: combine with img ####

    # add legend
    cv2.putText(combined_vis, goal_name, ((combined_vis.shape[1] * 2)//3, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0.,0.,0.),
                thickness=2)

    if arg.use_obj:
        combined_vis = np.concatenate((combined_vis,
                                       np.ones((50, combined_vis.shape[1], 3)).astype(np.uint8)*255),
                                      0)
        x_loc = 20
        for idx, obj_name in enumerate(category2objectid.keys()):
            cv2.putText(combined_vis, obj_name,
                        (x_loc, combined_vis.shape[0]-10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0.,0.,0.),
                        thickness=1)
            cv2.rectangle(combined_vis, (x_loc, combined_vis.shape[0]-40),
                          (x_loc+4*len(obj_name), combined_vis.shape[0]-30),
                          obj_color_palette[idx].astype(np.int).tolist(),
                          thickness=-1)
            x_loc += 20 + 10 * len(obj_name)

    if arg.use_region:
         combined_vis = np.concatenate((combined_vis,
                                        np.ones((50, combined_vis.shape[1], 3)).astype(np.uint8)*255),
                                       0)
         x_loc = 20
         for idx in mp3d_region_id2name.keys():
             region_name = mp3d_region_id2name[idx]
             cv2.putText(combined_vis, region_name,
                         (x_loc, combined_vis.shape[0]-10),
                         fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0.,0.,0.),
                         thickness=1)
             cv2.rectangle(combined_vis, (x_loc, combined_vis.shape[0]-40),
                           (x_loc+4*len(region_name), combined_vis.shape[0]-30),
                           region_color_palette[idx].astype(np.int).tolist(),
                           thickness=-1)
             x_loc += 20 + 15 * len(region_name)

    h, w = combined_vis.shape[:2]
    scale = width / w
    combined_vis = cv2.resize(combined_vis, dsize=(width, round(h*scale)))

    combined_vis = combined_vis[:,:,::-1]
    cv2.imshow(title, combined_vis / 255)
    cv2.waitKey(1)
    return
