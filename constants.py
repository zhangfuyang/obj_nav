import numpy as np
category2objectid = {
    'table': 0,
    'chair': 1,
    'picture': 2,
    'cabinet': 3,
    'chest_of_drawers': 4,
    'stool': 5,
    'counter': 6,
    'sink': 7,
    'tv_monitor': 8,
    'clothes': 9,
    'sofa': 10,
    'plant': 11,
    'seating': 12,
    'bed': 13,
    'cushion': 14,
    'shower': 15,
    'towel': 16,
    'toilet': 17,
    'bathtub': 18,
    'gym_equipment': 19,
    'fireplace': 20
}
objid_mp3did = [ # (obj_id, mp3d_id)
    (0, 5),
    (1, 3),
    (2, 6),
    (3, 7),
    (4, 13),
    (5, 19),
    (6, 26),
    (7, 15),
    (8, 22),
    (9, 38),
    (10, 10),
    (11, 14),
    (12, 34),
    (13, 11),
    (14, 8),
    (15, 23),
    (16, 20),
    (17, 18),
    (18, 25),
    (19, 33),
    (20, 27)
]
obj_color_palette = np.array([
        (51,13,13), (178,143,0), (0,179,167), (128,179,255), (179,89,161),
        (178,24,0), (51,48,13), (0,83,89), (13,23,51), (230,0,153),
        (217,170,163), (230,226,172), (191,251,255), (0,22,166), (102,51,71),
        (102,46,26), (173,204,51), (61,206,242), (57,57,77), (255,0,68),
        (255,162,128)])

#(182,214,242), (230,182,242), (102,87,77), (51,102,71), (0,102,255), (214,0,230), (166,138,83), (0,255,204), (45,98,179), (51,0,41)


mp3d_region_label2id = {
    'a': 2,#bathroom
    'b': 3,#bedroom
    'c': 4,#closet
    'd': 5,#dining room
    'e': 1,#entryway/foyer/lobby
    'f': 6,#familyroom
    'g': 1,#garage
    'h': 1,#hallway
    'i': 1,#library
    'j': 1,#laundryroom
    'k': 7,#kitchen
    'l': 6,#living room
    'm': 6,#meetingroom
    'n': 6,#lounge
    'o': 8,#office
    'p': 1,#porch/terrace/deck/driveway
    'r': 9,#rec/game
    's': 10,#stairs
    't': 2,#toilet
    'u': 1,#utilityroom/toolroom
    'v': 6,#tv
    'w': 9,#workout/gym/exercise
    'x': 1,#outdoor areas
    'y': 1,#balcony
    'z': 1,#other room
    'B': 6,#bar
    'C': 1,#classroom
    'D': 5,#dining booth
    'S': 2,#spa/sauna
    'Z': 0,#junk
    '-': 1,#no label
}
mp3d_region_id2name = {
    0: 'junk/undefined',
    1: 'other',
    2: 'bathroom',
    3: 'bedroom',
    4: 'closet',
    5: 'dining room',
    6: 'living room',
    7: 'kitchen',
    8: 'office',
    9: 'rec/game',
    10: 'stairs',
}
region_color_palette = np.array([(87,102,26), (13,43,51), (34,0,255), (242,121,153),
                                 (255,102,0), (89,179,89), (45,134,179), (66,26,102), (140,35,49),
                                 (178,98,45), (0,242,97)])
