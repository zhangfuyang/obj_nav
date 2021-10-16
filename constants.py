coco_categories = {
    "chair": 0,
    "sofa": 1,
    "plant": 2,
    "bed": 3,
    "toilet": 4,
    "tv_monitor": 5,
    "dining-table": 6,
    "oven": 7,
    "sink": 8,
    "refrigerator": 9,
    "book": 10,
    "clock": 11,
    "vase": 12,
    "cup": 13,
    "bottle": 14
}

coco_categories_mapping = {
   56: 0,# chair
   57: 1,# sofa
   58: 2,# plant
   59: 3,# bed
   61: 4,# toilet
   62: 5,# tv
   60: 6,  # dining-table
   69: 7,  # oven
   71: 8,  # sink
   72: 9,  # refrigerator
   73: 10,  # book
   74: 11,  # clock
   75: 12,  # vase
   41: 13,  # cup
   39: 14,  # bottle
}

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

#all mp3d
#coco_categories = {
#    "bed": 0, # 97150
#    "tv_monitor": 1, # 20677
#    "chest_of_drawers": 2, # 104915
#    "cushion": 3, # 363213
#    "table": 4, # 337131
#    "towel": 5, # 83044
#    "shower": 6, # 57437
#    "sink": 7, # 115331
#    "clothes": 8, # 17170
#    "stool": 9, # 82165
#    "toilet": 10, # 66693
#    "cabinet": 11, # 213887
#    "counter": 12, # 69853
#    "sofa": 13, # 78947
#    "chair": 14, # 545293
#    "picture": 15, # 135163
#    "seating": 16, # 67205
#    "bathtub": 17, # 29019
#    "plant": 18, # 116657
#    "gym_equipment": 19, # 7610
#    "fireplace": 20, # 23862
#}

#all mp3d
#coco_categories_mapping = {
#    59: 0,   #bed
#    62: 1,   #tv_monitor
#    -1: 2,   #chest_of_drawer
#    -1: 3,   #cushion
#    60: 4,   #table
#    -1: 5,   #towel
#    -1: 6,   #shower
#    71: 7,   #sink
#    -1: 8,   #clothes
#    -1: 9,   #stool
#    61: 10, #toilet
#    -1: 11, #cabinet
#    -1: 12, #counter
#    57: 13, #sofa
#    56: 14, #chair
#    -1: 15, #picture
#    16: 16, #seating
#    17: 17, #bathtub
#    18: 18, #plant
#    19: 19, #gym_equipment
#    20: 20  #fireplace
#}

#original
#coco_categories_mapping = {
#    56: 0,  # chair
#    57: 1,  # couch
#    58: 2,  # potted plant
#    59: 3,  # bed
#    61: 4,  # toilet
#    62: 5,  # tv
#    60: 6,  # dining-table
#    69: 7,  # oven
#    71: 8,  # sink
#    72: 9,  # refrigerator
#    73: 10,  # book
#    74: 11,  # clock
#    75: 12,  # vase
#    41: 13,  # cup
#    39: 14,  # bottle
#}
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
color_palette = [
    1.0, 1.0, 1.0,
    0.6, 0.6, 0.6,
    0.95, 0.95, 0.95,
    0.96, 0.36, 0.26,
    0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
    0.9400000000000001, 0.7818, 0.66,
    0.9400000000000001, 0.8868, 0.66,
    0.8882000000000001, 0.9400000000000001, 0.66,
    0.7832000000000001, 0.9400000000000001, 0.66,
    0.6782000000000001, 0.9400000000000001, 0.66,
    0.66, 0.9400000000000001, 0.7468000000000001,
    0.66, 0.9400000000000001, 0.8518000000000001,
    0.66, 0.9232, 0.9400000000000001,
    0.66, 0.8182, 0.9400000000000001,
    0.66, 0.7132, 0.9400000000000001,
    0.7117999999999999, 0.66, 0.9400000000000001,
    0.8168, 0.66, 0.9400000000000001,
    0.9218, 0.66, 0.9400000000000001,
    0.9400000000000001, 0.66, 0.8531999999999998,
    0.9400000000000001, 0.66, 0.748199999999999,
]
