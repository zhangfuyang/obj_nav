import numpy as np
from constants import mp3d_region_label2id

def parse_house(file_name):
    f = open(file_name, 'r')
    data = f.readlines()
    f.close()
    l_id = 0
    r_id = 0
    object2region = np.load(file_name.split('.')[0]+'_object2region.npy', allow_pickle=True).tolist()
    temp = np.zeros(len(object2region.keys()))
    for key in object2region.keys():
        temp[key] = object2region[key]
    object2region = temp
    house = {'name': file_name, 'levels': [], 'regions': [], 'obj2region': object2region}
    for line in data:
        if line[0] == 'L':
            # level
            prim = line.split(' ')
            prim = [x for x in prim if x]
            xl = float(prim[7])
            yl = float(prim[8])
            zl = float(prim[9])
            xh = float(prim[10])
            yh = float(prim[11])
            zh = float(prim[12])

            xc = (xl + xh) / 2
            yc = (zl + zh) / 2
            zc = -(yl + yh) / 2
            xdim = xh - xl
            ydim = zh - zl
            zdim = yh - yl
            assert xdim >= 0 and ydim >= 0 and zdim >= 0
            level = {'id': l_id, 'regions': [], 'center': (xc, yc, zc),
                     'dim': (xdim, ydim, zdim)}
            house['levels'].append(level)
            l_id += 1
        elif line[0] == 'R':
            # region
            prim = line.split(' ')
            prim = [x for x in prim if x]
            level_id = int(prim[2])
            label = prim[5]
            xl = float(prim[9])
            yl = float(prim[10])
            zl = float(prim[11])
            xh = float(prim[12])
            yh = float(prim[13])
            zh = float(prim[14])

            xc = (xl + xh) / 2
            yc = (zl + zh) / 2
            zc = -(yl + yh) / 2
            xdim = xh - xl
            ydim = zh - zl
            zdim = yh - yl
            assert xdim >= 0 and ydim >= 0 and zdim >= 0
            region = {'id': r_id, 'vertices': [], 'level_id': level_id,
                      'label': label, 'center': (xc, yc, zc), 'dim': (xdim, ydim, zdim)}
            house['regions'].append(region)
            house['levels'][level_id]['regions'].append(region)
            r_id += 1
        elif line[0] == 'V':
            # boundary vertices
            prim = line.split(' ')
            prim = [x for x in prim if x]
            x = float(prim[4])
            y = float(prim[6])
            z = -float(prim[5])
            region_id = int(prim[2])
            house['regions'][region_id]['vertices'].append((x,y,z))

    region2typeid = np.zeros(len(house['regions']), dtype=np.int)
    for idx, region in enumerate(house['regions']):
        label = region['label']
        region_type_id = mp3d_region_label2id[label]
        region2typeid[idx] = region_type_id
    house['region2typeid'] = region2typeid
    return house
