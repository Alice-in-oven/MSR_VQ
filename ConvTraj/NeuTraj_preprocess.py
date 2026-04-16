import random
import os
import pickle
import numpy as np

beijing_lat_range = [39.6, 40.7]
beijing_lon_range = [115.9, 117.1]  # 修正：原来写成 [115.9,117,1] 看起来是笔误


class Preprocesser(object):
    def __init__(self, delta=0.005, lat_range=(1, 2), lon_range=(1, 2)):
        self.delta = float(delta)
        self.lat_range = list(lat_range)
        self.lon_range = list(lon_range)
        self._init_grid_hash_function()

    def _init_grid_hash_function(self):
        dXMax, dXMin = self.lon_range[1], self.lon_range[0]
        dYMax, dYMin = self.lat_range[1], self.lat_range[0]
        self.x = self._frange(dXMin, dXMax, self.delta)
        self.y = self._frange(dYMin, dYMax, self.delta)

    def _frange(self, start, end=None, inc=None):
        """A range function, that does accept float increments..."""
        if end is None:
            end = float(start)
            start = 0.0
        if inc is None:
            inc = 1.0

        L = []
        while True:
            nxt = start + len(L) * inc
            if inc > 0 and nxt >= end:
                break
            elif inc < 0 and nxt <= end:
                break
            L.append(nxt)
        return L

    def get_grid_index(self, pt):
        test_x, test_y = pt[0], pt[1]
        x_grid = int((test_x - self.lon_range[0]) / self.delta)
        y_grid = int((test_y - self.lat_range[0]) / self.delta)
        index = (y_grid * len(self.x)) + x_grid
        return x_grid, y_grid, index

    def traj2grid_seq(self, trajs=None, isCoordinate=False):
        if trajs is None:
            trajs = []

        grid_traj = []
        for r in trajs:
            x_grid, y_grid, index = self.get_grid_index((r[2], r[1]))
            grid_traj.append(index)

        previous = None
        hash_traj = []
        for idx, g in enumerate(grid_traj):
            if previous is None:
                previous = g
                if not isCoordinate:
                    hash_traj.append(g)
                else:
                    hash_traj.append(trajs[idx][1:])
            else:
                if g == previous:
                    pass
                else:
                    if not isCoordinate:
                        hash_traj.append(g)
                    else:
                        hash_traj.append(trajs[idx][1:])
                    previous = g
        return hash_traj

    def _traj2grid_preprocess(self, traj_feature_map, isCoordinate=False):
        trajs_hash = []
        trajs_keys = traj_feature_map.keys()  # dict_keys view in py3, 仍可迭代
        for traj_key in trajs_keys:
            traj = traj_feature_map[traj_key]
            trajs_hash.append(self.traj2grid_seq(traj, isCoordinate))
        return trajs_hash

    def preprocess(self, traj_feature_map, isCoordinate=False):
        if not isCoordinate:
            traj_grids = self._traj2grid_preprocess(traj_feature_map)
            print("gird trajectory nums {}".format(len(traj_grids)))

            useful_grids = {}
            count = 0
            max_len = 0
            for traj in traj_grids:
                if len(traj) > max_len:
                    max_len = len(traj)
                count += len(traj)
                for grid in traj:
                    if grid in useful_grids:
                        useful_grids[grid][1] += 1
                    else:
                        useful_grids[grid] = [len(useful_grids) + 1, 1]

            print(len(useful_grids.keys()))
            print(count, max_len)
            return traj_grids, useful_grids, max_len

        else:
            traj_grids = self._traj2grid_preprocess(traj_feature_map, isCoordinate=isCoordinate)
            max_len = 0
            useful_grids = {}
            for traj in traj_grids:
                if len(traj) > max_len:
                    max_len = len(traj)
            return traj_grids, useful_grids, max_len


def trajectory_feature_generation(
    path="./data/toy_trajs",
    lat_range=beijing_lat_range,
    lon_range=beijing_lon_range,
    min_length=50,
    delta=0.001,
):
    fname = path.split("/")[-1].split("_")[0]

    # ====== 新增：检查是否已处理 ======
    feat_dir = "./features"
    traj_index_path = os.path.join(feat_dir, f"{fname}_traj_index")
    traj_coord_path = os.path.join(feat_dir, f"{fname}_traj_coord")
    traj_grid_path  = os.path.join(feat_dir, f"{fname}_traj_grid")

    if (
        os.path.exists(traj_index_path)
        and os.path.exists(traj_coord_path)
        and os.path.exists(traj_grid_path)
    ):
        print(f"[SKIP] {fname} already processed.")
        return traj_coord_path, fname

    # Python3：pickle 需要二进制模式
    with open(path, "rb") as f:
        trajs = pickle.load(f)

    traj_index = {}
    max_len = 0
    preprocessor = Preprocesser(delta=delta, lat_range=lat_range, lon_range=lon_range)

    print(preprocessor.get_grid_index((lon_range[1], lat_range[1])))

    for i, traj in enumerate(trajs):
        new_traj = []
        coor_traj = []

        if len(traj) > min_length:
            inrange = True
            for p in traj:
                lon, lat = p[0], p[1]
                if not (
                    (lat > lat_range[0])
                    and (lat < lat_range[1])
                    and (lon > lon_range[0])
                    and (lon < lon_range[1])
                ):
                    inrange = False

                # 保持你原来的结构：[0, lat, lon]
                new_traj.append([0, p[1], p[0]])

            if inrange:
                coor_traj = preprocessor.traj2grid_seq(new_traj, isCoordinate=True)
                if len(coor_traj) == 0:
                    print(len(coor_traj))

                if (len(coor_traj) > 10) and (len(coor_traj) < 150):
                    if len(traj) > max_len:
                        max_len = len(traj)
                    traj_index[i] = new_traj

        if i % 200 == 0:
            print(coor_traj)
            print(i, len(traj_index.keys()))

    print(max_len)
    print(len(traj_index.keys()))

    # Python3：pickle dump 用 "wb"
    with open("./features/{}_traj_index".format(fname), "wb") as f:
        pickle.dump(traj_index, f, protocol=pickle.HIGHEST_PROTOCOL)

    trajs, useful_grids, max_len = preprocessor.preprocess(traj_index, isCoordinate=True)

    print(trajs[0])

    with open("./features/{}_traj_coord".format(fname), "wb") as f:
        pickle.dump((trajs, [], max_len), f, protocol=pickle.HIGHEST_PROTOCOL)

    all_trajs_grids_xy = []
    min_x, min_y, max_x, max_y = 2000, 2000, 0, 0

    for traj in trajs:
        for j in traj:
            x, y, index = preprocessor.get_grid_index((j[1], j[0]))
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y

    print(min_x, min_y, max_x, max_y)

    for traj in trajs:
        traj_grid_xy = []
        for j in traj:
            x, y, index = preprocessor.get_grid_index((j[1], j[0]))
            x = x - min_x
            y = y - min_y
            grids_xy = [y, x]
            traj_grid_xy.append(grids_xy)
        all_trajs_grids_xy.append(traj_grid_xy)

    print(all_trajs_grids_xy[0])
    print(len(all_trajs_grids_xy))
    print(all_trajs_grids_xy[0])

    with open("./features/{}_traj_grid".format(fname), "wb") as f:
        pickle.dump((all_trajs_grids_xy, [], max_len), f, protocol=pickle.HIGHEST_PROTOCOL)

    return "./features/{}_traj_coord".format(fname), fname
