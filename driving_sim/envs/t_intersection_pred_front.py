from gym import spaces

from driving_sim.utils.trajectory import *
from driving_sim.envs.t_intersection import TIntersection


# Observation space is a dict that contains
# 1. the node, temporal edge, and 12 spatial edges defined in DSRNN for the ego car, where the spatial edges contain all other cars
# 2. the node, temporal edge, and 1 spatial edge for other cars, where the spatial edges contain each car's front car

# Fix compared with TIntersectionPredict:
# 1. all info in y axis are removed for other cars,
# 2. added presence masks and true labels for other cars
# 3. removed ego car from pretext_spatial_edges

# Usage: works for both unsupervised learning for trait inference and RL for controlling the ego car
# only used for our method (VAE+RNN network)

class TIntersectionPredictFront(TIntersection):
    def __init__(self):
        super(TIntersectionPredictFront, self).__init__()


    @property
    def observation_space(self):
        d = {}
        # robot node: px, py
        d['robot_node'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)

        # only consider all temporal edges (human_num+1) and spatial edges pointing to robot (human_num)
        d['temporal_edges'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)

        # edge feature will be (px - px_robot, py - py_robot, intent)
        d['spatial_edges'] = spaces.Box(low=-np.inf, high=np.inf,
                                        shape=(self.max_veh_num, self.spatial_ob_size + self.latent_size),
                                        dtype=np.float32)

        # observation for pretext latent state inference with S-RNN
        # nodes: the px for all cars except ego car
        d['pretext_nodes'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num, 1,), dtype=np.float32)
        # spatial edges: the delta_px, delta_vx from each car i to its front car
        d['pretext_spatial_edges'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num, 2,),
                                                dtype=np.float32)
        # temporal edges: vx of all cars except ego car
        d['pretext_temporal_edges'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num, 1,),
                                                 dtype=np.float32)
        # mask to indicate whether each id has a car present or a dummy car
        d['pretext_masks'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num, ),
                                                 dtype=np.float32)
        # mask to indicate whether each car needs to be inferred (based on its lane and the y position of ego car)
        d['pretext_infer_masks'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num, ),
                                                 dtype=np.int32)
        # the true label of each car (for debugging purpose)
        d['true_labels'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num, ),
                                                 dtype=np.float32)
        return spaces.Dict(d)

    def fill_infer_masks(self):
        # mask to indicate whether car's intent needs to be inferred
        # if the ego car has not passed the lower lane yet, all cars in both lanes need to be inferred
        if self._cars[0].position[1] < -1.77: # old: -1.75, new: -1.77
            return self.car_lane_info != 0
        # if the ego car has passed the lower lane but not the upper lane, all cars in upper lane need to be inferred
        elif -1.77 <= self._cars[0].position[1] < 2.:
            return self.car_lane_info == 1
        # if the ego car has passed the upper lane, it doesn't need to infer anything
        else:
            return np.zeros(self.max_veh_num, dtype=bool)

    def observe(self, normalize=True):
        # if normalize: divide all positions by self.right_bound, divide all velocities by self.desire_speed
        obs = {}

        obs['robot_node'] = self._cars[0].position / self.right_bound if normalize else self._cars[0].position

        spatial_edge_size = self.spatial_ob_size + self.latent_size
        obs['spatial_edges'] = np.zeros((self.max_veh_num, spatial_edge_size))

        obs['pretext_nodes'] = np.zeros((self.max_veh_num, 1))
        obs['pretext_temporal_edges'] = np.zeros((self.max_veh_num, 1))
        obs['pretext_spatial_edges'] = np.zeros((self.max_veh_num, 2))
        obs['true_labels'] = np.zeros((self.max_veh_num, ))

        for i, car in enumerate(self._cars):
            # temporal edges
            if i == 0:
                obs['temporal_edges'] = np.array(self._cars[i].velocity / self.desire_speed) if normalize else np.array(
                    self._cars[i].velocity)
            # for every other car
            else:
                # spatial edges: vector pointing from ego car to this car
                relative_pos = np.array(self._cars[i].position) - np.array(self._cars[0].position)
                if normalize:
                    relative_pos = relative_pos / self.right_bound

                # insert it to the correct position based on the idx of this car
                obs['spatial_edges'][int(car._idx - 1), :] = np.append(relative_pos, [0]*self.latent_size)

                obs['true_labels'][int(car._idx - 1)] = self._drivers[i].yld

                # pretext_nodes: use displacement from starting position instead of absolute position
                px = self._cars[i].position[0]/self.right_bound if normalize else self._cars[i].position[0]
                obs['pretext_nodes'][int(car._idx-1), 0] = px * self._drivers[i].x_driver.direction

                # pretext temporal edges
                obs['pretext_temporal_edges'][int(car._idx-1), 0] = self._cars[i].velocity[0]/self.desire_speed \
                    if normalize else self._cars[i].velocity[0]
                obs['pretext_temporal_edges'][int(car._idx - 1), 0] = obs['pretext_temporal_edges'][int(car._idx-1), 0]\
                                                                      * self._drivers[i].x_driver.direction

                # pretext spatial edges: vectors pointing from this car to its front car
                front_pos, front_vel = self._drivers[i].x_driver.get_front_relative_x_pos_vel(self._cars, self._drivers[i].yld)
                if normalize: # normalize to [-1, 1]
                    front_pos = front_pos / self.right_bound
                    front_vel = front_vel / self.desire_speed
                obs['pretext_spatial_edges'][int(car._idx-1)] = np.array([front_pos, front_vel])* self._drivers[i].x_driver.direction

        # fill in masks
        obs['pretext_masks'] = self.car_present
        obs['pretext_infer_masks'] = self.fill_infer_masks()

        return obs

