from gym import spaces

from driving_sim.utils.trajectory import *
from driving_sim.envs.t_intersection_pred_front import TIntersectionPredictFront


# same as TIntersectionPredictFront except that the ob includes each driver's actions in pretext_nodes key
# Usage: used for Morton et. al baseline

class TIntersectionPredictFrontAct(TIntersectionPredictFront):
    def __init__(self):
        super(TIntersectionPredictFrontAct, self).__init__()

    @property
    def observation_space(self):
        d = {}
        # robot node: px, py
        d['robot_node'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)

        # only consider all temporal edges (human_num+1) and spatial edges pointing to robot (human_num)
        d['temporal_edges'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)

        # edge feature will be (px - px_robot, py - py_robot, intent)
        d['spatial_edges'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num, self.spatial_ob_size + self.latent_size),
                                        dtype=np.float32)

        # observation for pretext latent state inference with S-RNN
        # nodes: the (px, ax) for all cars except ego car
        d['pretext_nodes'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num, 1,), dtype=np.float32)
        # spatial edges: the delta_px, delta_vx from each car i to its front car
        d['pretext_spatial_edges'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num, 2,),
                                                dtype=np.float32)
        # temporal edges: vx of all cars except ego car
        d['pretext_temporal_edges'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num, 1,),
                                                 dtype=np.float32)
        # mask to indicate whether each id has a car present or a dummy car
        d['pretext_masks'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num,),
                                        dtype=np.float32)
        # mask to indicate whether each car needs to be inferred (based on its lane and the y position of ego car)
        d['pretext_infer_masks'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num,),
                                              dtype=np.int32)
        # the true label of each car (for debugging purpose)
        d['true_labels'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num,),
                                      dtype=np.float32)
        d['pretext_actions'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num, 1,), dtype=np.float32)
        return spaces.Dict(d)

    def observe(self, normalize=True):
        obs = super().observe(normalize)
        # add normalized actions of other cars
        obs['pretext_actions'] = self.action_with_idx.reshape((self.max_veh_num, 1)) / 9.
        return obs


