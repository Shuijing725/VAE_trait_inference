import random
import itertools
import numpy as np
import copy
from gym import spaces

from driving_sim.envs.traffic_env import TrafficEnv
from driving_sim.road import Road, RoadSegment
from driving_sim.car import Car
from driving_sim.drivers.driver import Driver, XYSeperateDriver, YNYDriver, EgoDriver
from driving_sim.drivers.oned_drivers import IDMDriver, PDDriver
from driving_sim.constants import *
from driving_sim.utils.trajectory import *

from driving_sim.utils.info import *

# Base class for the T-intersection env

class TIntersection(TrafficEnv):
    def __init__(self):
        super(TIntersection, self).__init__()

        self.v_noise = None
        self.vs_actions = None
        self.t_actions = None
        # we use target value instead of target change so system is Markovian
        self.rl_actions = None
        self.num_updates = None

        self.desire_speed = None
        self.collision_cost = None
        self.outroad_cost = None
        self.survive_reward = None
        self.goal_reward = None

        self.left_bound = None
        self.right_bound = None
        self.gap_min = None
        self.gap_max = None
        self.max_veh_num = None
        self.label_num = self.max_veh_num

        self._collision = False
        self._outroad = False # todo: seems that the ego car will never go out of road
        self._goal = False
        self._lower_lane_next_idx = 1
        self._upper_lane_next_idx = None

        self.car_length = 5.0
        self.car_width = 2.0
        self.car_max_accel = 10.0
        self.car_max_speed = 40.0
        self.car_max_rotation = np.pi / 18.
        self.car_expose_level = 4
        self.driver_sigma = None
        self.s_des = 3.0
        self.s_min = 3.0
        self.min_overlap = 1.0

        self.config = None

        # for approximating P(conservative) only
        self.con_count = 0
        self.agg_count = 0


    # use the given config to configurate the environment
    def configure(self, config):
        self.config = config

        self._road = config.env.road
        self.dt = config.env.dt

        self.v_noise = config.env.v_noise
        self.vs_actions = config.env.vs_actions
        self.t_actions = config.env.t_actions
        # we use target value instead of target change so system is Markovian
        self.rl_actions = list(itertools.product(config.env.vs_actions, config.env.t_actions))
        self.num_updates = config.env.num_updates

        self.desire_speed = config.env.desire_speed
        self.collision_cost = config.reward.collision_cost
        self.outroad_cost = config.reward.outroad_cost
        self.survive_reward = config.reward.survive_reward
        self.goal_reward = config.reward.goal_reward

        self.left_bound = config.car.left_bound
        self.right_bound = config.car.right_bound
        self.gap_min = config.car.gap_min
        self.gap_max = config.car.gap_max
        self.max_veh_num = config.car.max_veh_num # does NOT include ego car!
        self.label_num = self.max_veh_num

        self._lower_lane_next_idx = 1
        self._upper_lane_next_idx = int(self.max_veh_num / 2.) + 1

        self.driver_sigma = config.env.driver_sigma

        self.time_limit = config.env.time_limit

        self.action_with_idx = np.zeros(self.max_veh_num)

        # range of front gaps for 2 classes
        self.con_gap_min = config.car.con_gap_min
        self.con_gap_max = config.car.con_gap_max
        self.agg_gap_min = config.car.agg_gap_min
        self.agg_gap_max = config.car.agg_gap_max
        # the difference between the mean of front gaps between 2 classes
        # self.gap_diff = ((self.con_gap_max+self.con_gap_min) - (self.agg_gap_max+self.agg_gap_min))/2.

        self.latent_size = config.ob_space.latent_size
        # size of other cars' states
        self.spatial_ob_size = 2 # delta px, delta py

        # P(conservative)
        self.yld_prob = config.env.con_prob


    def update(self, action):

        # print(len(self._cars))

        rl_action = self.rl_actions[action]
        self._drivers[0].v_des = rl_action[0]
        self._drivers[0].t_des = rl_action[1]

        self._goal = False
        self._collision = False
        self._outroad = False
        for _ in range(self.num_updates):
            for driver in self._drivers:
                driver.observe(self._cars, self._road)
            self._actions = [driver.get_action() for driver in self._drivers]
            # create an action list that preserves the id of each driver
            self.action_with_idx = np.zeros(self.max_veh_num)
            for i, driver in enumerate(self._drivers):
                if i == 0:  # skip the ego driver
                    continue
                self.action_with_idx[int(driver._idx - 1)] = self._actions[i].a_x
            [action.update(car, self.dt) for (car, action) in zip(self._cars, self._actions)]

            ego_car = self._cars[0]
            for car in self._cars[1:]:
                if ego_car.check_collision(car):
                    self._collision = True
                    return

            if not self._road.is_in(ego_car):
                self._outroad = True
                return

            if (ego_car.position[0] > 8.) \
                and (ego_car.position[1] > 5.) \
                and (ego_car.position[1] < 7.):
                self._goal = True
                return

            # remove cars that are out-of bound
            for car, driver in zip(self._cars[1:], self._drivers[1:]):
                if (car.position[1] < 4.) and (car.position[0] < self.left_bound):
                    self.remove_car(car, driver)
                    removed_lower = True
                elif (car.position[1] > 4.) and (car.position[0] > self.right_bound):
                    self.remove_car(car, driver)
                    removed_upper = True

            # add cars when there is enough space
            # 1. find the right most car in the lower lane and the left most car in the upper lane and their idx
            # i.e. the cars that entered most recently in both lanes
            min_upper_x = np.inf
            max_lower_x = -np.inf
            for car in self._cars[1:]:
                if (car.position[1] < 4.) and (car.position[0] > max_lower_x):
                    max_lower_x = car.position[0]
                if (car.position[1] > 4.) and (car.position[0] < min_upper_x):
                    min_upper_x = car.position[0]

            # add a car to both lanes if there is space

            # lower lane
            # decide the new car's yld = True or False
            new_yld, gap_min, gap_max = self.init_trait()
            # condition for add a new car
            cond = max_lower_x < (self.right_bound - np.random.uniform(gap_min, gap_max) - self.car_length)
            # desired x location for the new car
            x = self.right_bound


            if cond:
                v_des = self.desire_speed + np.random.uniform(-1,1)*self.v_noise
                p_des = 2.
                direction = -1

                car, driver = self.add_car(x, 2., -v_des, 0., v_des, p_des, direction, np.pi, new_yld)
                if hasattr(self, 'viewer') and self.viewer:
                    car.setup_render(self.viewer)
                    driver.setup_render(self.viewer)

            # upper lane
            new_yld, gap_min, gap_max = self.init_trait()
            # condition for adding a new car
            cond = min_upper_x > (self.left_bound + np.random.uniform(gap_min, gap_max) + self.car_length)
            # desired x location for the new car
            x = self.left_bound

            if cond:
                v_des = self.desire_speed + np.random.uniform(-1,1)*self.v_noise
                p_des = 6.
                direction = 1

                car, driver = self.add_car(x, 6., v_des, 0., v_des, p_des, direction, 0., new_yld)
                if hasattr(self, 'viewer') and self.viewer:
                    car.setup_render(self.viewer)
                    driver.setup_render(self.viewer)


    # determine whether the episode ended by checking all terminal conditions
    def is_terminal(self):
        return (self._collision or self._outroad or self._goal or self.global_time >= self.time_limit)


    # get the info
    def get_info(self):
        info = {}

        if self.global_time >= self.time_limit:
            info['info'] = Timeout()
        elif self._collision:
            info['info']= Collision()
        elif self._outroad:
            info['info']=OutRoad()
        elif self._goal:
            info['info']=ReachGoal()
        else:
            info['info']=Nothing()

        return info


    # compute and return the observation
    def observe(self, normalize=True):
        # if normalize: divide all positions by self.right_bound, divide all velocities by self.desire_speed
        obs = {}

        # update the yld belief every few timesteps
        if self.step_counter % 20 == 0:
            # print('update belief')
            self.yld_belief = np.zeros(self.max_veh_num)
            for i, driver in enumerate(self._drivers):
                if i == 0:
                    continue
                else:
                    self.yld_belief[int(self._drivers[i]._idx-1)] = copy.deepcopy(self._drivers[i].yld)

        obs['robot_node'] = self._cars[0].position/self.right_bound if normalize else self._cars[0].position

        # 2 classes
        spatial_edge_size = self.spatial_ob_size + self.latent_size
        obs['spatial_edges'] = np.zeros((self.max_veh_num, spatial_edge_size))

        for i, car in enumerate(self._cars):
            # temporal edges
            if i == 0:
                obs['temporal_edges'] = np.array(self._cars[i].velocity/self.desire_speed) if normalize else np.array(self._cars[i].velocity)
            # spatial edges
            else:
                # vector pointing from human i to robot todo: should be pointing from robot to human i???
                relative_pos = np.array(self._cars[i].position) - np.array(self._cars[0].position)
                if normalize:
                    relative_pos = relative_pos / self.right_bound
                # include vx, vy
                if self.include_human_vel:
                    v = np.array(self._cars[i].velocity/self.desire_speed) if normalize else np.array(self._cars[i].velocity)
                    relative_pos = np.concatenate((relative_pos, v))
                # insert it to the correct position based on the idx of this car
                # initialize the latent state to be all zeros, will be set by the pretext models later
                obs['spatial_edges'][int(car._idx - 1), :] = np.append(relative_pos, [0]*self.latent_size)

        return obs


    # define the observation space
    @property
    def observation_space(self):
        d = {}
        # robot node: num_visible_humans, px, py
        d['robot_node'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)

        # only consider all temporal edges (human_num+1) and spatial edges pointing to robot (human_num)
        d['temporal_edges'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)
        # include ground truth information about whether other drivers will yield (0) or not (1)
        # edge feature will be (px - px_robot, py - py_robot, intent)
        d['spatial_edges'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num, self.spatial_ob_size + self.latent_size),
                                        dtype=np.float32)
        return spaces.Dict(d)


    # define the action space
    @property
    def action_space(self):
        return spaces.Discrete(len(self.rl_actions))


    # calculate the reward based on current state of ego car
    def get_reward(self):
        reward = 0.
        ego_car = self._cars[0]
        v_x, v_y = ego_car.velocity[0], ego_car.velocity[1]

        if self._collision:
            reward -= self.collision_cost
        elif self._outroad:
            reward -= self.outroad_cost
        elif self._goal:
            reward += self.goal_reward
        else:
            # add reward for larger speeds & a small constant penalty to discourage the ego car from staying in place
            reward = reward - self.survive_reward + 0.05 * np.linalg.norm([v_x, v_y]) / 3
            # reward = reward + 0.01 * np.linalg.norm([v_x, v_y]) / 3

        # print('reward:', reward)
        return reward


    # remove the given car and its corresponding controller (driver)
    def remove_car(self, car, driver):
        self._cars.remove(car)
        self._drivers.remove(driver)
        if hasattr(self, 'viewer') and self.viewer:
            car.remove_render(self.viewer)
            driver.remove_render(self.viewer)

        self.car_present[int(car._idx - 1)] = False
        self.car_lane_info[int(car._idx - 1)] = 0


    # given the pose of a car, initialize a Car & a Driver instance and append them to self._cars & self._drivers
    def add_car(self, x, y, vx, vy, v_des, p_des, direction, theta, yld):
        # P(conservative)
        if yld:
            self.con_count = self.con_count + 1
        else:
            self.agg_count = self.agg_count + 1
        if y < 4.:
            idx = self._lower_lane_next_idx
            self._lower_lane_next_idx += 1
            if self._lower_lane_next_idx > int(self.max_veh_num/2.):
                self._lower_lane_next_idx = 1
        elif y > 4.:
            idx = self._upper_lane_next_idx
            self._upper_lane_next_idx += 1
            if self._upper_lane_next_idx > self.max_veh_num:
                self._upper_lane_next_idx = int(self.max_veh_num/2.)+1
        car = Car(idx=idx, length=self.car_length, width=self.car_width, color=random.choice(RED_COLORS),
                          max_accel=self.car_max_accel, max_speed=self.car_max_speed,
                          max_rotation=0.,
                          expose_level=self.car_expose_level)
        driver = YNYDriver(idx=idx, car=car, dt=self.dt,
                    x_driver=IDMDriver(idx=idx, car=car, sigma=self.driver_sigma, s_des=self.s_des, s_min=self.s_min, axis=0, min_overlap=self.min_overlap, dt=self.dt), 
                    y_driver=PDDriver(idx=idx, car=car, sigma=0., axis=1, dt=self.dt)) 
        car.set_position(np.array([x, y]))
        car.set_init_position(np.array([x, y]))
        car.set_velocity(np.array([vx, vy]))
        car.set_rotation(theta)
        driver.x_driver.set_v_des(v_des)
        driver.x_driver.set_direction(direction)
        driver.y_driver.set_p_des(p_des)

        driver.set_yld(yld)

        self._cars.append(car)
        self._drivers.append(driver)

        self.car_present[int(idx-1)] = True
        self.car_lane_info[int(idx-1)] = direction

        return car, driver

    # randomly initialize yld for a new car, returns the yld and desired front gap depending on yld
    def init_trait(self, reset=False):
        if reset:
            # todo: hardcoded for now
            if self.yld_prob == 0.67:
                yld_prob = 0.25
            else:
                yld_prob = 0.5
        else:
            yld_prob = self.yld_prob
        if np.random.uniform() < yld_prob:
            new_yld = True
        else:
            new_yld = False
        # determine the range of front gap from the car's class (cons/agg)
        if new_yld:
            gap_min = self.con_gap_min
            gap_max = self.con_gap_max
        else:
            gap_min = self.agg_gap_min
            gap_max = self.agg_gap_max
        return new_yld, gap_min, gap_max

    # true reset function (will be called in reset in traffic_env.py)
    def _reset(self):
        self._collision = False
        self._outroad = False
        self._goal = False
        self._lower_lane_next_idx = 1
        self._upper_lane_next_idx = int(self.max_veh_num/2.)+1


        # whether each car is an actual car (True) or a dummy car (False)
        self.car_present = np.zeros(self.max_veh_num, dtype=bool)
        # whether each car is in lower lane(-1) or upper lane (1) or unknown (0)
        self.car_lane_info = np.zeros(self.max_veh_num, dtype=int)
        # the true class label of each car (updated every 20 steps)
        self.yld_belief = np.zeros(self.max_veh_num)

        # initialize a list of cars and drivers: self._cars, self._drivers
        self._cars, self._drivers = [], []
        # generate the ego car & driver
        car = Car(idx=0, length=self.car_length, width=self.car_width, color=random.choice(BLUE_COLORS),
                          max_accel=self.car_max_accel, max_speed=self.car_max_speed,
                          max_rotation=self.car_max_rotation,
                          expose_level=self.car_expose_level)
        driver = EgoDriver(trajectory=EgoTrajectory(),idx=0,car=car,dt=self.dt)
        car.set_position(np.array([0., -5.0]))
        car.set_velocity(np.array([0., 0.]))
        car.set_rotation(np.pi/2.)
        driver.v_des = 0.
        driver.t_des = 0.
        self._cars.append(car)
        self._drivers.append(driver)

        # generate other cars & drivers
        # randomly generate surrounding cars and drivers, fill as many cars as possible
        # lower lane
        new_yld, gap_min, gap_max = self.init_trait(reset=True)
        x = self.left_bound + np.random.rand() * (gap_max - gap_min)
        while (x < self.right_bound):
            v_des = self.desire_speed + np.random.uniform(-1,1)*self.v_noise
            p_des = 2.
            direction = -1
            self.add_car(x, p_des, -v_des, 0., v_des, p_des, direction, np.pi, new_yld)
            new_yld, gap_min, gap_max = self.init_trait(reset=True)
            x += (np.random.uniform(gap_min,gap_max) + self.car_length)

        # upper lane
        new_yld, gap_min, gap_max = self.init_trait(reset=True)
        x = self.right_bound - np.random.rand()*(gap_max-gap_min)
        while (x > self.left_bound):
            v_des = self.desire_speed + np.random.uniform(-1,1)*self.v_noise
            p_des = 6.
            direction = 1
            self.add_car(x, p_des, v_des, 0., v_des, p_des, direction, 0., new_yld)
            new_yld, gap_min, gap_max = self.init_trait(reset=True)
            x -= (np.random.uniform(gap_min,gap_max) + self.car_length)


    def setup_viewer(self):
        from driving_sim import rendering
        self.viewer = rendering.Viewer(1200, 800)
        self.viewer.set_bounds(-30.0, 30.0, -20.0, 20.0)
        # self.viewer.set_bounds(-50.0, 50.0, -20.0, 20.0)

    def update_extra_render(self, extra_input):
        start = np.array([self.left_bound,4.]) - self.get_camera_center()
        end = np.array([self.right_bound,4.]) - self.get_camera_center()
        attrs = {"color":(1.,1.,1.),"linewidth":2.}
        self.viewer.draw_line(start, end, **attrs)

