import numpy as np
import gym


# Abstract environment class, only used for inheritance & does not work alone!
class TrafficEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self):

        self.dt = None

        self._viewer = None

        self._road = None

        self._cars = []

        self._drivers = []

        # used to seperate training seeds and testing seeds
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 1000, 'test': 1000}
        self.case_size = {'train': np.iinfo(np.uint32).max - 1000, 'test': 1000}
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        self.thisSeed = 0 # will be set in make_env function in envs.py
        self.global_time = None
        self.step_counter = None
        self.nenv = None # will be set in make_env function in envs.py

        ########################
        # self.yld_front_gaps = {}
        # self.not_yld_front_gaps = {}
        ########################

    @property
    def observation_space(self):
        raise NotImplementedError

    @property
    def action_space(self):
        raise NotImplementedError

    def reset(self, phase='train'):
        # set the seed
        counter_offset = {'train': self.case_capacity['test'], 'test': 0}
        np.random.seed(counter_offset[phase] + self.case_counter[phase] + self.thisSeed)
        # print(counter_offset[phase] + self.case_counter[phase] + self.thisSeed)
        # case size is used to make sure that the case_counter is always between 0 and case_size[phase]
        self.case_counter[phase] = (self.case_counter[phase] + int(1 * self.nenv)) % self.case_size[phase]

        # initialize time counter
        self.global_time = 0.
        self.step_counter = 0

        # call the true reset function in the inherited environments
        self._reset()
        self.close()

        # find the observation and returns it
        return self.observe()

    def _reset(self):
        raise NotImplementedError

    def step(self, action):
        # to make it work with vec env wrapper and without the wrapper
        if isinstance(action, np.ndarray):
            action = action[0]

        # apply the ego car's action, compute the apply other cars' action to update all cars' states
        self.update(action)

        # update the current time and current step number
        self.global_time += self.dt
        self.step_counter = self.step_counter + 1

        obs = self.observe()

        reward = self.get_reward()

        done = self.is_terminal()

        info = self.get_info()

        return obs, reward, done, info

    def update(self, action):
        for driver in self._drivers:
            driver.observe(self._cars, self._road)
        self._actions = [driver.get_action() for driver in self._drivers]
        [action.update(car, self.dt) for (car, action) in zip(self._cars, self._actions)]

    def observe(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

    def get_info(self):
        raise NotImplementedError

    def is_terminal(self):
        raise NotImplementedError

    def setup_viewer(self):
        from driving_sim import rendering
        self.viewer = rendering.Viewer(800, 800)
        self.viewer.set_bounds(-20.0, 20.0, -20.0, 20.0)


    def update_extra_render(self, extra_input):
        pass

    def get_camera_center(self):
        return self._cars[0].position

    # default render function that will be used in its children classes
    def render(self, mode='human', screen_size=800, extra_input=None):
        if (not hasattr(self, 'viewer')) or (self.viewer is None):
            self.setup_viewer()

            self._road.setup_render(self.viewer)

            for driver in self._drivers:
                driver.setup_render(self.viewer)

            for car in self._cars:
                car.setup_render(self.viewer)


        camera_center = self.get_camera_center()
        self._road.update_render(camera_center)

        # infer_mask = self.fill_infer_masks()
        for driver in self._drivers:
            # visualize the cars that satisfy the data collection condition
            # idx = int(driver._idx - 1)
            # collected =infer_mask[idx]
            # driver.update_render(camera_center, collected=collected)
            driver.update_render(camera_center, collected=False)
            # driver.update_render(camera_center)

        for cid, car in enumerate(self._cars):
            car.update_render(camera_center)

        self.update_extra_render(extra_input)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if hasattr(self, 'viewer') and self.viewer:
            self.viewer.close()
            self.viewer = None
