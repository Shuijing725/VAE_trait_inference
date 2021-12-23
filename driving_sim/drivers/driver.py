import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from driving_sim.constants import *
from driving_sim.actions.xy_accel_action import XYAccelAction
from driving_sim.actions.trajectory_accel_action import TrajectoryAccelAction

class Driver:
	def __init__(self, idx, car, dt):
		self._idx = idx
		self.car = car
		self.dt = dt

		self.seed()

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def observe(self, cars, road):
		pass

	def get_action(self):
		pass

	def reset(self):
		pass

	def setup_render(self, viewer):
		pass

	def update_render(self, camera_center, collected=False):
		pass

	def remove_render(self, viewer):
		pass

class OneDDriver(Driver):
	def __init__(self, axis, direction=1, **kwargs):
		self.set_axis(axis)
		self.set_direction(direction)
		super(OneDDriver, self).__init__(**kwargs)

	def set_axis(self, axis):
		if axis == 0:
			self.axis0 = 0
			self.axis1 = 1
		else:
			self.axis0 = 1
			self.axis1 = 0

	def set_direction(self,direction):
		self.direction = direction

class XYSeperateDriver(Driver):
	def __init__(self, x_driver, y_driver, **kwargs):
		self.x_driver = x_driver
		self.y_driver = y_driver
		super(XYSeperateDriver, self).__init__(**kwargs)
		assert self.x_driver.car is self.car
		assert self.y_driver.car is self.car

	def observe(self, cars, road):
		self.x_driver.observe(cars, road)
		self.y_driver.observe(cars, road)

	def get_action(self):
		a_x = self.x_driver.get_action()
		a_y = self.y_driver.get_action()
		return XYAccelAction(a_x, a_y)

	def reset(self):
		self.x_driver.reset()
		self.y_driver.reset()


class YNYDriver(XYSeperateDriver):
	def __init__(self, yld=True, t1=1.0, t2=0.,
				s_min=0., v_min=0.5,
				dv_yld=-1.0, dv_nyld=1.0,
				ds_yld=1.0, ds_nyld=-1.0,
				**kwargs):
		self.yld = yld
		self.t1 = t1
		self.t2 = t2
		self.s_min = s_min
		self.v_min = v_min
		self.dv_yld = dv_yld
		self.dv_nyld = dv_nyld
		self.ds_yld = ds_yld
		self.ds_nyld = ds_nyld
		self.v_des_0 = None
		self.s_des_0 = None
		self.intention = 0 # 0: normal drive; 1: yield 2: not yield
		super(YNYDriver, self).__init__(**kwargs)

	# whether the driver yields to
	def set_yld(self, yld):
		self.yld = yld

	def observe(self, cars, road):
		self.x_driver.observe(cars[1:], road)
		v_front = self.x_driver.front_speed
		if self.v_des_0 is None:
			self.v_des_0 = self.x_driver.v_des
		if self.s_des_0 is None:
			self.s_des_0 = self.x_driver.front_distance

		ego_s = cars[0].position[0] - self.car.position[0]
		ego_s = ego_s * self.x_driver.direction
		ego_t = self.car.get_distance(cars[0],1)
		ego_vy = cars[0].velocity[1] * np.sign(self.car.position[1]-cars[0].position[1])
		# print("t: ",t, self.t1, self.t2)
		# conservative cars: desired velocity (v_des) is smaller, desired front gap range is smaller
		if self.yld:
			# desired velocity
			if v_front is None:
				self.x_driver.set_v_des(self.v_des_0 * 0.8)
			else:
				self.x_driver.set_v_des(v_front * 1.5 * 0.8)
				# desired front gap range: 0.5 ~ 0.7
				self.x_driver.s_des = self.s_des_0 * np.random.uniform(0.5,0.7)
		# aggressive cars: desired velocity (v_des) is bigger, desired front gap range is bigger
		else:
			# desired velocity
			if v_front is None:
				self.x_driver.set_v_des(self.v_des_0)
			else:
				self.x_driver.set_v_des(v_front*1.5)
				# desired front gap range: 0.3 ~ 0.5
				self.x_driver.s_des = self.s_des_0 * np.random.uniform(0.3,0.5)

		# if the ego car is far away
		if  (ego_s < self.s_min) or (ego_t > self.t1)\
			 or ((ego_vy <= self.v_min) and (ego_t > self.t2)): # normal drive
			self.x_driver.observe(cars[1:], road)
			self.intention = 0
		# if the ego car is in the front
		else:
			if self.yld: # yield
				self.x_driver.min_overlap = self.t1 # this line does not change anything!!!
				self.x_driver.observe(cars, road)
				self.intention = 1
			else: # not yield
				self.x_driver.observe(cars[1:], road)
				self.intention = 2

		self.y_driver.observe(cars, road)

	def setup_render(self, viewer):
		if self.yld:
			self.car._color = [*GREEN_COLORS[0],0.5]
		else:
			self.car._color = [*RED_COLORS[0],0.5]
		self.car._arr_color = [0.8, 0.8, 0.8, 0.5]

	def update_render(self, camera_center, collected=False):
		if self.yld:
			self.car._color = [*GREEN_COLORS[0],0.5]
		else:
			self.car._color = [*RED_COLORS[0],0.5]
		if collected:
			self.car._color = [*YELLOW_COLOR, 0.5]
		self.car._arr_color = [0.8, 0.8, 0.8, 0.5]

class EgoDriver(Driver):
	def __init__(self,
				trajectory=None,
				v_des=0.0,
				t_des=0.0,
				k_s_p=2.0,
				k_t_p=2.0,
				k_t_d=2.0,
				sigma=0.0,
				as_max=3.0,
				at_max=3.0,
				as_max_safe=6.0,
				at_max_safe=6.0,
				concern_distance=1.0,
				safe_distance=0.8,
				safe_speed=1.0,
				**kwargs):

		self.trajectory = trajectory
		self.v_des = v_des
		self.t_des = t_des
		self.k_s_p = k_s_p
		self.k_t_p = k_t_p
		self.k_t_d = k_t_d
		self.as_max = as_max
		self.at_max = at_max
		self.as_max_safe = as_max_safe
		self.at_max_safe = at_max_safe

		self.a_s = None
		self.a_t = None
		super(EgoDriver, self).__init__(**kwargs)
		# np.sqrt(self.car.length**2+self.car.width**2)/2
		self.concern_distance = concern_distance
		self.safe_distance = safe_distance
		self.safe_speed = safe_speed
		self.k_d_safe = 5.0
		self.k_v_safe = 5.0 # 2.0

	def set_trajectory(self, trajectory):
		self.trajectory = trajectory

	def observe(self, cars, road):
		s, t, theta, curv = self.trajectory.xy_to_traj(self.car.position)
		v_x, v_y = self.car.velocity[0], self.car.velocity[1]
		v_s = v_x*np.cos(theta) + v_y*np.sin(theta)
		v_t = -v_x*np.sin(theta) + v_y*np.cos(theta)

		self.a_s = self.k_s_p*(self.v_des-v_s)
		self.a_t = self.k_t_p*(self.t_des-t) - self.k_t_d*v_t
		self.a_s = np.clip(self.a_s,-self.as_max,self.as_max)
		self.a_t = np.clip(self.a_t,-self.at_max,self.at_max)

		# safety check
		a_x_safe = 0.
		a_y_safe = 0.
		unsafe = False
		for cid, car in enumerate(cars):
			if car is self.car:
				continue
			else:
				p1, p2 = self.car.get_closest_points(car)
				distance = np.linalg.norm(p1-p2)
				direction = (p1-p2)/distance
				v_rel = self.car.velocity - car.velocity
				speed_rel = np.sum(v_rel * direction)
				# print(distance)
				if distance < self.concern_distance:
					if distance < self.safe_distance:
						unsafe = True
					elif speed_rel < -self.safe_speed:
						unsafe = True
		if unsafe:
			self.a_s = -self.k_v_safe * v_s
			self.a_t = -self.k_v_safe * v_t
			self.a_s = np.clip(self.a_s,-self.as_max_safe,self.as_max_safe)
			self.a_t = np.clip(self.a_t,-self.at_max_safe,self.at_max_safe)
			# print('emergency brake!', self.a_s, self.a_t)

	def get_action(self):
		return TrajectoryAccelAction(self.a_s, self.a_t, self.trajectory)




