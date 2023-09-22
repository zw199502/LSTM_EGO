import numpy as np
import random
from math import hypot, fabs, pow

class DWA:
	def __init__(self):
		self.name = 'DWA'
		self.max_speed = 1.0
		self.time_horizon = 4
		self.time_step = 0.2
		self.obstacle_dist_penalty = 0.2
		self.speed_penalty = 0.1
		self.time_gamma = 0.9
		self.speed_set = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, \
			     					0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
	
	def collision_detect(self, robot_state, human_states):
		min_dist = 9999.9
		for human_state in human_states:
			dist = hypot(robot_state[0] - human_state[0], robot_state[1] - human_state[1])
			if (dist < robot_state[2] + human_state[2]):
				return True, -1
			if (dist - (robot_state[2] + human_state[2]) < min_dist):
				min_dist = dist - (robot_state[2] + human_state[2])
		return False, min_dist

	def predict(self, state):
		robot = state.self_state
		humans = state.human_states
		preference_speed = np.array([robot.gx - robot.px, robot.gy - robot.py]) / self.time_step
		preference_speed = np.clip(preference_speed, -self.max_speed, self.max_speed)
		# print(preference_speed)
		cost_min = 9999.9
		output = np.zeros(2)
		for x_vel in self.speed_set:
			for y_vel in self.speed_set:
				speed_cost = self.speed_penalty * \
					        (fabs(x_vel - preference_speed[0]) + fabs(y_vel - preference_speed[1]))
				obstacle_cost = 0.0
				done_next = False
				for i in range(self.time_horizon):
					robot_state_time = (robot.px + self.time_step * (i + 1) * x_vel, \
			 							robot.py + self.time_step * (i + 1) * y_vel, \
										robot.radius)
					human_states_time = []
					for human in humans:
						human_state_time = (human.px + self.time_step * (i + 1) * human.vx, \
			 							    human.py + self.time_step * (i + 1) * human.vy, \
										    human.radius)
						human_states_time.append(human_state_time)
					collision, min_dist = self.collision_detect(robot_state_time, human_states_time)
					if collision:
						done_next = True
						break
					else:
						if min_dist <= self.time_step * self.max_speed:
							obstacle_cost = obstacle_cost + pow(self.time_gamma, i) * \
						                (1.0 - min_dist / (self.time_step * self.max_speed)) * \
										self.obstacle_dist_penalty
				if done_next:
					break
				else:
					cost_all = speed_cost + obstacle_cost
					if cost_all < cost_min:
						cost_min = cost_all
						output[0] = x_vel
						output[1] = y_vel
		return output
			

		