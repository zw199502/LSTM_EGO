import logging
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
from matplotlib import collections as mc
from numpy.linalg import norm
from utils.human import Human
from utils.robot import Robot
from utils.state import *
from policy.policy_factory import policy_factory
from info import *
from math import atan2, hypot, sqrt, cos, sin, fabs, inf, ceil
from time import sleep, time
from C_library.motion_plan_lib import *
import pyastar2d
from gym import spaces

# laser scan parameters
# number of all laser beams
n_laser = 1800
laser_angle_resolute = 0.003490659
laser_min_range = 0.27
laser_max_range = 6.0

# environment size
square_width = 10.0
# environment size

### cost map ###
map_size = 200
map_margin = 5
map_resolution = 0.05
### cost map ###

class CrowdSim:
    def __init__(self):
        self.human_policy_name = 'orca' # human policy is fixed orca policy
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None

        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        self.goal_distance_factor = None

        # last-time distance from the robot to the goal
        self.goal_distance_last = None

        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.lines = None # environment margin
        self.circles = None # human margin
        self.circle_radius = None
        self.human_num = None

        # scan_intersection, each line connects the robot and the end of each laser beam
        self.scan_intersection = None # used for visualization

        self.action_space = spaces.Box(low = -1, high = 1, dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(19202,), dtype=np.float32)

        # laser state
        self.sequential_scans = []
        self.scan_current = np.zeros(n_laser, dtype=np.float32)
        self.scan_end_current = np.zeros((n_laser, 2), dtype=np.float32)

        # cost map
        self.cost_map = np.zeros((map_size, map_size), dtype=np.int32)

        plt.ion()
        plt.show()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.count = 0
        
    def configure(self, reward_simple=False):
        self.reward_simple = reward_simple
        self.time_limit = 20
        self.time_step = 0.2
        self.randomize_attributes = False
        self.success_reward = 1.0
        self.collision_penalty = -0.3
        self.discomfort_dist = 0.2
        self.discomfort_penalty_factor = 0.5
        self.goal_distance_factor = 0.1
       
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': 100, 'test': 500}

        # margin = square_width / 2.0
        margin = 35.0  
        # here, more lines can be added to simulate obstacles
        self.lines = [[(-margin, -margin), (-margin,  margin)], \
                        [(-margin,  margin), ( margin,  margin)], \
                        [( margin,  margin), ( margin, -margin)], \
                        [( margin, -margin), (-margin, -margin)]]
        self.circle_radius = 4.0
        self.human_num = 5

        # self.circle_radius = 10.0
        # self.human_num = 25

        self.robot = Robot()
        self.robot.time_step = self.time_step

        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Square width: {}, circle width: {}'.format(square_width, self.circle_radius))
        
    def generate_random_human_position(self):
        # initial min separation distance to avoid danger penalty at beginning
        self.humans = []
        for i in range(self.human_num):
            self.humans.append(self.generate_circle_crossing_human())

        for i in range(len(self.humans)):
            human_policy = policy_factory[self.human_policy_name]()
            human_policy.time_step = self.time_step
            self.humans[i].set_policy(human_policy)

    def generate_circle_crossing_human(self):
        human = Human()
        human.time_step = self.time_step

        if self.randomize_attributes:
            # Sample agent radius and v_pref attribute from certain distribution
            human.sample_random_attributes()
        else:
            human.radius = 0.3
            human.v_pref = 1.0
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        # px, py, gx, gy, vx, vy, theta
        human.set(px, py, -px, -py, 0, 0, 0)
        return human
    
    def position_to_grid(self, position):
        grid_x = int((map_margin - position[0]) / map_resolution)
        grid_y = int((map_margin - position[1]) / map_resolution)
        return (grid_x, grid_y)
    
    def grid_to_position(self, grid):
        x = map_margin - grid[0] * map_resolution
        y = map_margin - grid[1] * map_resolution
        return (x, y)
    
    def create_cost_map_python(self, scan_data):
        grid_map = 255 * np.ones((map_size, map_size), dtype=np.uint8)
        
        for i in range(n_laser):
            beta = (i - (n_laser - 1.0) / 2.0) * laser_angle_resolute
            x_robot_frame = scan_data[i] * cos(beta)
            y_robot_frame = scan_data[i] * sin(beta)
            c_theta = cos(self.robot.theta)
            s_theta = sin(self.robot.theta)
            x_odom_frame = self.robot.px + x_robot_frame * c_theta - y_robot_frame * s_theta
            y_odom_frame = self.robot.py + x_robot_frame * s_theta + y_robot_frame * c_theta
            x_grid_map_frame = map_margin - x_odom_frame
            y_grid_map_frame = map_margin - y_odom_frame
            x_grid = int(x_grid_map_frame / map_resolution)
            y_grid = int(y_grid_map_frame / map_resolution)
            if (0 <= x_grid < map_size) and (0 <= y_grid < map_size):
                grid_map[x_grid, y_grid] = 0
        # dilate
        dst = 255 * np.ones((map_size + 22, map_size + 22), dtype=np.uint8)
        
        layer = np.array([[ 0,  0,  0,  0,  0,  3,  4,  5,  5,  5],
                          [ 0,  0,  0,  0,  3,  4,  5,  6,  6,  6],
                          [ 0,  0,  0,  3,  5,  6,  6,  7,  7,  7],
                          [ 0,  0,  4,  5,  6,  7,  8,  8,  8,  8],
                          [ 0,  4,  5,  7,  7,  8,  9,  9,  9,  9],
                          [ 4,  6,  7,  8,  9,  9, 10, 10, 10, 10]])
        for i in range(map_size):
            for j in range(map_size):
                if grid_map[i, j] == 0:
                    # up, dowm, left, right
                    dst[i + 11 - 6 : i + 11 + 6 + 1, j + 11] = 0
                    dst[i + 11, j + 11 - 6 : j + 11 + 6 + 1] = 0
                    for m in range(7, 12):
                        grid_value = m * m
                        if dst[i + 11 - m, j + 11] > grid_value:
                            dst[i + 11 - m, j + 11] = grid_value
                        if dst[i + 11 + m, j + 11] > grid_value:
                            dst[i + 11 + m, j + 11] = grid_value
                        if dst[i + 11, j + 11 - m] > grid_value:
                            dst[i + 11, j + 11 - m] = grid_value
                        if dst[i + 11, j + 11 + m] > grid_value:
                            dst[i + 11, j + 11 + m] = grid_value
                    # others
                    for i1 in range(layer.shape[0]):
                        for j1 in range(layer.shape[1]):
                            if i1 == 0 and layer[i1, j1] != 0:
                                dst[i + 11 - layer[i1, j1] : i + 11 + layer[i1, j1] + 1, j + 11 - (layer.shape[1] - j1)] = 0
                                dst[i + 11 - layer[i1, j1] : i + 11 + layer[i1, j1] + 1, j + 11 + (layer.shape[1] - j1)] = 0
                            if i1 > 0 and layer[i1, j1] != 0:
                                for n in range(layer[i1 - 1, j1] + 1, layer[i1, j1] + 1):
                                    grid_value = (6 + i1) * (6 + i1)
                                    if dst[i + 11 - n, j + 11 - (layer.shape[1] - j1)] > grid_value:
                                        dst[i + 11 - n, j + 11 - (layer.shape[1] - j1)] = grid_value
                                    if dst[i + 11 + n, j + 11 - (layer.shape[1] - j1)] > grid_value:
                                        dst[i + 11 + n, j + 11 - (layer.shape[1] - j1)] = grid_value
                                    if dst[i + 11 - n, j + 11 + (layer.shape[1] - j1)] > grid_value:
                                        dst[i + 11 - n, j + 11 + (layer.shape[1] - j1)] = grid_value
                                    if dst[i + 11 + n, j + 11 + (layer.shape[1] - j1)] > grid_value:
                                        dst[i + 11 + n, j + 11 + (layer.shape[1] - j1)] = grid_value
                       
        self.cost_map = dst[11 : 11 + map_size, 11 : 11 + map_size]
    
    def get_cost_map(self, cython=True, scan_data=None):
        if cython:
            create_cost_map()
            for i in range(map_size):
                for j in range(map_size):
                    self.cost_map[i, j] = get_cost_map(j + i * map_size)
        else:
            self.create_cost_map_python(scan_data)

    def path_finding(self):
        start = (self.robot.px, self.robot.py)
        end = (self.robot.gx, self.robot.gy)
        start_grid = self.position_to_grid(start)
        end_grid = self.position_to_grid(end)
        weights = 255 - self.cost_map + 1
        weights = weights.astype(np.float32)
        path_find_time = time()
        path = pyastar2d.astar_path(weights, start_grid, end_grid, allow_diagonal=False)
        # print('path finding time: ', time() - path_find_time)
        return path

    def get_lidar(self, isReset=False):
        scan = np.zeros(n_laser, dtype=np.float32)
        scan_end = np.zeros((n_laser, 2), dtype=np.float32)
        self.circles = np.zeros((self.human_num, 3), dtype=np.float32)
        # here, more circles can be added to simulate obstacles
        for i in range(self.human_num):
            self.circles[i, :] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius])
        robot_pose = np.array([self.robot.px, self.robot.py, self.robot.theta])
        num_line = len(self.lines)
        num_circle = self.human_num
        InitializeEnv(num_line, num_circle, n_laser, laser_angle_resolute, map_size, map_resolution, map_margin)
        for i in range (num_line):
            set_lines(4 * i    , self.lines[i][0][0])
            set_lines(4 * i + 1, self.lines[i][0][1])
            set_lines(4 * i + 2, self.lines[i][1][0])
            set_lines(4 * i + 3, self.lines[i][1][1])
        for i in range (num_circle):
            set_circles(3 * i    , self.humans[i].px)
            set_circles(3 * i + 1, self.humans[i].py)
            set_circles(3 * i + 2, self.humans[i].radius)
        set_robot_pose(robot_pose[0], robot_pose[1], robot_pose[2])
        t1 = time()
        cal_laser()
        t2 = time()
        # print('time 5: ', t2 - t1) # 1.8ms
        # print('time 10: ', t2 - t1) # 2.4ms
        # print('time 15: ', t2 - t1) # 2.8ms
        # print('time 20: ', t2 - t1) # 3.1ms
        # print('time 25: ', t2 - t1) # 3.7ms

        self.scan_intersection = []
        for i in range(n_laser):
            scan[i] = get_scan(i)
            scan_end[i, :] = np.array([get_scan_line(4 * i + 2), get_scan_line(4 * i + 3)])
            ### used for visualization
            self.scan_intersection.append([(get_scan_line(4 * i + 0), get_scan_line(4 * i + 1)), \
                                           (get_scan_line(4 * i + 2), get_scan_line(4 * i + 3))])
            ### used for visualization
        # print(scan[1000:1100])
        #### proposed method ####
        self.scan_current = np.clip(scan, laser_min_range, laser_max_range) / laser_max_range
        self.scan_end_current = np.copy(scan_end)
        
        t_mapping = time()
        self.get_cost_map() # cython
        # self.get_cost_map(cython=False, scan_data=scan) # python
        # print('time cython mapping: ', time() - t_mapping)
        # print('time python mapping: ', time() - t_mapping)

        ReleaseEnv() 

    def get_observation(self):
        # scan map
        scan_avg = np.zeros((20,80))
        for i in range(10):
            for j in range(80):
                scan_avg[2 * i, j] = np.min(self.sequential_scans[i][j * 22:(j + 1) * 22])
                scan_avg[2 * i + 1, j] = np.mean(self.sequential_scans[i][j * 22:(j + 1) * 22])
        scan_avg = scan_avg.reshape(1600)
        scan_avg_map = np.matlib.repmat(scan_avg, 1, 4)
        obs_scan = scan_avg_map.reshape(6400)

        # human map
        human_pos_map_tmp = np.zeros((2,80,80))  # cartesian velocity map
        x_robot = self.robot.px
        y_robot = self.robot.py
        theta_robot = self.robot.theta
        for human in self.humans:
            x_human = human.px
            y_human = human.py
            vx_human = human.vx
            vy_human = human.vy
            x_in_robot = (x_human - x_robot) * cos(theta_robot) + (y_human - y_robot) * sin(theta_robot)
            y_in_robot = (y_human - y_robot) * cos(theta_robot) - (x_human - x_robot) * sin(theta_robot)
            vx_in_robot = vx_human * cos(theta_robot) + vy_human * sin(theta_robot)
            vy_in_robot = vy_human * cos(theta_robot) - vx_human * sin(theta_robot)
            if(np.abs(x_in_robot) <= 10 and np.abs(y_in_robot) <= 10):
                # bin size: 0.25 m
                c = int(np.floor(-(y_in_robot-10)/0.25))
                r = int(np.floor(-(x_in_robot-10)/0.25))

                if(r == 80):
                    r = r - 1
                if(c == 80):
                    c = c - 1
                # cartesian velocity map
                human_pos_map_tmp[0,r,c] = vx_in_robot
                human_pos_map_tmp[1,r,c] = vy_in_robot
        obs_human = human_pos_map_tmp.reshape(12800)

        # goal position
        goal_x_in_robot = (self.robot.gx - x_robot) * cos(theta_robot) + (self.robot.gy - y_robot) * sin(theta_robot)
        goal_y_in_robot = (self.robot.gy - y_robot) * cos(theta_robot) - (self.robot.px - x_robot) * sin(theta_robot)
        obs_goal = np.array([goal_x_in_robot, goal_y_in_robot]) / square_width # normalization

        obs = np.concatenate((obs_human, obs_scan, obs_goal), axis=None)
        return obs

    def reset(self, phase='test'):
        assert phase in ['train', 'val', 'test']
        self.global_time = 0

        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                            'val': 0, 'test': self.case_capacity['val']}
        # px, py, gx, gy, vx, vy, theta
        self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
        self.goal_distance_last = self.robot.get_goal_distance()
        
        if self.case_counter[phase] >= 0:
            # for every training/valuation/test, generate same initial human states
            np.random.seed(counter_offset[phase] + self.case_counter[phase])
            self.generate_random_human_position()
    
            # case_counter is always between 0 and case_size[phase]
            self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
        else:
            raise NotImplementedError

        self.get_lidar(isReset=True)

        self.sequential_scans = []
        for i in range (10):
            self.sequential_scans.append(self.scan_current)

        # get the observation
        obs = self.get_observation()
        return obs


    def step(self, action):
        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            human_actions.append(human.act(ob))

        # uodate states
        robot_x, robot_y, robot_theta = action[0], action[1], self.robot.theta
        robot_v = [(robot_x - self.robot.px) / self.time_step, (robot_y - self.robot.py) / self.time_step]
        self.robot.update_states(robot_x, robot_y, robot_theta, robot_v)
        for i, human_action in enumerate(human_actions):
            self.humans[i].update_states(human_action)

        # t1 = time()
        # get new laser scan and grid map
        self.get_lidar()  
        # t2 = time()
        # print('time 5: ', t2 - t1)
        del self.sequential_scans[0]
        self.sequential_scans.append(self.scan_current)
        self.global_time += self.time_step
        
        # if reaching goal
        goal_dist = hypot(robot_x - self.robot.gx, robot_y - self.robot.gy)
        reaching_goal = goal_dist < self.robot.radius

        # collision detection between humans
        for i in range(self.human_num):
            for j in range(i + 1, self.human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # collision detection between the robot and humans
        collision = False
        dmin = (self.scan_current * laser_max_range).min()
        if dmin <= self.robot.radius:
            collision = True

        reward = 0
        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif ((dmin - self.robot.radius) < self.discomfort_dist):
            # penalize agent for getting too close 
            reward = (dmin - self.robot.radius - self.discomfort_dist) * self.discomfort_penalty_factor
            done = False
            info = Danger(dmin)
        else:
            reward = 0
            done = False
            info = Nothing()

        if not done:
            reward = reward + self.goal_distance_factor * (self.goal_distance_last - goal_dist)
        self.goal_distance_last = goal_dist
  
        for i, human in enumerate(self.humans):
            # let humans move circularly from two points
            if human.reached_destination():
                self.humans[i].gx = -self.humans[i].gx
                self.humans[i].gy = -self.humans[i].gy

        # get the observation
        obs = self.get_observation()
        return obs, reward, done, info
    
    def plot_grid_map(self, alpha=1, min_val=0, origin='upper'):
        plt.imshow(self.cost_map, vmin=min_val, vmax=1, origin=origin, interpolation='none', alpha=alpha)

    def plot_path(self, path):
        start_x, start_y = path[0]
        goal_x, goal_y = path[-1]
        # plot path
        path_arr = np.array(path)
        plt.plot(path_arr[:, 1], path_arr[:, 0], 'y')

        # plot start point
        plt.plot(start_y, start_x, 'ro')

        # plot goal point
        plt.plot(goal_y, goal_x, 'go')

    def render(self):
        self.ax.set_xlim(-5.0, 5.0)
        self.ax.set_ylim(-5.0, 5.0)
        self.ax.set_xlabel('x/m', fontproperties = 'Times New Roman', fontsize=16)
        self.ax.set_ylabel('y/m', fontproperties = 'Times New Roman', fontsize=16)
        for human in self.humans:
            human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
            self.ax.add_artist(human_circle)
        self.ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
        plt.xticks(fontname = "Times New Roman", fontsize=16)
        plt.yticks(fontname = "Times New Roman", fontsize=16)

        # plt.savefig('test_data/' + str(self.count) + '.pdf')  
        self.count = self.count + 1
        plt.draw()
        plt.pause(0.001)
        plt.cla()

# if __name__ == '__main__':
#     my_move_base = MoveBase()
#     my_move_base.configure()
#     my_move_base.reset()
#     # my_move_base.render()
#     for i in range(40):
#         my_move_base.step(np.arrays([my_move_base.robot.px, my_move_base.robot.py]))
#         path = my_move_base.path_finding()
#         my_move_base.plot_grid_map()
#         my_move_base.plot_path(path)
#         plt.show()
#     # sleep(10)